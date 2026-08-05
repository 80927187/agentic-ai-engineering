"""
人在回路——“审批关卡”

演示如何在关键检查点暂停智能体工作流，以便人工审核。
LLM 起草邮件，人工批准或拒绝并提供反馈，随后 LLM 根据反馈修改——
这个过程展示了人工监督在何处最有价值。
"""

from collections.abc import Callable

import anthropic
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel

from common import AnthropicTokenTracker, interactive_menu, setup_logging

load_dotenv(find_dotenv())
logger = setup_logging(__name__)

MODEL = "deepseek-v4-flash"
MAX_REVISIONS = 2

SUGGESTED_SCENARIOS = [
    "礼貌地拒绝工作邀请——表达感谢，但已选择另一个机会",
    "请求团队本周末加班——截止日期紧迫，语气诚恳并带有歉意",
    "请求与副总裁开会讨论预算——正式、以数据为依据",
    "跟进一份尚未收到回复的提案——坚持但保持尊重",
]

# --- 提示词 ---

SYSTEM_PROMPT = (
    "你是一名专业的邮件撰稿人。请根据要求的语气撰写清晰、简洁的中文邮件。"
    "只输出邮件内容——先写主题，再写正文。不要添加元说明。全文不超过 300 字。"
)

REVISE_SYSTEM_PROMPT = (
    "你是一名专业的邮件撰稿人。请根据收到的反馈修改邮件。"
    "只返回修改后的邮件——先写主题，再写正文。不要解释修改内容。"
)

# 检查点函数类型：(标题, 内容, 问题) -> (是否批准, 反馈)
CheckpointFn = Callable[[str, str, str], tuple[bool, str]]


class EmailDrafter:
    """通过人工检查点起草和修改邮件。"""

    def __init__(self, model: str, token_tracker: AnthropicTokenTracker) -> None:
        self.client = anthropic.Anthropic()
        self.model = model
        self.token_tracker = token_tracker

    def _call_llm(self, system: str, user_msg: str, *, max_tokens: int = 21333) -> str:
        """调用 LLM 并返回文本响应。"""
        logger.info("正在调用 %s", self.model)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        self.token_tracker.track(response.usage)
        text_parts = [block.text for block in response.content if block.type == "text"]
        if not text_parts:
            block_types = [block.type for block in response.content]
            raise ValueError(
                f"模型响应中没有文本内容（stop_reason={response.stop_reason}，"
                f"内容块={block_types}，output_tokens={response.usage.output_tokens}）。"
            )
        return "\n\n".join(text_parts)

    def _draft(self, scenario: str) -> str:
        """根据场景描述生成邮件初稿。"""
        return self._call_llm(SYSTEM_PROMPT, f"请为以下场景撰写一封邮件：{scenario}")

    def _revise(self, draft: str, feedback: str) -> str:
        """根据人工反馈修改草稿。"""
        user_msg = f"原始邮件：\n{draft}\n\n需要处理的反馈：\n{feedback}"
        return self._call_llm(REVISE_SYSTEM_PROMPT, user_msg)

    def run(self, scenario: str, *, checkpoint_fn: CheckpointFn | None = None) -> str:
        """起草邮件，并通过人工检查点进行审核。"""
        check = checkpoint_fn or (lambda _t, _c, _q: (True, ""))

        # 步骤 1：生成初稿
        logger.info("正在为以下场景生成草稿：%s", scenario)
        draft = self._draft(scenario)
        self.token_tracker.report()

        # === 检查点 1：审核草稿（高杠杆——尽早发现方向错误）===
        approved, feedback = check(
            "草稿审核",
            draft,
            "这封邮件是否合适？批准即可定稿，也可以拒绝并提供反馈。",
        )

        if approved and not feedback:
            return draft
        # 编辑模式：人工提供了替换文本
        if approved and feedback:
            return feedback

        # 已拒绝：进入修改循环
        for revision in range(1, MAX_REVISIONS + 1):
            logger.info("正在修改草稿（第 %d/%d 轮）", revision, MAX_REVISIONS)
            draft = self._revise(draft, feedback)
            self.token_tracker.report()

            # === 检查点 2：审核修改稿 ===
            approved, feedback = check(
                f"修改稿审核（{revision}/{MAX_REVISIONS}）",
                draft,
                "修改后是否更合适？批准即可定稿，也可以拒绝并提供更多反馈。",
            )

            if approved and not feedback:
                return draft
            if approved and feedback:
                return feedback

        logger.info("已达到最大修改轮数，返回最后一版草稿")
        return draft


def human_checkpoint(console: Console, title: str, content: str, question: str) -> tuple[bool, str]:
    """暂停并等待人工审核。返回（是否批准, 反馈）。"""
    console.print(Panel(content, title=f"检查点：{title}", border_style="bright_magenta"))
    console.print(f"\n[bold magenta]{question}[/bold magenta]")
    console.print("[dim](y) 批准 / (n) 拒绝并反馈 / (e) 编辑并提供替换内容[/dim]")
    console.print("[bold magenta]> [/bold magenta]", end="")

    response = input().strip().lower()

    if response in ("y", "yes", ""):
        return True, ""
    elif response.startswith("e"):
        console.print("[dim]请输入替换内容（连续按两次 Enter 结束）：[/dim]")
        lines: list[str] = []
        empty = 0
        while empty < 1:
            line = input()
            if line.strip() == "":
                empty += 1
            else:
                empty = 0
                lines.append(line)
        return True, "\n".join(lines)
    else:
        console.print("[dim]请输入反馈：[/dim] ", end="")
        feedback = input().strip()
        return False, feedback


def main() -> None:
    """运行人在回路邮件起草示例。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()

    def checkpoint_fn(title: str, content: str, question: str) -> tuple[bool, str]:
        return human_checkpoint(console, title, content, question)

    header = Panel(
        "[bold cyan]人在回路——审批关卡[/bold cyan]\n\n"
        "LLM 起草邮件 → 你在检查点进行审核：\n"
        "1. 草稿完成后——语气和内容是否合适？\n"
        "2. 修改完成后——反馈是否得到落实？\n\n"
        "选项：(y) 批准、(n) 拒绝并反馈、(e) 编辑替换内容\n"
        f"每封邮件最多修改 {MAX_REVISIONS} 轮。",
        title="人在回路",
    )

    try:
        while True:
            scenario = interactive_menu(
                console,
                SUGGESTED_SCENARIOS,
                title="选择邮件场景",
                header=header,
                allow_custom=True,
                custom_prompt="请描述你的邮件场景",
            )
            if not scenario:
                break

            console.print(f"\n[bold green]场景：[/bold green] {scenario}")
            drafter = EmailDrafter(MODEL, token_tracker)

            try:
                result = drafter.run(scenario, checkpoint_fn=checkpoint_fn)

                console.print("\n[bold blue]最终邮件：[/bold blue]")
                console.print(Panel(result, border_style="green"))

                console.print("\n[dim]按 Enter 继续……[/dim]")
                input()
            except Exception as e:
                logger.error("邮件起草失败：%s", e)
                console.print(f"\n[red]错误：{e}[/red]")
            finally:
                token_tracker.reset()

    except KeyboardInterrupt:
        console.print("\n[yellow]操作已中断。[/yellow]")


if __name__ == "__main__":
    main()
