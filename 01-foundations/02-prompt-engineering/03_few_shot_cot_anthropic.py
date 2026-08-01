"""
少样本与思维链提示（Anthropic）

演示三种提示技术，每种技术都用于最适合它的任务：
1. 零样本——情感分析（任务定义清晰，无需示例）
2. 少样本——使用自定义领域标签分类（教授你自己的分类体系）
3. 思维链——根本原因分析（需要多步推理）

每个演示都会说明为什么要选择该技术而不是其他技术。
"""

import anthropic
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from common import AnthropicTokenTracker, interactive_menu, setup_logging

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

# --- 演示 A：零样本（情感分析）---
# 当模型已经充分理解任务时，零样本方法效果很好
REVIEWS = [
    "这台笔记本电脑太棒了——速度快、重量轻，而且电池能用一整天。",
    "充电线用了两周就坏了。完全是在浪费钱。",
    "以这个价格来说还行。没什么特别之处，但能满足需求。",
]

# --- 演示 B：少样本（自定义领域标签）---
# 少样本示例向模型教授它原本不知道的自定义类别
FEW_SHOT_EXAMPLES = [
    ("同一项订阅向我收取了两次费用", "BILLING_DISPUTE"),
    ("重置三次密码后仍然无法登录", "ACCOUNT_ACCESS"),
    ("报告超过 1000 行时，导出功能会崩溃", "TECHNICAL_BUG"),
    ("如果可以安排报告自动运行就太好了", "FEATURE_REQUEST"),
]

FEW_SHOT_TEST_INPUTS = [
    "我的发票显示了一笔上个月的费用，但我已经对此提出过异议",
    "仪表板一直显示加载动画，始终无法加载图表",
    "希望能为我们团队的工单添加自定义标签",
]

# --- 演示 C：思维链（根本原因分析）---
# 当任务需要多步推理时，思维链方法表现出色
BUG_REPORT = (
    "用户报告称，应用在上午运行正常，但午饭后会变得极其缓慢。"
    "所有用户会同时受到速度下降的影响，而不只是个别会话。"
    "重启应用服务器可以暂时解决问题，但几小时后问题又会出现。"
    "服务器的内存使用情况看起来正常。"
)


class PromptingClient:
    """演示零样本、少样本和思维链提示。"""

    def __init__(self, model: str, token_tracker: AnthropicTokenTracker):
        self.client = anthropic.Anthropic()
        self.model = model
        self.token_tracker = token_tracker

    def _call(self, system_prompt: str, user_content: str, max_tokens: int = 256) -> str:
        """执行一次 API 调用并跟踪 Token 使用量。"""
        response = self.client.messages.create(
            model=self.model,
            temperature=0.0,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        self.token_tracker.track(response.usage)
        return str(response.content[0].text).strip()

    # --- 零样本 ---
    def classify_sentiment(self, review: str) -> str:
        """不提供示例进行情感分类——模型已经理解该任务。"""
        system = (
            "对以下产品评价的情感进行分类。\n"
            "只能用一个词作答：正面、负面或中性。"
        )
        return self._call(system, review)

    # --- 少样本 ---
    def classify_ticket_few_shot(self, ticket: str) -> str:
        """使用模型在没有示例时并不了解的领域特定标签进行分类。"""
        examples = "\n".join(
            f'工单："{text}"\n类别：{label}' for text, label in FEW_SHOT_EXAMPLES
        )
        system = (
            "将支持工单归入以下类别之一："
            "BILLING_DISPUTE, ACCOUNT_ACCESS, TECHNICAL_BUG, FEATURE_REQUEST\n\n"
            f"示例：\n\n{examples}\n\n"
            "只能回答类别名称。"
        )
        return self._call(system, f'工单："{ticket}"\n类别：')

    # --- 思维链 ---
    def analyze_zero_shot(self, bug_report: str) -> str:
        """在没有推理引导的情况下分析缺陷报告——基线。"""
        system = (
            "你是一名高级工程师。请找出该缺陷最可能的根本原因。\n"
            "保持简洁——用一两句话作答。"
        )
        return self._call(system, bug_report)

    def analyze_cot(self, bug_report: str) -> str:
        """使用思维链进行分析——逐步推理问题。"""
        system = (
            "你是一名高级工程师。请逐步分析这份缺陷报告：\n"
            "1. 你观察到了哪些模式？（时间、范围、触发条件）\n"
            "2. 每条线索支持或排除了哪些可能性？\n"
            "3. 最可能的根本原因是什么？\n"
            "4. 你会首先检查什么来验证结论？\n\n"
            "得出结论前，请逐步完成推理。"
        )
        return self._call(system, bug_report, max_tokens=512)


DEMO_LABELS = [
    "A：零样本 — 情感分析",
    "B：少样本 — 自定义标签分类",
    "C：思维链 — 根本原因分析",
]


ZERO_SHOT_SYSTEM = (
    "对以下产品评价的情感进行分类。\n"
    "只能用一个词作答：正面、负面或中性。"
)

FEW_SHOT_SYSTEM_TEMPLATE = (
    "将支持工单归入以下类别之一："
    "BILLING_DISPUTE, ACCOUNT_ACCESS, TECHNICAL_BUG, FEATURE_REQUEST\n\n"
    "示例：\n\n{examples}\n\n"
    "只能回答类别名称。"
)

COT_SYSTEM = (
    "你是一名高级工程师。请逐步分析这份缺陷报告：\n"
    "1. 你观察到了哪些模式？（时间、范围、触发条件）\n"
    "2. 每条线索支持或排除了哪些可能性？\n"
    "3. 最可能的根本原因是什么？\n"
    "4. 你会首先检查什么来验证结论？\n\n"
    "得出结论前，请逐步完成推理。"
)


def _run_zero_shot(console: Console, client: PromptingClient) -> None:
    """运行零样本情感分析演示。"""
    console.print("[dim]无需示例——模型已经理解情感分类。[/dim]\n")
    console.print(Panel(ZERO_SHOT_SYSTEM, title="系统提示词", border_style="dim"))

    sentiment_table = Table(show_lines=True)
    sentiment_table.add_column("评价", style="cyan", max_width=55)
    sentiment_table.add_column("情感", style="green", max_width=12)

    for review in REVIEWS:
        try:
            result = client.classify_sentiment(review)
            sentiment_table.add_row(review, result)
        except Exception as e:
            logger.error("情感分类出错：%s", e)
            sentiment_table.add_row(review, "错误")

    console.print(sentiment_table)


def _run_few_shot(console: Console, client: PromptingClient) -> None:
    """运行少样本自定义标签分类演示。"""
    console.print(
        "[dim]模型并不了解 BILLING_DISPUTE 之类的标签——"
        "示例会教授你的分类体系。[/dim]\n"
    )
    examples = "\n".join(
        f'工单："{text}"\n类别：{label}' for text, label in FEW_SHOT_EXAMPLES
    )
    system_prompt = FEW_SHOT_SYSTEM_TEMPLATE.format(examples=examples)
    console.print(Panel(system_prompt, title="系统提示词", border_style="dim"))

    ticket_table = Table(show_lines=True)
    ticket_table.add_column("支持工单", style="cyan", max_width=55)
    ticket_table.add_column("类别", style="green", max_width=18)

    for ticket in FEW_SHOT_TEST_INPUTS:
        try:
            result = client.classify_ticket_few_shot(ticket)
            ticket_table.add_row(ticket, result)
        except Exception as e:
            logger.error("少样本分类出错：%s", e)
            ticket_table.add_row(ticket, "错误")

    console.print(ticket_table)


def _run_cot(console: Console, client: PromptingClient) -> None:
    """运行思维链根本原因分析演示。"""
    console.print("[dim]对一份需要多步推理的缺陷报告使用思维链。[/dim]\n")
    console.print(Panel(COT_SYSTEM, title="系统提示词", border_style="dim"))
    console.print(Panel(BUG_REPORT, title="用户消息", border_style="dim"))

    try:
        cot = client.analyze_cot(BUG_REPORT)
        console.print(Panel(cot, title="思维链分析", border_style="green"))
    except Exception as e:
        logger.error("思维链分析出错：%s", e)


def main() -> None:
    """运行三个演示，展示每种提示技术的适用场景。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()
    client = PromptingClient("claude-sonnet-4-6", token_tracker)

    header = Panel(
        "[bold cyan]少样本与思维链提示[/bold cyan]\n\n"
        "三个演示，分别在最适合的场景中使用对应技术：\n"
        "  A. 零样本——情感分析（模型已经了解的任务）\n"
        "  B. 少样本——自定义标签分类（教授你自己的分类体系）\n"
        "  C. 思维链——根本原因分析（多步推理）",
        title="提示工程 — Anthropic",
    )

    demos = {
        DEMO_LABELS[0]: _run_zero_shot,
        DEMO_LABELS[1]: _run_few_shot,
        DEMO_LABELS[2]: _run_cot,
    }

    try:
        while True:
            selection = interactive_menu(
                console,
                DEMO_LABELS,
                title="选择一个演示",
                header=header,
            )
            if not selection:
                break

            console.print(f"\n[bold yellow]━━━ {selection} ━━━[/bold yellow]")

            try:
                demos[selection](console, client)
            except Exception as e:
                logger.error("演示出错：%s", e)

            token_tracker.report()
            token_tracker.reset()

            console.print("\n[dim]按 Enter 键继续……[/dim]")
            input()

    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")


if __name__ == "__main__":
    main()
