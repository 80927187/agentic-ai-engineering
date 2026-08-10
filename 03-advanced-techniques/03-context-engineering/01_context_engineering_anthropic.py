"""
上下文工程（DeepSeek，使用 Anthropic 兼容接口）

演示如何通过令牌计数、预算分配和摘要自动压缩来管理上下文窗口。
程序会人为设置一个较低的上下文预算，以便只进行几轮对话就能触发压缩。
"""

import json
from dataclasses import dataclass
from typing import Any

import anthropic
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from common import AnthropicTokenTracker, setup_logging

# 从根目录的 .env 文件加载环境变量
load_dotenv(find_dotenv())

# 配置日志
logger = setup_logging(__name__)

# 模型配置
MODEL = "deepseek-v4-flash"

SYSTEM_PROMPT = (
    "你是一名知识渊博的研究助理。你会承接之前的讨论要点，帮助用户深入探索主题。"
    "引用对话中的较早内容时，请提及具体细节，以体现对话的连贯性。"
)

# 人为降低预算，让演示能够快速触发压缩
MAX_CONTEXT_TOKENS = 4096
RESPONSE_RESERVE = 2048
RECENT_MESSAGES_TO_KEEP = 4


def _estimate_tokens(*values: Any) -> int:
    """在网关不支持 count_tokens 时，按 UTF-8 字节数近似估算令牌数。"""
    serialized = json.dumps(values, ensure_ascii=False, default=str)
    return max(1, (len(serialized.encode("utf-8")) + 3) // 4)


@dataclass
class ContextBudget:
    """上下文各组成部分的令牌预算分配。"""

    max_context: int
    system_tokens: int = 0
    response_reserve: int = RESPONSE_RESERVE

    @property
    def history_budget(self) -> int:
        """可供对话历史使用的令牌数。"""
        return self.max_context - self.system_tokens - self.response_reserve


@dataclass
class TokenSnapshot:
    """用于预算面板的令牌用量快照。"""

    system: int = 0
    history: int = 0
    history_budget: int = 0
    reserve: int = 0
    message_count: int = 0
    compression_count: int = 0


class ContextManager:
    """管理上下文窗口分配和对话压缩。"""

    def __init__(self, model: str, max_context: int, token_tracker: AnthropicTokenTracker):
        self.client = anthropic.Anthropic()
        self.model = model
        self.token_tracker = token_tracker
        self.messages: list[dict] = []
        self.budget = ContextBudget(max_context=max_context)
        self.compression_count = 0

        # 初始化时只估算一次系统提示词的令牌数
        self.budget.system_tokens = self._count_tokens([])
        logger.info(
            "上下文预算——系统：%d，历史记录：%d，预留：%d",
            self.budget.system_tokens,
            self.budget.history_budget,
            self.budget.response_reserve,
        )

    def chat(self, user_input: str) -> str:
        """发送消息，按需压缩，然后返回响应。"""
        self.messages.append({"role": "user", "content": user_input})

        # 如果历史记录超出预算，则在发送前压缩
        self._compress_if_needed()

        logger.info(
            "正在发送请求（消息数：%d，历史记录令牌数：约 %d/%d）",
            len(self.messages),
            self._count_tokens(self.messages) - self.budget.system_tokens,
            self.budget.history_budget,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.budget.response_reserve,
            system=SYSTEM_PROMPT,
            messages=self.messages,
        )

        self.token_tracker.track(response.usage)

        assistant_message = _extract_text(response)
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def _count_tokens(self, messages: list[dict]) -> int:
        """在 DeepSeek 兼容网关未提供计数端点时，本地估算令牌数。"""
        # 原 Anthropic 实现：使用令牌计数 API 计算令牌数。
        # API 要求至少有一条消息，因此使用最小占位消息来测量系统开销。
        # msgs = messages if messages else [{"role": "user", "content": "."}]
        # result = self.client.messages.count_tokens(
        #     model=self.model,
        #     system=SYSTEM_PROMPT,
        #     messages=msgs,
        # )
        # token_count: int = result.input_tokens
        # return token_count

        msgs = messages if messages else [{"role": "user", "content": "."}]
        return _estimate_tokens(SYSTEM_PROMPT, msgs)

    def _compress_if_needed(self) -> None:
        """如果历史记录超出预算，则总结最早的消息。"""
        history_tokens = self._count_tokens(self.messages) - self.budget.system_tokens

        if history_tokens <= self.budget.history_budget:
            return

        logger.info(
            "历史记录（%d 个令牌）超出预算（%d 个令牌）——正在压缩",
            history_tokens,
            self.budget.history_budget,
        )

        # 拆分消息：近期消息保留原文，其余消息生成摘要
        keep_count = min(RECENT_MESSAGES_TO_KEEP, len(self.messages))
        old_messages = self.messages[:-keep_count] if keep_count > 0 else self.messages
        recent_messages = self.messages[-keep_count:] if keep_count > 0 else []

        if not old_messages:
            logger.warning("没有可压缩的消息——预算可能过小")
            return

        old_tokens = self._count_tokens(old_messages) - self.budget.system_tokens

        # 总结较早的消息
        summary = self._summarize_messages(old_messages)

        # 用摘要替换较早的消息
        summary_message = {
            "role": "user",
            "content": (
                f"[先前对话摘要]\n{summary}\n"
                "[摘要结束——请从这里继续对话]"
            ),
        }

        # 确保角色交替：摘要（用户消息）之后接近期消息
        # 如果近期消息以用户消息开头，则需要插入一条助手确认消息
        if recent_messages and recent_messages[0]["role"] == "user":
            self.messages = [
                summary_message,
                {"role": "assistant", "content": "明白，我已经掌握了对话上下文。"},
                *recent_messages,
            ]
        else:
            self.messages = [summary_message, *recent_messages]

        new_tokens = self._count_tokens(self.messages) - self.budget.system_tokens
        self.compression_count += 1

        logger.info(
            "已压缩 %d 条消息：%d → %d 个令牌（节省 %d 个令牌）",
            len(old_messages),
            old_tokens,
            new_tokens,
            old_tokens - new_tokens,
        )

    def _summarize_messages(self, messages: list[dict]) -> str:
        """使用 LLM 总结一组消息。"""
        # 为摘要模型构建易读的对话记录
        transcript = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}" for m in messages
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=21333,
            system=(
                "简洁地总结以下对话。"
                "保留关键事实、决策以及用户提到的具体细节。"
                "使用第三人称和过去时态。文字应简短但全面。"
            ),
            messages=[{"role": "user", "content": transcript}],
        )

        self.token_tracker.track(response.usage)
        return _extract_text(response)

    def get_token_snapshot(self) -> TokenSnapshot:
        """返回当前令牌计数，用于预算面板。"""
        history_tokens = 0
        if self.messages:
            history_tokens = self._count_tokens(self.messages) - self.budget.system_tokens

        return TokenSnapshot(
            system=self.budget.system_tokens,
            history=history_tokens,
            history_budget=self.budget.history_budget,
            reserve=self.budget.response_reserve,
            message_count=len(self.messages),
            compression_count=self.compression_count,
        )


def _render_budget_display(console: Console, snapshot: TokenSnapshot) -> None:
    """渲染上下文预算可视化面板。"""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("组成部分", style="dim")
    table.add_column("令牌数", justify="right")
    table.add_column("用量", min_width=30)

    # 历史记录用量条
    usage_ratio = snapshot.history / snapshot.history_budget if snapshot.history_budget > 0 else 0
    bar_width = 25
    filled = int(usage_ratio * bar_width)
    bar_color = "green" if usage_ratio < 0.7 else "yellow" if usage_ratio < 0.9 else "red"
    bar = f"[{bar_color}]{'█' * filled}[/{bar_color}][dim]{'░' * (bar_width - filled)}[/dim]"

    table.add_row("系统", f"[cyan]{snapshot.system:,}[/cyan]", "[dim]固定[/dim]")
    table.add_row(
        "历史记录",
        f"[{bar_color}]{snapshot.history:,}[/{bar_color}] / {snapshot.history_budget:,}",
        bar,
    )
    table.add_row("响应预留", f"[cyan]{snapshot.reserve:,}[/cyan]", "[dim]max_tokens[/dim]")
    table.add_row("消息数", f"[cyan]{snapshot.message_count}[/cyan]", "")

    footer = f"消息数：{snapshot.message_count}"
    if snapshot.compression_count > 0:
        footer += f" │ 压缩次数：{snapshot.compression_count}"

    console.print(
        Panel(table, title="上下文预算", subtitle=footer, border_style="dim", padding=(0, 1))
    )


def _extract_text(response: Any) -> str:
    """跳过思考块，只提取模型响应中的文本块。"""
    text_parts = [block.text for block in response.content if block.type == "text"]
    if not text_parts:
        block_types = [block.type for block in response.content]
        raise ValueError(
            f"模型响应中没有文本内容（stop_reason={response.stop_reason}，"
            f"内容块={block_types}，output_tokens={response.usage.output_tokens}）。"
        )
    return "\n\n".join(text_parts)


def main() -> None:
    """上下文工程演示的主编排函数。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()
    manager = ContextManager(MODEL, MAX_CONTEXT_TOKENS, token_tracker)

    console.print(
        Panel(
            "[bold cyan]上下文工程演示[/bold cyan]\n\n"
            "此聊天使用人为设置的较低上下文预算"
            f"（总计 {MAX_CONTEXT_TOKENS:,} 个令牌，"
            f"约 {manager.budget.history_budget:,} 个用于历史记录）。\n"
            "几轮对话后，你会看到自动压缩开始生效——\n"
            "较早的消息会被总结，以确保用量保持在预算内。\n\n"
            "请尝试深入讨论一个话题，并观察预算面板。\n"
            "输入 [bold]'quit'[/bold] 或 [bold]'exit'[/bold] 结束程序。",
            title="研究助理",
        )
    )

    # 显示初始预算
    _render_budget_display(console, manager.get_token_snapshot())

    while True:
        console.print("\n[bold green]你：[/bold green] ", end="")
        user_input = input().strip()

        if user_input.lower() in ["quit", "exit", ""]:
            console.print("\n[yellow]正在结束会话……[/yellow]")
            break

        try:
            response = manager.chat(user_input)

            console.print("\n[bold blue]DeepSeek:[/bold blue]")
            console.print(Markdown(response))

            # 每轮对话后显示预算
            console.print()
            _render_budget_display(console, manager.get_token_snapshot())

        except Exception as e:
            logger.error("聊天期间发生错误：%s", e)
            console.print(f"\n[red]错误：{e}[/red]")
            break

    # 最终报告
    console.print()
    token_tracker.report()
    console.print(
        f"\n[dim]消息数：{len(manager.messages)} │ "
        f"压缩次数：{manager.compression_count}[/dim]"
    )


if __name__ == "__main__":
    main()
