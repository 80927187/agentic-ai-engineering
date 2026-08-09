"""
流式传输基础（Anthropic）

演示如何使用 Claude 逐 Token 实时流式传输。涵盖两种方式：用于快速上手的简单
`.text_stream` 迭代器，以及能够完整控制流式传输生命周期的事件迭代方式。
"""

import anthropic
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from common import AnthropicTokenTracker, setup_logging

# 从根目录的 .env 文件加载环境变量
load_dotenv(find_dotenv())

# 配置日志
logger = setup_logging(__name__)

MODEL = "deepseek-v4-flash"

SYSTEM_PROMPT = (
    "你是一名乐于助人的助手。回答应简洁、结构清晰。"
    "使用 Markdown 格式（标题、项目符号、粗体）提高可读性。"
)


class StreamingChat:
    """实时渲染流式响应的交互式聊天。"""

    def __init__(self, model: str, token_tracker: AnthropicTokenTracker) -> None:
        self.client = anthropic.Anthropic()
        self.model = model
        self.token_tracker = token_tracker
        self.messages: list[dict[str, str]] = []

    def stream_simple(self, user_input: str, console: Console) -> str:
        """使用简单的 .text_stream 迭代器进行流式传输。

        这是最简单的流式传输方式，只需遍历文本分片。
        最适合不需要事件级控制的简单用例。
        """
        self.messages.append({"role": "user", "content": user_input})
        logger.info("正在流式输出响应（简单模式，历史消息数：%d）", len(self.messages))

        accumulated = ""

        with self.client.messages.stream(
            model=self.model,
            max_tokens=21333,
            system=SYSTEM_PROMPT,
            messages=self.messages,
        ) as stream:
            # .text_stream 产出纯文本字符串，也就是内容增量
            with Live(Markdown(""), refresh_per_second=15, console=console) as live:
                for text in stream.text_stream:
                    accumulated += text
                    live.update(Markdown(accumulated))

            # 流完成后即可获取 Token 用量
            final_message = stream.get_final_message()
            self.token_tracker.track(final_message.usage)
            logger.info(
                "流式传输完成——输入：%d Token，输出：%d Token",
                final_message.usage.input_tokens,
                final_message.usage.output_tokens,
            )

        self.messages.append({"role": "assistant", "content": accumulated})
        return accumulated

    def stream_with_events(self, user_input: str, console: Console) -> str:
        """使用事件迭代进行流式传输，以观察完整生命周期。

        这种方式可以访问每一个流式事件，包括内容块开始/停止、文本增量和消息元数据。
        当你需要细粒度控制时使用它，例如检测工具调用或跟踪内容块边界。
        """
        self.messages.append({"role": "user", "content": user_input})
        logger.info("正在流式输出响应（事件模式，历史消息数：%d）", len(self.messages))

        accumulated = ""

        with self.client.messages.stream(
            model=self.model,
            max_tokens=21333,
            system=SYSTEM_PROMPT,
            messages=self.messages,
        ) as stream:
            with Live(Markdown(""), refresh_per_second=15, console=console) as live:
                for event in stream:
                    # --- 事件生命周期 ---
                    # message_start：流开始，包含模型信息
                    # content_block_start：新的内容块（text、tool_use 等）开始
                    # content_block_delta：内容增量更新
                    # content_block_stop：内容块完成
                    # message_delta：顶层变化（stop_reason、usage）
                    # message_stop：流结束

                    if event.type == "content_block_start":
                        logger.debug(
                            "内容块开始：index=%d，type=%s",
                            event.index,
                            event.content_block.type,
                        )

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            accumulated += event.delta.text
                            live.update(Markdown(accumulated))

                    elif event.type == "message_delta":
                        logger.debug("停止原因：%s", event.delta.stop_reason)

            final_message = stream.get_final_message()
            self.token_tracker.track(final_message.usage)
            logger.info(
                "流式传输完成——输入：%d Token，输出：%d Token",
                final_message.usage.input_tokens,
                final_message.usage.output_tokens,
            )

        self.messages.append({"role": "assistant", "content": accumulated})
        return accumulated

    def reset(self) -> None:
        """清空对话历史，以便重新开始。"""
        self.messages.clear()
        logger.info("对话历史已清空")


def main() -> None:
    """可选择模式的交互式流式聊天。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()
    chat = StreamingChat(MODEL, token_tracker)

    console.print(
        Panel(
            "[bold cyan]流式聊天[/bold cyan]\n\n"
            "体验 Claude 逐 Token 的实时流式传输。\n\n"
            "[bold]两种流式传输模式：[/bold]\n"
            "  [green]simple[/green]   — .text_stream 迭代器（最简单，只有文本）\n"
            "  [green]events[/green]   — 事件迭代（完整控制生命周期）\n\n"
            "输入 [bold]mode simple[/bold] 或 [bold]mode events[/bold] 切换模式。\n"
            "输入 [bold]clear[/bold] 清空对话历史。\n"
            "输入 [bold]quit[/bold] 或 [bold]exit[/bold] 结束会话。",
            title="02-streaming / 01 — 流式传输基础",
        )
    )

    mode = "simple"
    console.print(f"\n[dim]当前模式：{mode}[/dim]")

    while True:
        console.print("\n[bold green]你：[/bold green] ", end="")
        try:
            user_input = input().strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]已中断。[/yellow]")
            break

        if not user_input or user_input.lower() in ("quit", "exit"):
            console.print("[yellow]正在结束会话……[/yellow]")
            break

        if user_input.lower() == "clear":
            chat.reset()
            console.print("[dim]对话已清空。[/dim]")
            continue

        if user_input.lower().startswith("mode "):
            new_mode = user_input.split(" ", 1)[1].strip().lower()
            if new_mode in ("simple", "events"):
                mode = new_mode
                console.print(f"[dim]已切换到 {mode} 模式。[/dim]")
            else:
                console.print("[red]未知模式。请使用 'simple' 或 'events'。[/red]")
            continue

        try:
            console.print("\n[bold blue]Claude：[/bold blue]")
            if mode == "simple":
                chat.stream_simple(user_input, console)
            else:
                chat.stream_with_events(user_input, console)
        except anthropic.APIError as e:
            logger.error("API 错误：%s", e)
            console.print(f"\n[red]API 错误：{e}[/red]")

    console.print()
    token_tracker.report()
    console.print(f"[dim]已交换消息数：{len(chat.messages)}[/dim]")


if __name__ == "__main__":
    main()
