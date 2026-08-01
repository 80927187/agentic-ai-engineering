"""
交互式聊天（OpenAI）

演示一个带有简单消息历史管理功能的交互式聊天循环。
"""

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from common import OpenAITokenTracker, setup_logging

# 从根目录的 .env 文件加载环境变量
load_dotenv(find_dotenv())

# 配置日志记录
logger = setup_logging(__name__)


class ChatSession:
    """
    维护对话历史的聊天智能体，封装了消息管理和 API 交互等
    全部聊天逻辑。
    """

    def __init__(self, model: str, token_callback: OpenAITokenTracker):
        """
        初始化聊天会话。
        """
        self.client = OpenAI()
        self.token_callback = token_callback
        self.messages: list[dict[str, str]] = []
        self.model = model

    def send_message(self, user_message: str) -> str:
        """
        使用 Responses API 发送消息并获取响应。
        """
        # 将用户消息添加到历史记录
        self.messages.append({"role": "user", "content": user_message})

        logger.info("正在向 OpenAI 发送消息（历史记录长度：%d）", len(self.messages))

        # 使用 Responses API 携带完整的消息历史记录调用 API
        response = self.client.responses.create(
            model=self.model,
            temperature=0.1,
            max_output_tokens=2048,
            input=self.messages,
        )

        # 记录 token 用量
        self.token_callback.track(response.usage)

        # 使用新的 output_text 属性提取助手的响应
        assistant_message = response.output_text or ""

        # 将助手的响应添加到历史记录
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message


def main() -> None:
    """
    处理用户交互并协调聊天流程的主编排函数。
    """

    # 使用 Rich 控制台美化输出
    console = Console()
    # 创建 token 跟踪器和聊天会话
    token_tracker = OpenAITokenTracker()
    chat = ChatSession("gpt-5.4", token_tracker)

    # 欢迎消息
    console.print(
        Panel(
            "[bold cyan]欢迎使用 OpenAI 聊天！[/bold cyan]\n\n"
            "输入消息后按 Enter 键发送。\n"
            "输入 'quit' 或 'exit' 结束对话。",
            title="聊天会话",
        )
    )

    # 聊天循环
    while True:
        # 获取用户输入
        console.print("\n[bold green]你：[/bold green] ", end="")
        user_input = input().strip()

        if user_input.lower() in ["quit", "exit", ""]:
            console.print("\n[yellow]正在结束聊天会话……[/yellow]")
            break

        # 发送消息并获取响应
        try:
            response = chat.send_message(user_input)

            # 使用 Markdown 格式显示响应
            console.print("\n[bold blue]GPT:[/bold blue]")
            console.print(Markdown(response))

        except Exception as e:
            logger.error("聊天过程中发生错误：%s", e)
            console.print(f"\n[red]错误：{e}[/red]")
            break

    # 报告最终的 token 用量
    console.print()
    token_tracker.report()
    console.print(f"\n[dim]交换的消息总数：{len(chat.messages)}[/dim]")


if __name__ == "__main__":
    main()
