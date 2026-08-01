"""
增强型 LLM——代码库导航器（Anthropic）

演示“增强型 LLM”模式：通过检索（RAG）、工具和记忆来增强 LLM。
正如 Anthropic《构建高效智能体》指南所述，这是所有智能体系统的基础构件。

代码库导航器可帮助工程师探索和理解陌生的代码库。只需指定任意 GitHub 仓库，
它便会克隆并建立索引，利用语义搜索回答问题，同时跨会话保留记忆。
"""

import json
import time
from typing import Any
import anthropic
from anthropic.types import TextBlock, ToolUseBlock
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from common import AnthropicTokenTracker, setup_logging
from common.menu import interactive_menu

# 加载环境变量
load_dotenv(find_dotenv())

logger = setup_logging(__name__)

SUGGESTED_REPOS = [
    "openai/swarm",
    "strands-agents/sdk-python",
    "anthropics/anthropic-sdk-python",
    "pallets/flask",
]


SYSTEM_PROMPT = """你是代码库导航器——一个帮助软件工程师探索和理解代码库的 AI 助手。

## 你的能力

你可以使用以下工具：
- 克隆 GitHub 仓库并建立索引（clone_and_index）
- 列出已索引的仓库（list_repos）
- 对代码进行语义搜索（search_code）
- 读取完整文件内容（read_file）
- 探索目录结构（list_directory）
- 使用正则表达式查找精确模式（grep）
- 保存记忆供未来会话使用（save_memory）
- 回忆已保存的记忆（recall_memory）

## 如何帮助用户

当用户提到 GitHub 仓库时（例如“pallets/flask”或“查看 httpie 仓库”）：
1. 首先使用 clone_and_index 克隆仓库并建立索引
2. 然后使用 search_code、read_file 等工具回答问题

对于语义或概念性问题，使用 search_code：
- “身份验证是如何工作的？”
- “数据库连接在哪里处理？”

对于精确匹配，使用 grep：
- “查找所有 TODO 注释”
- “UserModel 定义在哪里？”

找到相关代码块后，如果需要完整上下文，请使用 read_file。

## 记忆

将重要见解保存到记忆中，尤其是：
- 发现的架构模式
- 关键文件及其用途
- 不同仓库之间的联系
- 用户对信息呈现方式的偏好

在对话开始时检查 recall_memory，以回忆上下文。

## 回答风格

- 简洁但全面
- 展示相关代码片段，并标明文件路径和行号
- 发现架构决策时加以说明
- 推荐可以继续探索的相关领域
- 使用中文回答"""


# -- 工具注册表 -------------------------------------------------------------


def _build_tool_definitions() -> list[dict[str, Any]]:
    """收集所有工具模块中的工具定义。"""
    from tools.files import FILE_TOOLS
    from tools.memory import MEMORY_TOOLS
    from tools.repo import REPO_TOOLS
    from tools.search import SEARCH_TOOLS

    return MEMORY_TOOLS + REPO_TOOLS + FILE_TOOLS + SEARCH_TOOLS


class CodeNavigatorAgent:
    """
    一个通过检索、工具和记忆增强的 LLM。

    实现智能体循环：发送消息 → 执行工具 → 发送结果 → 重复，
    直到 LLM 仅返回文本。
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.client = anthropic.Anthropic()
        self.model = model
        self.token_tracker = AnthropicTokenTracker()
        self.tools = _build_tool_definitions()
        self.messages: list[dict[str, Any]] = []
        self.max_iterations = 15

        # 初始化共享组件（三种增强能力）
        from indexer.embedder import Embedder
        from store.memory import MemoryStore
        from store.vector import VectorStore

        self.memory = MemoryStore()
        self.vector_store = VectorStore()
        self.embedder = Embedder()

    def _build_system_prompt(self) -> str:
        """构建包含记忆上下文的系统提示词。"""
        memory_summary = self.memory.summary()
        if memory_summary and memory_summary != "尚未保存任何记忆。":
            return SYSTEM_PROMPT + f"\n\n## 已回忆的记忆\n{memory_summary}"
        return SYSTEM_PROMPT

    def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        """将工具调用分派给相应的处理程序。"""
        from tools.files import execute_list_directory, execute_read_file
        from tools.memory import execute_recall_memory, execute_save_memory
        from tools.repo import execute_clone_and_index, execute_list_repos
        from tools.search import execute_grep, execute_search_code

        dispatch: dict[str, Any] = {
            "save_memory": lambda inp: execute_save_memory(self.memory, inp),
            "recall_memory": lambda inp: execute_recall_memory(self.memory, inp),
            "clone_and_index": lambda inp: execute_clone_and_index(
                self.vector_store, self.embedder, inp
            ),
            "list_repos": lambda inp: execute_list_repos(self.vector_store, inp),
            "read_file": lambda inp: execute_read_file(self.vector_store, inp),
            "list_directory": lambda inp: execute_list_directory(self.vector_store, inp),
            "search_code": lambda inp: execute_search_code(self.vector_store, self.embedder, inp),
            "grep": lambda inp: execute_grep(self.vector_store, self.embedder, inp),
        }

        handler = dispatch.get(name)
        if not handler:
            return f"未知工具：{name}"

        try:
            return str(handler(tool_input))
        except Exception as e:
            logger.error("工具“%s”执行失败：%s", name, e)
            return f"执行 {name} 时出错：{e}"

    def chat(self, user_message: str, console: Console) -> str:
        """发送消息并处理智能体工具调用循环。"""
        self.messages.append({"role": "user", "content": user_message})

        for _iteration in range(self.max_iterations):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self._build_system_prompt(),
                    tools=self.tools,
                    messages=self.messages,
                )
            except anthropic.RateLimitError:
                logger.warning("受到速率限制——等待 30 秒后重试……")
                time.sleep(30)
                continue
            except anthropic.APIError as e:
                logger.error("API 错误：%s", e)
                return f"API 错误：{e}"

            self.token_tracker.track(response.usage)

            # 收集响应内容
            assistant_content = []
            text_parts: list[str] = []
            tool_uses: list[ToolUseBlock] = []

            for block in response.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                    assistant_content.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    tool_uses.append(block)
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            # 确保助手内容永不为空（API 要求）
            if not assistant_content:
                assistant_content = [{"type": "text", "text": "已完成。"}]
                text_parts = ["已完成。"]

            self.messages.append({"role": "assistant", "content": assistant_content})

            # 如果没有工具调用，则返回文本响应
            if response.stop_reason == "end_turn":
                return "\n".join(text_parts) if text_parts else "已完成。"

            # 执行各个工具并输出进度
            tool_results = []
            for tool_use in tool_uses:
                # 输出工具调用，使教学过程更加透明
                input_summary = json.dumps(tool_use.input, separators=(",", ":"))
                if len(input_summary) > 80:
                    input_summary = input_summary[:77] + "..."
                console.print(f"  [dim][工具：{tool_use.name}] {input_summary}[/dim]")

                result = self._execute_tool(tool_use.name, tool_use.input)

                # 输出简要结果
                result_preview = result.split("\n")[0][:80]
                console.print(f"  [dim]  → {result_preview}[/dim]")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result,
                    }
                )

            self.messages.append({"role": "user", "content": tool_results})

        return "已达到最大迭代次数，请尝试提出更具体的问题。"


# -- 主程序 -----------------------------------------------------------------


def main() -> None:
    """主编排函数。"""

    agent = CodeNavigatorAgent()

    console = Console()

    header = Panel(
        "[bold cyan]代码库导航器[/bold cyan]\n\n"
        "一个由[green]检索（RAG）[/green]、[yellow]工具[/yellow]和"
        "[magenta]记忆[/magenta]增强的 LLM。\n\n"
        "请选择要建立索引的仓库，然后提出有关代码库的问题。",
        title="代码库导航器",
    )

    repo = interactive_menu(
        console,
        SUGGESTED_REPOS,
        title="选择要探索的仓库",
        header=header,
        allow_custom=True,
        custom_prompt="输入所有者/仓库（例如 pallets/flask）",
    )
    if not repo:
        return

    console.print(f"\n[bold green]正在建立索引：[/bold green] {repo}")
    response = agent.chat(f"为仓库 {repo} 建立索引", console)
    if response:
        console.print("\n[bold blue]导航器：[/bold blue]")
        console.print(Markdown(response))

    console.print("\n[dim]请输入有关代码库的问题。输入“quit”可退出。[/dim]")

    try:
        while True:
            console.print("\n[bold green]你：[/bold green] ", end="")
            try:
                user_input = input().strip()
            except EOFError:
                break

            if user_input.lower() in ("exit", "quit", "q", ""):
                console.print("\n[yellow]正在结束会话……[/yellow]")
                break

            response = agent.chat(user_input, console)

            if response:
                console.print("\n[bold blue]导航器：[/bold blue]")
                console.print(Markdown(response))

            agent.token_tracker.report()

    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")

    console.print()
    agent.token_tracker.report()


if __name__ == "__main__":
    main()
