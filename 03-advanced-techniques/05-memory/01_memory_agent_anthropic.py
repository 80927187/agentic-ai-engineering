"""记忆系统——具备三层持久化记忆的个人助手。

演示智能体对话循环中的工作记忆（会话缓冲区）、情景记忆（持久化到 JSON 的事件）
和语义记忆（ChromaDB 向量存储）。智能体使用工具跨会话记住、回忆和遗忘信息。
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
from memory import MemoryManager

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
你是一名具备持久记忆的个人助手。你通过三层记忆系统跨会话记住用户信息：

1. **工作记忆**——临时会话笔记（自动清除）
2. **情景记忆**——带时间戳的事件和互动（持久化到 JSON）
3. **语义记忆**——事实、偏好和知识（持久化到向量数据库）

## 记忆准则

- 当用户分享个人信息（姓名、偏好、事实）时，将其以合适的重要性存入 \
**语义**记忆
- 当值得注意的事件或互动发生时，将其存入**情景**记忆
- 在向用户提问前，主动使用 **recall** 检查自己是否已知道相关信息
- 调整重要性分数：常规信息 = 0.3-0.5，个人详情 = 0.6-0.8，关键信息 = 0.9-1.0
- 对你记得的内容保持透明——回忆起某事时要告知用户

{memory_context}"""

# 智能体用来管理记忆的三个工具
MEMORY_TOOLS = [
    {
        "name": "remember",
        "description": (
            "将信息存入记忆。事实、偏好和知识使用 'semantic'；"
            "事件和互动使用 'episodic'；临时会话笔记使用 'working'。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "要记住的信息",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["working", "episodic", "semantic"],
                    "description": "要存入的记忆层",
                },
                "importance": {
                    "type": "number",
                    "description": "0.0 到 1.0 之间的重要性分数",
                    "default": 0.5,
                },
            },
            "required": ["content", "memory_type"],
        },
    },
    {
        "name": "recall",
        "description": (
            "在所有记忆层中搜索相关信息。"
            "在向用户提问前，使用此工具检查你已经知道的内容。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要在记忆中搜索的内容",
                },
                "limit": {
                    "type": "integer",
                    "description": "最大结果数（默认为 5）",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "forget",
        "description": "根据 ID 和类型删除指定记忆。",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "要删除的记忆 ID",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["episodic", "semantic"],
                    "description": "要从哪个记忆层删除",
                },
            },
            "required": ["memory_id", "memory_type"],
        },
    },
]


class MemoryAgent:
    """具备三层记忆和工具调用循环的个人助手。"""

    def __init__(self) -> None:
        self.client = anthropic.Anthropic()
        self.token_tracker = AnthropicTokenTracker()
        self.memory = MemoryManager()
        self.messages: list[dict[str, Any]] = []
        self.max_iterations = 10

    def _build_system_prompt(self) -> str:
        """将回忆起的记忆注入系统提示词。"""
        memory_context = self.memory.build_memory_context()
        return SYSTEM_PROMPT.format(memory_context=memory_context)

    def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        """将工具调用分派到相应的 MemoryManager 方法。"""
        try:
            if name == "remember":
                return self.memory.remember(
                    content=tool_input["content"],
                    memory_type=tool_input["memory_type"],
                    importance=tool_input.get("importance", 0.5),
                )
            elif name == "recall":
                return self.memory.recall(
                    query=tool_input["query"],
                    limit=tool_input.get("limit", 5),
                )
            elif name == "forget":
                return self.memory.forget(
                    memory_id=tool_input["memory_id"],
                    memory_type=tool_input["memory_type"],
                )
            else:
                return f"未知工具：{name}"
        except Exception as e:
            logger.error("工具 '%s' 执行失败：%s", name, e)
            return f"执行 {name} 时出错：{e}"

    def chat(self, user_message: str, console: Console) -> str:
        """发送消息并处理智能体工具调用循环。"""
        self.messages.append({"role": "user", "content": user_message})

        for _iteration in range(self.max_iterations):
            try:
                response = self.client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    system=self._build_system_prompt(),
                    tools=MEMORY_TOOLS,
                    messages=self.messages,
                )
            except anthropic.RateLimitError:
                logger.warning("触发速率限制——30 秒后重试……")
                time.sleep(30)
                continue
            except anthropic.APIError as e:
                logger.error("API 错误：%s", e)
                return f"API 错误：{e}"

            self.token_tracker.track(response.usage)

            # 收集响应内容
            assistant_content: list[dict[str, Any]] = []
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

            if not assistant_content:
                assistant_content = [{"type": "text", "text": "已完成。"}]
                text_parts = ["已完成。"]

            self.messages.append({"role": "assistant", "content": assistant_content})

            # 如果没有工具调用，则返回文本响应
            if response.stop_reason == "end_turn":
                return "\n".join(text_parts) if text_parts else "已完成。"

            # 执行每个工具并显示进度
            tool_results: list[dict[str, Any]] = []
            for tool_use in tool_uses:
                input_summary = json.dumps(tool_use.input, separators=(",", ":"))
                if len(input_summary) > 80:
                    input_summary = input_summary[:77] + "..."
                console.print(f"  [dim][工具：{tool_use.name}] {input_summary}[/dim]")

                result = self._execute_tool(tool_use.name, tool_use.input)

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

        return "已达到最大迭代次数。"

    def loaded_memory_count(self) -> int:
        """从之前会话中加载的持久记忆数量。"""
        stats = self.memory.get_stats()
        total: int = stats["episodic"]["count"] + stats["semantic"]["count"]
        return total


def main() -> None:
    """运行记忆增强的个人助手。"""
    console = Console()

    agent = MemoryAgent()
    loaded = agent.loaded_memory_count()

    # 欢迎面板
    status_line = (
        f"[green]已从之前的会话中加载 {loaded} 条记忆[/green]"
        if loaded
        else ("[dim]没有之前的记忆——这是一次全新的开始[/dim]")
    )

    header = Panel(
        "[bold cyan]记忆系统——个人助手[/bold cyan]\n\n"
        "一个使用三层记忆跨会话保持记忆的个人助手：\n"
        "  [bold]工作记忆[/bold]——临时会话缓冲区（自动清除）\n"
        "  [bold]情景记忆[/bold]——带时间戳的事件（持久化到 JSON）\n"
        "  [bold]语义记忆[/bold]——事实和知识（持久化到 ChromaDB）\n\n"
        f"{status_line}\n\n"
        "[bold]试试这些：[/bold]\n"
        '  • “你好，我叫小明，在示例公司工作”\n'
        '  • “比起 JavaScript，我更喜欢 Python”\n'
        '  • “你还记得我的哪些事？”（重启后）\n\n'
        '[dim]输入 "exit" 或 "quit" 结束会话[/dim]',
        title="教程 05——记忆系统",
    )
    console.print(header)

    try:
        while True:
            console.print("\n[bold green]你：[/bold green] ", end="")
            try:
                user_input = input().strip()
            except EOFError:
                break

            if user_input.lower() in ("exit", "quit", "q", ""):
                break

            response = agent.chat(user_input, console)
            if response:
                console.print("\n[bold blue]助手：[/bold blue]")
                console.print(Markdown(response))

    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")

    # 将会话整合到长期记忆中
    console.print("\n[dim]正在整合会话记忆……[/dim]")
    saved = agent.memory.consolidate(agent.messages, agent.client, MODEL)
    if saved:
        console.print(f"[green]已为下次会话保存 {len(saved)} 条记忆：[/green]")
        for item in saved:
            console.print(f"  [dim]{item}[/dim]")
    else:
        console.print("[dim]没有需要整合的新记忆。[/dim]")

    console.print()
    agent.token_tracker.report()


if __name__ == "__main__":
    main()
