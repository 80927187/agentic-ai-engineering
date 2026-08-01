"""
智能体循环（Anthropic）

演示一个最小的自主智能体，它能够：
- 接收用户任务
- 决定使用哪些工具
- 在循环中执行工具，直至任务完成
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import anthropic
from anthropic.types import TextBlock, ToolUseBlock
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from common.logging_config import setup_logging
from common.token_tracking import AnthropicTokenTracker

# 从根目录的 .env 文件加载环境变量
load_dotenv(find_dotenv())

# 配置日志记录
logger = setup_logging(__name__)

SYSTEM_PROMPT = """你是一个编程智能体。请使用提供的工具完成任务。

准则：
- 修改文件前先读取文件
- 逐步进行更改，并验证每一步
- 如果命令失败，请分析错误并尝试其他方法
- 完成后，简要总结你完成的工作"""


# 工具定义
TOOLS = [
    {
        "name": "read_file",
        "description": "读取给定路径下的文件内容。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要读取的文件路径",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "将内容写入给定路径下的文件。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要写入的文件路径",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的内容",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "bash",
        "description": "执行一条 bash 命令并返回其输出。",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 bash 命令",
                }
            },
            "required": ["command"],
        },
    },
]


def execute_tool(name: str, tool_input: dict[str, Any]) -> str:
    """执行工具，并以字符串形式返回结果。"""
    if name == "read_file":
        try:
            return Path(tool_input["path"]).read_text(encoding="utf-8")
        except Exception as e:
            return f"错误：{e}"

    elif name == "write_file":
        try:
            Path(tool_input["path"]).write_text(tool_input["content"], encoding="utf-8")
            return f"已成功写入 {tool_input['path']}"
        except Exception as e:
            return f"错误：{e}"

    elif name == "bash":
        try:
            result = subprocess.run(
                tool_input["command"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout + result.stderr
            return output if output else "（无输出）"
        except subprocess.TimeoutExpired:
            return "错误：命令执行超时"
        except Exception as e:
            return f"错误：{e}"

    return f"未知工具：{name}"


class CodingAgent:
    """
    最小自主编程智能体。

    在循环中执行工具，直至任务完成。
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_iterations = 10
        self.token_tracker = AnthropicTokenTracker()

    def run(self, task: str) -> str:
        """针对给定任务执行智能体循环。"""
        logger.info(f"任务：{task}")

        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]

        for iteration in range(self.max_iterations):
            logger.info(f"--- 第 {iteration + 1} 次迭代 ---")

            # 调用模型
            response = self.client.messages.create(
                model=self.model,
                temperature=0.1,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            self.token_tracker.track(response.usage)

            # 处理响应内容
            assistant_content = []
            for block in response.content:
                if isinstance(block, TextBlock):
                    logger.info(f"🤖 智能体：{block.text}")
                    assistant_content.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    logger.info(f"🔧 工具：{block.name}({json.dumps(block.input)})")
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            messages.append({"role": "assistant", "content": assistant_content})

            # 如果没有使用工具，则任务已完成
            if response.stop_reason == "end_turn":
                return response.content[0].text if response.content else "已完成"

            # 执行工具并收集结果
            tool_results = []
            for block in response.content:
                if isinstance(block, ToolUseBlock):
                    result = execute_tool(block.name, block.input)
                    logger.info(f"📋 结果：{result[:100]}{'...' if len(result) > 100 else ''}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

        return "已达到最大迭代次数"


def main() -> None:
    """主编排函数。"""
    console = Console()
    console.print(
        Panel(
            "示例：\n"
            "  - 参照现有文件的风格创建一个计算器\n"
            "  - 列出当前依赖项\n"
            "  - 解释当前文件夹中的代码\n\n"
            "输入 'quit' 退出。",
            title="编程智能体（Anthropic）",
        )
    )

    agent = CodingAgent()

    try:
        while True:
            console.print("\n[bold green]你：[/bold green] ", end="")
            user_input = input().strip()

            if user_input.lower() in ("exit", "quit", "q", ""):
                console.print("\n[yellow]正在结束会话……[/yellow]")
                break

            response = agent.run(user_input)
            console.print("\n[bold blue]智能体：[/bold blue]")
            console.print(Markdown(response))

    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")

    console.print()
    agent.token_tracker.report()


if __name__ == "__main__":
    main()
