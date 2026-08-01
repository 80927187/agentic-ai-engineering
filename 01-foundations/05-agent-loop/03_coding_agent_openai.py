"""
智能体循环（OpenAI）

演示一个最小的自主智能体，它能够：
- 接收用户任务
- 决定使用哪些工具
- 在循环中执行工具，直至任务完成
"""

import json
import subprocess
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from common.logging_config import setup_logging
from common.token_tracking import OpenAITokenTracker

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


# 工具定义（OpenAI Responses API 格式）
TOOLS = [
    {
        "type": "function",
        "name": "read_file",
        "description": "读取给定路径下的文件内容。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要读取的文件路径",
                }
            },
            "additionalProperties": False,
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "将内容写入给定路径下的文件。",
        "parameters": {
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
            "additionalProperties": False,
            "required": ["path", "content"],
        },
    },
    {
        "type": "function",
        "name": "bash",
        "description": "执行一条 bash 命令并返回其输出。",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 bash 命令",
                }
            },
            "additionalProperties": False,
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

    def __init__(self, model: str = "gpt-5.5"):
        self.client = OpenAI()
        self.model = model
        self.max_iterations = 10
        self.token_tracker = OpenAITokenTracker()

    def run(self, task: str) -> str:
        """针对给定任务执行智能体循环。"""
        logger.info(f"任务：{task}")

        input_messages: list[Any] = [{"role": "user", "content": task}]
        previous_response_id: str | None = None

        for iteration in range(self.max_iterations):
            logger.info(f"--- 第 {iteration + 1} 次迭代 ---")

            # 使用 Responses API 调用模型
            response = self.client.responses.create(
                model=self.model,
                tools=TOOLS,
                instructions=SYSTEM_PROMPT,
                input=input_messages,
                **({"previous_response_id": previous_response_id} if previous_response_id else {}),
            )

            if response.usage:
                self.token_tracker.track(response.usage)

            # 记录所有文本输出
            if response.output_text:
                logger.info(f"🤖 智能体：{response.output_text}")

            # 检查是否存在函数调用
            function_calls = [o for o in response.output if o.type == "function_call"]

            # 如果没有函数调用，则任务已完成
            if not function_calls:
                return response.output_text or "已完成"

            # 执行工具并收集结果
            tool_outputs: list[dict[str, str]] = []
            for call in function_calls:
                try:
                    args = json.loads(call.arguments)
                except json.JSONDecodeError as e:
                    args = {}
                    logger.error(f"无效的工具参数：{e}")

                logger.info(f"🔧 工具：{call.name}({json.dumps(args)})")
                result = execute_tool(call.name, args)
                logger.info(f"📋 结果：{result[:100]}{'...' if len(result) > 100 else ''}")

                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps({"result": result}),
                    }
                )

            # 使用工具输出继续对话
            previous_response_id = response.id
            input_messages = tool_outputs

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
            title="编程智能体（OpenAI）",
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
