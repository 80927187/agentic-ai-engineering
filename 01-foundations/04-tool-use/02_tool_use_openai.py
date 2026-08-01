"""
工具使用（OpenAI）

演示如何让模型调用函数和使用工具。
使用以下实用工具：计算器、文件读取器和 Bash 命令执行器。
"""

import json
import subprocess
from typing import Any

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from common import OpenAITokenTracker, setup_logging

# 从根目录的 .env 文件加载环境变量
load_dotenv(find_dotenv())

# 配置日志
logger = setup_logging(__name__)


# 使用 OpenAI Responses API 格式定义可用工具
TOOLS = [
    {
        "type": "function",
        "name": "calculator",
        "description": "执行基本算术运算，支持加法、减法、乘法和除法。",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "要执行的算术运算",
                },
                "a": {"type": "number", "description": "第一个数"},
                "b": {"type": "number", "description": "第二个数"},
            },
            "required": ["operation", "a", "b"],
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "读取指定路径下的文件内容，并以文本形式返回。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要读取的文件路径",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "最多读取的行数（默认值：100）",
                    "default": 100,
                },
            },
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "run_bash",
        "description": "执行 Bash 命令并返回输出。适用于 ls、pwd、echo、date 等系统命令。",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 Bash 命令",
                },
                "timeout": {
                    "type": "integer",
                    "description": "超时时间（秒，默认值：30）",
                    "default": 30,
                },
            },
            "required": ["command"],
        },
    },
]


def calculator(operation: str, a: float, b: float) -> dict[str, Any]:
    """执行计算器工具。"""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "错误：除数不能为零",
    }

    result = operations[operation](a, b)
    logger.info("计算器：%s %s %s = %s", a, operation, b, result)

    return {"result": result, "operation": operation, "operands": [a, b]}


def read_file(path: str, max_lines: int = 100) -> dict[str, Any]:
    """读取文件内容。"""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)
        content = "".join(lines[:max_lines])
        truncated = total_lines > max_lines

        logger.info("读取文件：%s（%d 行）", path, total_lines)

        return {
            "path": path,
            "content": content,
            "total_lines": total_lines,
            "truncated": truncated,
        }
    except FileNotFoundError:
        return {"error": f"找不到文件：{path}"}
    except PermissionError:
        return {"error": f"没有权限访问：{path}"}
    except Exception as e:
        return {"error": str(e)}


BLOCKED_COMMANDS = ["rm", "sudo", "chmod", "chown", "mkfs", "dd", "shutdown", "reboot", ">", ">>"]


def run_bash(command: str, timeout: int = 30) -> dict[str, Any]:
    """执行 Bash 命令并返回输出。"""
    # 简单的安全防护：拦截危险命令
    cmd_lower = command.lower().strip()
    for blocked in BLOCKED_COMMANDS:
        if blocked in cmd_lower:
            logger.warning("已拦截危险命令：%s", command)
            return {"error": f"出于安全考虑，命令已被拦截：包含 '{blocked}'"}

    logger.info("正在运行 Bash 命令：%s", command)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return {
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"命令执行超过 {timeout} 秒后超时"}
    except Exception as e:
        return {"error": str(e)}


# 工具执行映射
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "read_file": read_file,
    "run_bash": run_bash,
}


def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> Any:
    """执行工具并返回结果。"""
    if tool_name not in TOOL_FUNCTIONS:
        return {"error": f"未知工具：{tool_name}"}

    try:
        func = TOOL_FUNCTIONS[tool_name]
        return func(**tool_input)  # type: ignore[operator]
    except Exception as e:
        logger.error("工具执行错误：%s", e)
        return {"error": str(e)}


class ToolUseChat:
    """具备工具使用能力的聊天会话。"""

    def __init__(
        self,
        model: str,
        token_tracker: OpenAITokenTracker,
        console: Console,
    ):
        """初始化带工具的聊天会话。"""
        self.client = OpenAI()
        self.token_tracker = token_tracker
        self.console = console
        self.messages: list[dict[str, Any]] = []
        self.model = model

    def send_message(self, user_message: str) -> str:
        """发送消息并处理可能发生的工具调用。"""
        # 添加用户消息
        self.messages.append({"role": "user", "content": user_message})

        # 持续处理，直到获得最终文本回复
        while True:
            logger.info("API 调用（消息数：%d）", len(self.messages))

            # 使用 Responses API 携带工具发起 API 调用
            response = self.client.responses.create(
                model=self.model,
                max_output_tokens=4096,
                tools=TOOLS,
                input=self.messages,
            )

            # 记录 token 用量
            self.token_tracker.track(response.usage)

            # 检查回复中是否包含函数调用
            function_calls = [o for o in response.output if o.type == "function_call"]

            if function_calls:
                # 将回复中的所有输出项添加到消息列表（包括函数调用）
                self.messages.extend(response.output)

                # 执行工具
                self.console.print("\n[yellow]-> 正在执行工具...[/yellow]")

                for func_call in function_calls:
                    function_name = func_call.name
                    function_args = json.loads(func_call.arguments)

                    self.console.print(
                        f"  [dim]* {function_name}({json.dumps(function_args, indent=2)})[/dim]"
                    )

                    # 执行工具
                    result = execute_tool(function_name, function_args)

                    # 将函数调用输出添加到消息列表，供下一轮迭代使用
                    self.messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": func_call.call_id,
                            "output": json.dumps(result),
                        }
                    )

                # 继续循环以获取最终回复
                continue

            else:
                # 没有函数调用，提取并返回文本回复
                return response.output_text or ""

    def get_message_count(self) -> int:
        """获取对话中的消息总数。"""
        return len(self.messages)


def main() -> None:
    """处理用户交互并协调聊天流程的主编排函数。"""
    console = Console()
    token_tracker = OpenAITokenTracker()
    chat = ToolUseChat("gpt-5.5", token_tracker, console)

    # 欢迎消息
    console.print(
        Panel(
            "[bold cyan]带工具的智能体！[/bold cyan]\n\n"
            "可用工具：\n"
            "* 计算器（加、减、乘、除）\n"
            "* 读取文件（读取任意文件的内容）\n"
            "* 运行 Bash（执行 Shell 命令）\n\n"
            "试试输入：'123 * 456 等于多少？' 或 '列出当前目录中的文件'\n"
            "也可以输入：'读取 pyproject.toml 文件'\n\n"
            "输入 'quit' 退出。",
            title="工具使用演示",
        )
    )

    # 聊天循环
    try:
        while True:
            console.print("\n[bold green]你：[/bold green] ", end="")
            user_input = input().strip()

            if user_input.lower() in ["quit", "exit", ""]:
                console.print("\n[yellow]正在结束聊天会话...[/yellow]")
                break

            try:
                response = chat.send_message(user_input)

                if response:
                    console.print("\n[bold blue]智能体：[/bold blue]")
                    console.print(Markdown(response))

            except Exception as e:
                logger.error("聊天过程中发生错误：%s", e)
                console.print(f"\n[red]错误：{e}[/red]")
                break

    except KeyboardInterrupt:
        console.print("\n[yellow]操作已中断，正在结束聊天会话...[/yellow]")

    # 报告用量
    console.print()
    token_tracker.report()
    console.print(f"\n[dim]已交换的消息总数：{chat.get_message_count()}[/dim]")


if __name__ == "__main__":
    main()
