"""
共享的工具定义、实现与分发逻辑。

提供所有单元测试教程脚本共用的 calculator、read_file 和 run_bash 工具，
其中包括安全防护措施（禁止的命令）以及通用的 execute_tool 分发器。
"""

import subprocess
from pathlib import Path
from typing import Any

from common import setup_logging

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# 工具定义（Anthropic 工具调用格式）
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "calculator",
        "description": "执行基本算术运算。",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        },
    },
    {
        "name": "read_file",
        "description": "读取指定路径的文件内容。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_lines": {"type": "integer", "default": 100},
            },
            "required": ["path"],
        },
    },
    {
        "name": "run_bash",
        "description": "执行 Bash 命令并返回输出。",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 30},
            },
            "required": ["command"],
        },
    },
]

BLOCKED_COMMANDS = ["rm", "sudo", "chmod", "chown", "mkfs", "dd", "shutdown", "reboot", ">", ">>"]


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------


def calculator(operation: str, a: float, b: float) -> dict[str, Any]:
    """执行计算器工具。"""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "错误：除数不能为零",
    }
    if operation not in operations:
        return {"error": f"未知运算：{operation}"}
    result = operations[operation](a, b)
    logger.info("计算器：%s %s %s = %s", a, operation, b, result)
    return {"result": result, "operation": operation, "operands": [a, b]}


def read_file(path: str, max_lines: int = 100) -> dict[str, Any]:
    """读取文件内容。"""
    try:
        with Path(path).open(encoding="utf-8") as f:
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
        return {"error": f"权限不足：{path}"}


def run_bash(command: str, timeout: int = 30) -> dict[str, Any]:
    """执行 Bash 命令并返回输出。"""
    cmd_lower = command.lower().strip()
    for blocked in BLOCKED_COMMANDS:
        if blocked in cmd_lower:
            logger.warning("已阻止危险命令：%s", command)
            return {"error": f"出于安全考虑，命令已被阻止：包含“{blocked}”"}
    logger.info("正在运行 Bash 命令：%s", command)
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"命令在 {timeout} 秒后超时"}


# ---------------------------------------------------------------------------
# 工具分发
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS: dict[str, Any] = {
    "calculator": calculator,
    "read_file": read_file,
    "run_bash": run_bash,
}


def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> Any:
    """执行工具并返回结果。"""
    if tool_name not in TOOL_FUNCTIONS:
        return {"error": f"未知工具：{tool_name}"}
    try:
        return TOOL_FUNCTIONS[tool_name](**tool_input)
    except TypeError as e:
        logger.error("工具 %s 的参数无效：%s", tool_name, e)
        return {"error": f"参数无效：{e}"}
    except Exception as e:
        logger.error("工具执行错误：%s", e)
        return {"error": str(e)}
