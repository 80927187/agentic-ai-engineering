"""
支持工具调用的流式智能体（Anthropic）

演示流式传输中的难点：在流式传输途中处理 tool_use 块。智能体实时流式输出文本，
检测 Claude 何时需要调用工具，执行工具并反馈结果，然后恢复流式传输，同时保持
终端界面流畅且响应及时。
"""

import ast
import json
import operator
import random
from typing import Any

import anthropic
from anthropic.types import ContentBlock, TextBlock, ToolUseBlock
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
    "你是一名可以使用工具、乐于助人的助手。"
    "当工具能够提供准确答案时，请使用工具，不要猜测天气或数学计算结果。"
    "获得工具结果后，将其自然地融入回答。回答应简洁，并使用 Markdown 格式。"
)

# --- 工具定义 ---

TOOLS = [
    {
        "name": "get_weather",
        "description": (
            "获取某个城市当前的天气状况，返回温度、天气情况和湿度。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称（例如 'San Francisco'、'Tokyo'、'London'）",
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "安全地计算数学表达式。支持算术运算符（+、-、*、/、**、%）和括号。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式（例如 '(15 * 3) + 42'）",
                },
            },
            "required": ["expression"],
        },
    },
]


# --- 工具实现 ---

# 模拟天气数据——无需 API 密钥，使教程能够独立运行
WEATHER_DATA: dict[str, dict[str, Any]] = {
    "san francisco": {"temp_f": 62, "conditions": "有雾", "humidity": 78},
    "new york": {"temp_f": 45, "conditions": "局部多云", "humidity": 55},
    "tokyo": {"temp_f": 58, "conditions": "晴朗", "humidity": 42},
    "london": {"temp_f": 48, "conditions": "阴天", "humidity": 82},
    "paris": {"temp_f": 52, "conditions": "小雨", "humidity": 75},
    "sydney": {"temp_f": 77, "conditions": "晴天", "humidity": 60},
}


def get_weather(city: str) -> dict[str, Any]:
    """使用仿真数据模拟天气查询。"""
    key = city.lower().strip()
    if key in WEATHER_DATA:
        data = WEATHER_DATA[key]
    else:
        # 为未知城市生成合理的天气数据
        data = {
            "temp_f": random.randint(35, 85),
            "conditions": random.choice(["晴朗", "多云", "局部多云", "小雨"]),
            "humidity": random.randint(30, 90),
        }

    logger.info("天气查询：%s → %s", city, data["conditions"])
    return {"city": city, **data}


# 用于数学计算的安全运算符
SAFE_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """仅使用安全运算符递归计算 AST 节点。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    elif isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        result: float = SAFE_OPERATORS[type(node.op)](left, right)
        return result
    elif isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
        result_u: float = SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
        return result_u
    raise ValueError(f"不支持的表达式：{ast.dump(node)}")


def calculate(expression: str) -> dict[str, Any]:
    """安全的数学计算器——不使用 eval()，仅通过解析 AST 进行算术运算。"""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        logger.info("计算：%s = %s", expression, result)
        return {"expression": expression, "result": result}
    except (ValueError, TypeError, ZeroDivisionError, SyntaxError) as e:
        logger.error("计算错误：%s — %s", expression, e)
        return {"expression": expression, "error": str(e)}


TOOL_FUNCTIONS: dict[str, Any] = {
    "get_weather": get_weather,
    "calculate": calculate,
}


def execute_tool(name: str, tool_input: dict[str, Any]) -> str:
    """执行工具并返回 JSON 结果字符串。"""
    if name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"未知工具：{name}"}, ensure_ascii=False)

    try:
        result = TOOL_FUNCTIONS[name](**tool_input)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("工具执行错误（%s）：%s", name, e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# --- 流式智能体 ---


class StreamingAgent:
    """流式输出响应并在传输途中处理工具调用的智能体。

    核心挑战在于，单个 API 响应可以包含交错的文本块和 tool_use 块。
    该智能体将文本实时流式输出到终端，在 tool_use 块到达时检测并执行工具，
    然后进入下一轮循环，获取模型的后续响应。
    """

    def __init__(self, model: str, token_tracker: AnthropicTokenTracker) -> None:
        self.client = anthropic.Anthropic()
        self.model = model
        self.token_tracker = token_tracker
        self.messages: list[dict[str, Any]] = []

    def run(self, user_input: str, console: Console) -> str:
        """运行完整的智能体循环：流式传输 → 检测工具 → 执行 → 恢复传输。

        循环持续到模型返回 stop_reason="end_turn"，表示不再需要调用工具。
        """
        self.messages.append({"role": "user", "content": user_input})
        full_response_text = ""
        iteration = 0
        max_iterations = 100  # 防止无限循环的安全上限

        while iteration < max_iterations:
            iteration += 1
            logger.info("智能体循环第 %d 次迭代", iteration)

            # 流式输出响应，同时渲染文本并检测工具调用
            response_message = self._stream_response(console)

            # 将助手的完整响应添加到对话历史
            self.messages.append({"role": "assistant", "content": response_message.content})

            # 收集本次响应中的所有文本
            for block in response_message.content:
                if isinstance(block, TextBlock) and block.text:
                    full_response_text += block.text

            # 如果没有工具调用，任务即完成
            if response_message.stop_reason != "tool_use":
                logger.info("智能体已完成（stop_reason：%s）", response_message.stop_reason)
                break

            # 执行工具调用并反馈结果
            tool_results = self._execute_tool_calls(response_message.content, console)
            self.messages.append({"role": "user", "content": tool_results})

            # 下一次迭代将流式输出模型的后续响应

        return full_response_text

    def _stream_response(self, console: Console) -> anthropic.types.Message:
        """执行一次流式 API 调用，实时渲染文本和工具调用。

        返回完整累积的消息，用于跟踪对话历史。
        """
        with self.client.messages.stream(
            model=self.model,
            max_tokens=21333,
            system=SYSTEM_PROMPT,
            messages=self.messages,
            tools=TOOLS,
        ) as stream:
            self._render_stream(stream, console)
            final_message = stream.get_final_message()
            self.token_tracker.track(final_message.usage)
            return final_message

    def _render_stream(self, stream: anthropic.MessageStream, console: Console) -> None:
        """渲染由文本块和 tool_use 块组成的混合流。

        这是实现工具流式传输的核心方法：
        - 文本增量 → 实时渲染为 Markdown
        - tool_use 块 → 显示为状态指示信息
        - input_json_delta → 累积工具参数（记录日志但不显示）
        """
        accumulated_text = ""
        current_block_type: str | None = None
        current_tool_name: str | None = None
        live: Live | None = None

        try:
            for event in stream:
                if event.type == "content_block_start":
                    current_block_type = event.content_block.type

                    if current_block_type == "text":
                        # 开始实时渲染文本
                        live = Live(Markdown(""), refresh_per_second=15, console=console)
                        live.start()

                    elif current_block_type == "tool_use":
                        # 工具调用开始——显示正在调用的工具
                        current_tool_name = event.content_block.name
                        console.print(
                            f"\n[dim]  ⚡ 正在调用 [bold]{current_tool_name}[/bold]……[/dim]",
                            end="",
                        )

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        # 文本到达——更新实时显示
                        accumulated_text += event.delta.text
                        if live is not None:
                            live.update(Markdown(accumulated_text))

                    elif event.delta.type == "input_json_delta":
                        # 工具输入参数正在流式传入
                        # 不显示这些参数，只让它们累积
                        # SDK 的 get_final_message() 会提供解析后的输入
                        logger.debug("工具输入增量：%s", event.delta.partial_json)

                elif event.type == "content_block_stop":
                    if current_block_type == "text" and live is not None:
                        # 文本块结束——停止实时渲染
                        live.stop()
                        live = None

                    elif current_block_type == "tool_use":
                        # 工具调用块结束
                        console.print()  # 在“正在调用……”消息后换行

                    current_block_type = None
                    current_tool_name = None

        finally:
            # 确保在流式传输途中发生错误时停止实时显示
            if live is not None:
                live.stop()

    def _execute_tool_calls(
        self, content: list[ContentBlock], console: Console
    ) -> list[dict[str, Any]]:
        """执行响应中的所有工具调用，并将结果格式化为 API 所需形式。"""
        tool_results = []

        for block in content:
            if isinstance(block, ToolUseBlock):
                logger.info("正在执行工具：%s(%s)", block.name, json.dumps(block.input))
                console.print(
                    f"[dim]  → {block.name}({json.dumps(block.input, separators=(',', ':'))})[/dim]"
                )

                result = execute_tool(block.name, block.input)
                console.print(
                    f"[dim]  ✓ 结果：{result[:100]}{'...' if len(result) > 100 else ''}[/dim]"
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        return tool_results

    def reset(self) -> None:
        """清空对话历史。"""
        self.messages.clear()
        logger.info("对话历史已清空")


def main() -> None:
    """可使用工具的交互式流式智能体。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()
    agent = StreamingAgent(MODEL, token_tracker)

    console.print(
        Panel(
            "[bold cyan]支持工具的流式智能体[/bold cyan]\n\n"
            "实时观察响应的流式输出，即使 Claude 在响应途中调用工具也不例外。\n\n"
            "[bold]可用工具：[/bold]\n"
            "  🌤️  [green]get_weather[/green] — 获取任意城市的当前天气\n"
            "  🔢 [green]calculate[/green]    — 计算数学表达式\n\n"
            "[bold]可以尝试以下提示词：[/bold]\n"
            '  • “旧金山的天气怎么样？”\n'
            '  • “计算复利：10000 * (1 + 0.05) ** 10”\n'
            '  • “比较东京和伦敦的天气，并计算温度差”\n\n'
            "输入 [bold]clear[/bold] 重置，输入 [bold]quit[/bold] 退出。",
            title="02-streaming / 02 — 流式智能体",
        )
    )

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
            agent.reset()
            console.print("[dim]对话已清空。[/dim]")
            continue

        try:
            console.print("\n[bold blue]Claude：[/bold blue]")
            agent.run(user_input, console)
        except anthropic.APIError as e:
            logger.error("API 错误：%s", e)
            console.print(f"\n[red]API 错误：{e}[/red]")

    console.print()
    token_tracker.report()
    console.print(f"[dim]已交换消息数：{len(agent.messages)}[/dim]")


if __name__ == "__main__":
    main()
