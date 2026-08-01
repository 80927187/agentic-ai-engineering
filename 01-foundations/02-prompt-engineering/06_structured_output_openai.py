"""
结构化输出与提示词脚手架（OpenAI）

演示从 OpenAI 获取可解析结构化输出的技术：
1. 通过提示词指令输出 JSON——在系统提示词中要求返回 JSON
2. Markdown 脚手架——使用结构化分节引导输出
3. JSON Schema 强制执行——OpenAI 的原生结构化输出功能

三种方法从同一段描述中提取相同的产品信息，便于比较不同技术的可靠性。
"""

import json

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from common import OpenAITokenTracker, interactive_menu, setup_logging

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

# 大语言模型需要填充的模式（便于人类阅读）
PRODUCT_SCHEMA = {
    "name": "字符串——产品名称",
    "category": "字符串——产品类别（例如电子产品、服装）",
    "price": "数字——以美元计价的价格",
    "features": "字符串列表——主要产品特性",
    "in_stock": "布尔值——产品当前是否有货",
}

# 用于 OpenAI 原生结构化输出强制执行的 JSON Schema
PRODUCT_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "product_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "产品名称"},
            "category": {
                "type": "string",
                "description": "产品类别（例如电子产品、服装）",
            },
            "price": {"type": "number", "description": "以美元计价的价格"},
            "features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "主要产品特性",
            },
            "in_stock": {
                "type": "boolean",
                "description": "产品当前是否有货",
            },
        },
        "required": ["name", "category", "price", "features", "in_stock"],
        "additionalProperties": False,
    },
}

# 单条产品描述——三种方法都从这份相同的输入中提取数据
PRODUCT_DESCRIPTION = (
    "UltraSound Pro X1 无线降噪耳机配备 40 毫米定制驱动单元和自适应主动降噪，"
    "可提供录音室级音质。其特性包括 30 小时续航、可同时连接两台设备的"
    "多点蓝牙 5.3，以及配有高级便携盒的可折叠设计。现售价 249.99 美元。"
    "目前有货，并会在 24 小时内发货。"
)


class StructuredOutputClient:
    """使用 OpenAI API 演示结构化输出技术。"""

    def __init__(self, model: str, token_tracker: OpenAITokenTracker):
        self.client = OpenAI()
        self.model = model
        self.token_tracker = token_tracker

    def _call(self, instructions: str, user_input: str, **kwargs) -> str:
        """执行一次 API 调用并跟踪 Token 使用量。"""
        response = self.client.responses.create(
            model=self.model,
            temperature=0.0,
            max_output_tokens=512,
            instructions=instructions,
            input=user_input,
            **kwargs,
        )
        if hasattr(response, "usage") and response.usage:
            self.token_tracker.track(response.usage)
        return response.output_text or ""

    def extract_json_prompted(self, description: str) -> str:
        """通过在提示词中要求 JSON 来提取结构化数据。"""
        schema_str = json.dumps(PRODUCT_SCHEMA, indent=2, ensure_ascii=False)
        instructions = (
            "你是一名产品数据提取助手。请从产品描述中提取结构化信息。\n\n"
            f"只输出符合以下模式的有效 JSON：\n{schema_str}\n\n"
            "不要使用 Markdown，不要解释——只输出 JSON 对象。"
        )
        return self._call(instructions, description)

    def extract_with_scaffolding(self, description: str) -> str:
        """使用 Markdown 分节搭建输入脚手架并引导输出。"""
        schema_str = json.dumps(PRODUCT_SCHEMA, indent=2, ensure_ascii=False)
        # OpenAI 对 Markdown 结构的提示词支持良好
        instructions = (
            "你是一名产品数据提取助手。你会接收结构化输入，并以 JSON 提取产品数据。\n\n"
            "只输出符合所提供模式的有效 JSON。不要使用 Markdown 代码围栏，不要解释。"
        )
        user_input = (
            f"## 模式\n```json\n{schema_str}\n```\n\n"
            f"## 产品描述\n{description}\n\n"
            "## 输出\n以 JSON 提取产品信息："
        )
        return self._call(instructions, user_input)

    def extract_with_schema(self, description: str) -> str:
        """使用 OpenAI 原生 JSON Schema 强制执行——保证 JSON 有效。"""
        instructions = (
            "你是一名产品数据提取助手。请从产品描述中提取结构化信息，"
            "并根据描述填充所有字段。"
        )
        # OpenAI 的 text.format 参数在 API 层强制执行模式
        return self._call(
            instructions,
            description,
            text={"format": PRODUCT_JSON_SCHEMA},
        )


def _try_parse_json(raw: str) -> dict | None:
    """尝试解析 JSON；如果存在 Markdown 代码围栏，则将其移除。"""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        parsed: dict[str, object] = json.loads(text)
        return parsed
    except json.JSONDecodeError as e:
        logger.warning("JSON 解析失败：%s", e)
        return None


def _display_result(console: Console, method_name: str, raw: str) -> None:
    """解析并显示结构化输出方法返回的 JSON 结果。"""
    parsed = _try_parse_json(raw)
    if parsed:
        formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
        syntax = Syntax(formatted, "json", theme="monokai")
        console.print(Panel(syntax, title=f"{method_name} [green]有效 JSON[/green]"))
    else:
        console.print(Panel(raw[:300], title=f"{method_name} [red]解析失败[/red]"))


METHOD_LABELS = [
    "1：提示词 JSON",
    "2：Markdown 脚手架",
    "3：模式强制执行",
]


def _run_method_1(console: Console, client: StructuredOutputClient) -> None:
    """运行提示词 JSON 提取方法。"""
    schema_str = json.dumps(PRODUCT_SCHEMA, indent=2, ensure_ascii=False)
    console.print("[dim]将模式嵌入指令，并要求输出 JSON。[/dim]\n")
    prompt_1 = (
        "**指令：**\n"
        "```\n"
        "你是一名产品数据提取助手……\n"
        f"只输出符合以下模式的有效 JSON：\n{schema_str}\n"
        "不要使用 Markdown，不要解释——只输出 JSON 对象。\n"
        "```\n\n"
        "**输入：** _（原始产品描述）_\n"
    )
    console.print(Markdown(prompt_1))

    try:
        raw = client.extract_json_prompted(PRODUCT_DESCRIPTION)
        _display_result(console, "提示词 JSON", raw)
    except Exception as e:
        logger.error("方法 1 出错：%s", e)


def _run_method_2(console: Console, client: StructuredOutputClient) -> None:
    """运行 Markdown 脚手架提取方法。"""
    schema_str = json.dumps(PRODUCT_SCHEMA, indent=2, ensure_ascii=False)
    console.print("[dim]使用 Markdown 分节构造输入，以引导输出。[/dim]\n")
    prompt_2 = (
        "**指令：**\n"
        "```\n"
        "你是一名产品数据提取助手。\n"
        "只输出符合所提供模式的有效 JSON。\n"
        "不要使用 Markdown 代码围栏，不要解释。\n"
        "```\n\n"
        "**输入（Markdown 结构）：**\n"
        "```markdown\n"
        f"## 模式\n```json\n{schema_str}\n```\n\n"
        "## 产品描述\n（此处为产品描述）\n\n"
        "## 输出\n以 JSON 提取产品信息：\n"
        "```\n"
    )
    console.print(Markdown(prompt_2))

    try:
        raw = client.extract_with_scaffolding(PRODUCT_DESCRIPTION)
        _display_result(console, "Markdown 脚手架", raw)
    except Exception as e:
        logger.error("方法 2 出错：%s", e)


def _run_method_3(console: Console, client: StructuredOutputClient) -> None:
    """运行模式强制执行提取方法。"""
    console.print("[dim]通过 text.format 在 API 层强制执行——保证 JSON 有效。[/dim]\n")
    schema_preview = json.dumps(PRODUCT_JSON_SCHEMA, indent=2, ensure_ascii=False)
    prompt_3 = (
        "**指令：**\n"
        "```\n"
        "你是一名产品数据提取助手……\n"
        "根据描述填充所有字段。\n"
        "```\n\n"
        "**输入：** _（原始产品描述）_\n\n"
        "**text.format（JSON 模式）：**\n"
        f"```json\n{schema_preview}\n```\n\n"
        "_API 保证响应符合此模式——无需解析。_\n"
    )
    console.print(Markdown(prompt_3))

    try:
        raw = client.extract_with_schema(PRODUCT_DESCRIPTION)
        _display_result(console, "模式强制执行", raw)
    except Exception as e:
        logger.error("方法 3 出错：%s", e)


def main() -> None:
    """使用三种结构化输出方法处理同一条产品描述。"""
    console = Console()
    token_tracker = OpenAITokenTracker()
    client = StructuredOutputClient("gpt-5.5", token_tracker)

    header = Panel(
        "[bold cyan]结构化输出与提示词脚手架[/bold cyan]\n\n"
        "比较从自由文本中提取结构化 JSON 的 3 种技术：\n"
        "  1. 通过提示词指令输出 JSON\n"
        "  2. Markdown 脚手架\n"
        "  3. JSON Schema 强制执行（OpenAI 特有）\n\n"
        f"[bold]产品描述：[/bold]\n{PRODUCT_DESCRIPTION}",
        title="提示工程 — OpenAI",
    )

    methods = {
        METHOD_LABELS[0]: _run_method_1,
        METHOD_LABELS[1]: _run_method_2,
        METHOD_LABELS[2]: _run_method_3,
    }

    try:
        while True:
            selection = interactive_menu(
                console,
                METHOD_LABELS,
                title="选择一种方法",
                header=header,
            )
            if not selection:
                break

            console.print(f"\n[bold yellow]━━━ {selection} ━━━[/bold yellow]")

            try:
                methods[selection](console, client)
            except Exception as e:
                logger.error("方法出错：%s", e)

            token_tracker.report()
            token_tracker.reset()

            console.print("\n[dim]按 Enter 键继续……[/dim]")
            input()

    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")


if __name__ == "__main__":
    main()
