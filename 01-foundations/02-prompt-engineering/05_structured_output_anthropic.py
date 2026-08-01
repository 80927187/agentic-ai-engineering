"""
结构化输出与提示词脚手架（Anthropic）

演示从 Claude 获取结构化 JSON 的三种方法，可靠性由低到高：
1. 基于提示词的 JSON——在系统提示词中要求返回 JSON（可能失败）
2. XML 脚手架——Anthropic 特有的提示技术（更可靠）
3. 原生 JSON Schema——通过 output_config 在 API 层强制执行模式（有保证）

注意：本教程的早期版本还演示了助手消息“预填充”（在助手轮次中预先放入 `{`
以强制输出 JSON）。Claude 4.6 已移除对助手预填充的支持——对话必须以用户消息
结束——因此删除了该步骤。详见：
https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-6

三种方法从同一段描述中提取相同的产品信息，便于比较不同技术的可靠性。
"""

import json

import anthropic
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from common import AnthropicTokenTracker, interactive_menu, setup_logging

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

# 用于提示词方法的模式描述（便于人类阅读）
PRODUCT_SCHEMA_DESCRIPTION = {
    "name": "字符串——产品名称",
    "category": "字符串——产品类别（例如电子产品、服装）",
    "price": "数字——以美元计价的价格",
    "features": "字符串列表——主要产品特性",
    "in_stock": "布尔值——产品当前是否有货",
}


# 用于原生结构化输出的 Pydantic 模型（由机器强制执行）
class ProductExtraction(BaseModel):
    """从自由文本描述中提取结构化产品数据的模式。"""

    name: str
    category: str
    price: float
    features: list[str]
    in_stock: bool


# 单条产品描述——三种方法都从这份相同的输入中提取数据
PRODUCT_DESCRIPTION = (
    "UltraSound Pro X1 无线降噪耳机配备 40 毫米定制驱动单元和自适应主动降噪，"
    "可提供录音室级音质。其特性包括 30 小时续航、可同时连接两台设备的"
    "多点蓝牙 5.3，以及配有高级便携盒的可折叠设计。现售价 249.99 美元。"
    "目前有货，并会在 24 小时内发货。"
)


class StructuredOutputClient:
    """使用 Anthropic API 演示结构化输出技术。"""

    def __init__(self, model: str, token_tracker: AnthropicTokenTracker):
        self.client = anthropic.Anthropic()
        self.model = model
        self.token_tracker = token_tracker

    def _call(self, system: str, messages: list[dict], **kwargs: object) -> str:
        """执行一次 API 调用并跟踪 Token 使用量。"""
        response = self.client.messages.create(
            model=self.model,
            temperature=0.0,
            max_tokens=512,
            system=system,
            messages=messages,
            **kwargs,
        )
        self.token_tracker.track(response.usage)
        return str(response.content[0].text)

    def extract_json_prompted(self, description: str) -> str:
        """通过在提示词中要求 JSON 来提取结构化数据——可靠性最低。"""
        schema_str = json.dumps(PRODUCT_SCHEMA_DESCRIPTION, indent=2)
        system = (
            "你是一名产品数据提取助手。请从产品描述中提取结构化信息。\n\n"
            f"只输出符合以下模式的有效 JSON：\n{schema_str}\n\n"
            "不要使用 Markdown，不要解释——只输出 JSON 对象。"
        )
        messages = [{"role": "user", "content": description}]
        return self._call(system, messages)

    def extract_with_xml_scaffolding(self, description: str) -> str:
        """使用 XML 脚手架——Anthropic 特有的提示技术。"""
        schema_str = json.dumps(PRODUCT_SCHEMA_DESCRIPTION, indent=2)
        system = (
            "你是一名产品数据提取助手。请从产品描述中提取结构化信息，"
            "并以符合所提供模式的 JSON 返回。\n\n"
            "只能回答 JSON 对象——不要使用 Markdown 代码围栏，不要添加评论。"
        )
        # XML 标签帮助 Claude 解析输入结构
        user_content = (
            f"<schema>\n{schema_str}\n</schema>\n\n"
            f"<product_description>\n{description}\n</product_description>"
        )
        # 注意：Claude 4.6 移除了助手消息预填充，因此以前的
        # {"role": "assistant", "content": "{"} 技巧不再有效。
        # 单独使用 XML 标签仍能显著提高对结构的遵循程度。
        messages = [{"role": "user", "content": user_content}]
        return self._call(system, messages)

    def extract_with_native_schema(self, description: str) -> str:
        """通过 output_config 使用原生 JSON Schema 强制执行——保证 JSON 有效。"""
        system = (
            "你是一名产品数据提取助手。请从产品描述中提取结构化信息。"
        )
        messages = [{"role": "user", "content": description}]

        # 原生结构化输出：API 保证 JSON 有效且符合 Pydantic 模式
        response = self.client.beta.messages.parse(
            model=self.model,
            temperature=0.0,
            max_tokens=512,
            system=system,
            messages=messages,
            output_format=ProductExtraction,
        )
        self.token_tracker.track(response.usage)

        # parsed_output 是经过验证的 Pydantic 模型实例
        if response.parsed_output:
            result: str = response.parsed_output.model_dump_json(indent=2)
            return result
        return str(response.content[0].text)


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
        formatted = json.dumps(parsed, indent=2)
        syntax = Syntax(formatted, "json", theme="monokai")
        console.print(Panel(syntax, title=f"{method_name} [green]有效 JSON[/green]"))
    else:
        console.print(Panel(raw[:300], title=f"{method_name} [red]解析失败[/red]"))


METHOD_LABELS = [
    "A：基于提示词的 JSON",
    "B：XML 脚手架",
    "C：原生 JSON Schema",
]


def _run_method_a(console: Console, client: StructuredOutputClient) -> None:
    """运行基于提示词的 JSON 提取方法。"""
    schema_str = json.dumps(PRODUCT_SCHEMA_DESCRIPTION, indent=2)
    console.print("[dim]将模式嵌入系统提示词，并要求输出 JSON。[/dim]\n")
    prompt_a = (
        "**系统提示词：**\n"
        "```\n"
        "你是一名产品数据提取助手……\n"
        f"只输出符合以下模式的有效 JSON：\n{schema_str}\n"
        "不要使用 Markdown，不要解释——只输出 JSON 对象。\n"
        "```\n\n"
        "**用户消息：** _（原始产品描述）_\n"
    )
    console.print(Markdown(prompt_a))

    try:
        raw = client.extract_json_prompted(PRODUCT_DESCRIPTION)
        _display_result(console, "A：基于提示词的 JSON", raw)
    except Exception as e:
        logger.error("方法 A 出错：%s", e)


def _run_method_b(console: Console, client: StructuredOutputClient) -> None:
    """运行 XML 脚手架提取方法。"""
    schema_str = json.dumps(PRODUCT_SCHEMA_DESCRIPTION, indent=2)
    console.print(
        "[dim]用 XML 标签包裹输入，使 Claude 能清楚地区分模式与数据。[/dim]\n"
        "[dim]助手预填充过去常与此技术配合使用，但 Claude 4.6 已移除支持。[/dim]\n"
    )
    prompt_b = (
        "**系统提示词：**\n"
        "```\n"
        "你是一名产品数据提取助手……\n"
        "只能回答 JSON 对象。\n"
        "```\n\n"
        "**用户消息（XML 结构）：**\n"
        "```xml\n"
        f"<schema>\n{schema_str}\n</schema>\n\n"
        "<product_description>\n（此处为产品描述）\n</product_description>\n"
        "```\n"
    )
    console.print(Markdown(prompt_b))

    try:
        raw = client.extract_with_xml_scaffolding(PRODUCT_DESCRIPTION)
        _display_result(console, "B：XML 脚手架", raw)
    except Exception as e:
        logger.error("方法 B 出错：%s", e)


def _run_method_c(console: Console, client: StructuredOutputClient) -> None:
    """运行原生 JSON Schema 提取方法。"""
    console.print("[dim]通过 Pydantic 模型在 API 层强制执行——保证 JSON 有效。[/dim]\n")
    prompt_c = (
        "**系统提示词：**\n"
        "```\n"
        "你是一名产品数据提取助手……\n"
        "```\n\n"
        "**用户消息：** _（原始产品描述）_\n\n"
        "**output_format (Pydantic model):**\n"
        "```python\n"
        "class ProductExtraction(BaseModel):\n"
        "    name: str\n"
        "    category: str\n"
        "    price: float\n"
        "    features: list[str]\n"
        "    in_stock: bool\n"
        "```\n\n"
        "_API 保证响应符合此模式——无需解析。_\n"
    )
    console.print(Markdown(prompt_c))

    try:
        raw = client.extract_with_native_schema(PRODUCT_DESCRIPTION)
        _display_result(console, "C：原生模式", raw)
    except Exception as e:
        logger.error("方法 C 出错：%s", e)


def main() -> None:
    """使用三种结构化输出方法处理同一条产品描述。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()
    client = StructuredOutputClient("claude-sonnet-4-6", token_tracker)

    header = Panel(
        "[bold cyan]结构化输出与提示词脚手架[/bold cyan]\n\n"
        "比较从自由文本中提取结构化 JSON 的 3 种技术：\n"
        "  A. 基于提示词的 JSON——在系统提示词中要求返回 JSON\n"
        "  B. XML 脚手架——Anthropic 特有的提示技术\n"
        "  C. 原生 JSON Schema——通过 output_config 在 API 层强制执行（推荐）\n\n"
        f"[bold]产品描述：[/bold]\n{PRODUCT_DESCRIPTION}",
        title="提示工程 — Anthropic",
    )

    methods = {
        METHOD_LABELS[0]: _run_method_a,
        METHOD_LABELS[1]: _run_method_b,
        METHOD_LABELS[2]: _run_method_c,
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
