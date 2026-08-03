"""
结构化输出与验证（OpenAI）

演示如何使用 OpenAI API 实现同样的结构化输出目标。OpenAI 使用 `text.format` 强制
执行严格的 JSON Schema；它在概念上等同于 Anthropic 的 `output_config`（两者都使用
约束解码），但 API 接口不同。

本示例使用与 Anthropic 脚本相同的客服工单场景，演示简单和复杂模式。OpenAI 特有的
关键细节是：严格模式要求每一层嵌套对象都设置 `additionalProperties: false`，并将
所有属性标记为 `required`。

请先运行 Anthropic 脚本了解主要技术，再运行本脚本进行比较。
"""

import json
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from common import OpenAITokenTracker, setup_logging

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

MODEL = "gpt-4.1"


# ---------------------------------------------------------------------------
# Pydantic 模型——与 Anthropic 脚本使用相同模式
# ---------------------------------------------------------------------------


class TicketClassification(BaseModel):
    """包含类别、优先级和情感倾向的基础工单分类。"""

    category: Literal["billing", "technical", "account", "feature_request", "general"]
    priority: Literal["critical", "high", "medium", "low"]
    sentiment: Literal["positive", "neutral", "negative", "frustrated"]
    summary: str = Field(description="用一句话概括工单")


class Entity(BaseModel):
    """工单中提到的实体。"""

    name: str = Field(description="工单中提到的实体名称")
    type: Literal["product", "feature", "error_code", "account_id", "person"]
    context: str = Field(description="简要说明该实体是在什么语境中被提及的")


class ActionItem(BaseModel):
    """为解决工单而建议执行的操作。"""

    action: str = Field(description="需要执行的具体操作")
    assignee: Literal["support", "engineering", "billing", "account_manager"]
    urgency: Literal["immediate", "next_business_day", "backlog"]


class TicketAnalysis(BaseModel):
    """包含分类、实体和操作项的完整工单分析。"""

    classification: TicketClassification
    entities: list[Entity]
    action_items: list[ActionItem]
    requires_escalation: bool
    escalation_reason: str | None = None
    customer_tier: Literal["free", "pro", "enterprise"] | None = None


# ---------------------------------------------------------------------------
# 工单示例（与 Anthropic 脚本相同）
# ---------------------------------------------------------------------------

SAMPLE_TICKETS = [
    (
        "主题：Pro 订阅被重复扣费\n"
        "您好，我的 Pro 订阅本月被扣了两次款——1 月 3 日扣了 49.99 美元，"
        "1 月 5 日又扣了一次。我的账户 ID 是 ACC-78234。这已经是第三次发生这种事了，"
        "我真的非常恼火。请尽快退还重复扣取的费用。若今天还不能解决，我会取消订阅。"
    ),
    (
        "主题：企业评估期间遇到 SSO 阻断问题\n"
        "我们正在为一支由 200 名工程师组成的团队评估贵公司的产品。与 Okta 的 SSO 集成"
        "运行良好，但我们遇到了一个阻断问题——同步成员超过 50 人的用户组时，SCIM 配置"
        "端点会返回 500 错误（错误代码：SCIM-ERR-4012）。另外，是否可以提供批量采购价？"
        "我们目前与 Acme Corp 的合同将在下个月续约。联系人：工程副总裁 Sarah Chen。"
    ),
]

SYSTEM_PROMPT = (
    "你是一个客服工单分析系统。请分析客户支持工单并提取结构化数据。"
    "分类务必准确，并提取所有相关实体和操作项。"
)


# ---------------------------------------------------------------------------
# 模式转换辅助函数
# ---------------------------------------------------------------------------


def _pydantic_to_openai_schema(name: str, model: type[BaseModel]) -> dict:
    """将 Pydantic 模型转换为 OpenAI 的 response_format JSON Schema 定义。

    OpenAI 严格模式具有不同于标准 JSON Schema 的特定要求：
    - 每个对象层级都要设置 `additionalProperties: false`
    - 所有属性都必须列入 `required`（包括可选属性）
    这些约束会递归应用，以处理嵌套模型。
    """
    schema = model.model_json_schema()
    _add_strict_constraints(schema)
    return {
        "type": "json_schema",
        "name": name,
        "strict": True,
        "schema": schema,
    }


def _add_strict_constraints(schema: dict) -> None:
    """递归地为模式中的所有对象类型添加 additionalProperties: false。"""
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        if "properties" in schema:
            schema.setdefault("required", list(schema["properties"].keys()))
    for value in schema.values():
        if isinstance(value, dict):
            _add_strict_constraints(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _add_strict_constraints(item)
    # 处理 $defs（Pydantic 会将嵌套模型的模式放在这里）
    if "$defs" in schema:
        for defn in schema["$defs"].values():
            _add_strict_constraints(defn)


# ---------------------------------------------------------------------------
# 核心提取器类
# ---------------------------------------------------------------------------


class StructuredExtractor:
    """使用 OpenAI 的原生模式约束提取结构化数据。"""

    def __init__(self, model: str, token_tracker: OpenAITokenTracker):
        self.client = OpenAI()
        self.model = model
        self.token_tracker = token_tracker

    def extract(
        self,
        text: str,
        model_class: type[BaseModel] = TicketClassification,
        schema_name: str = "ticket_classification",
    ) -> BaseModel | None:
        """使用 OpenAI 的 text.format 和严格 JSON Schema 提取结构化数据。"""
        schema = _pydantic_to_openai_schema(schema_name, model_class)
        try:
            response = self.client.responses.create(
                model=self.model,
                temperature=0.0,
                max_output_tokens=2048,
                instructions=SYSTEM_PROMPT,
                input=f"分析以下工单：\n\n{text}",
                text={"format": schema},
            )
            if hasattr(response, "usage") and response.usage:
                self.token_tracker.track(response.usage)

            raw_json = response.output_text
            if raw_json:
                data = json.loads(raw_json)
                return model_class(**data)
        except Exception as e:
            logger.error("模式提取失败：%s", e)
        return None


# ---------------------------------------------------------------------------
# 显示辅助函数
# ---------------------------------------------------------------------------


def _display_result(console: Console, title: str, result: BaseModel | None) -> None:
    """以格式化 JSON 显示 Pydantic 模型。"""
    if result:
        formatted = json.dumps(result.model_dump(), indent=2, default=str)
        syntax = Syntax(formatted, "json", theme="monokai")
        console.print(Panel(syntax, title=f"{title} [green]成功[/green]"))
    else:
        console.print(Panel("[red]提取失败[/red]", title=title))


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main() -> None:
    """通过 OpenAI 的结构化输出运行客服工单分析。"""
    console = Console()
    token_tracker = OpenAITokenTracker()
    extractor = StructuredExtractor(MODEL, token_tracker)

    console.print(
        Panel(
            "[bold cyan]结构化输出——OpenAI 对比[/bold cyan]\n\n"
            "OpenAI 使用严格 JSON Schema 的 `text.format`，在概念上与 Anthropic 的\n"
            "`output_config` 相同——两者都使用约束解码来保证输出有效。区别在于 API\n"
            "接口，而非底层机制。\n\n"
            "[bold]OpenAI 特有的细节：[/bold]严格模式要求设置 "
            "`additionalProperties: false`\n"
            "并在每一层嵌套对象中将所有属性列入 `required`。",
            title="高级技术——OpenAI",
        )
    )

    console.print(
        Markdown(
            "**Anthropic 与 OpenAI——真正的区别：**\n\n"
            "| 方面 | Anthropic | OpenAI |\n"
            "|--------|-----------|--------|\n"
            "| 约束解码 | `output_config=Model` | 使用严格模式的 `text.format` |\n"
            "| 基于工具的提取 | `tool_use` + `tool_choice` | 同样支持（此处未演示） |\n"
            "| Pydantic 集成 | 直接传入模型 | 需要模式转换辅助函数 |\n"
            "| 严格模式要求 | 无 | 所有层级均需设置 `additionalProperties: false` |\n\n"
            "两者都保证 JSON 有效——可靠性相同，API 不同。\n"
        )
    )

    # A 部分：简单扁平模式
    console.print(
        "\n[bold]A 部分：简单模式[/bold]——`TicketClassification`（扁平，4 个字段）\n"
    )
    ticket_simple = SAMPLE_TICKETS[0]
    console.print(Panel(ticket_simple, title="输入工单（简单）"))

    result_simple = extractor.extract(ticket_simple)
    _display_result(console, "简单模式提取", result_simple)

    token_tracker.report()
    token_tracker.reset()

    # B 部分：复杂嵌套模式
    console.print(
        "\n[bold]B 部分：复杂模式[/bold]——`TicketAnalysis` "
        "（嵌套：分类 + 实体 + 操作项）\n"
    )
    console.print(
        "[dim]使用与 Anthropic 脚本相同的 Pydantic 模型——只有转换层不同"
        "（需要递归添加严格约束）。[/dim]\n"
    )
    ticket_complex = SAMPLE_TICKETS[1]
    console.print(Panel(ticket_complex, title="输入工单（复杂）"))

    result_complex = extractor.extract(
        ticket_complex,
        model_class=TicketAnalysis,
        schema_name="ticket_analysis",
    )
    _display_result(console, "复杂模式提取", result_complex)

    token_tracker.report()


if __name__ == "__main__":
    main()
