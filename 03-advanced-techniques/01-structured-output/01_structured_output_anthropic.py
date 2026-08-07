"""
结构化输出与验证（Anthropic）

演示从 Claude 提取可靠结构化数据的四种生产级技术，内容由浅入深：

1. 将工具调用用作结构化输出——通过 tool_choice 强制生成结构化响应（简单 + 复杂）
2. 原生结构化输出——API 级约束解码（保证有效）
3. 验证与重试——通过错误反馈循环实现自修复提取
4. 批量提取——通过一次调用处理多个项目

所有技术都使用同一个真实业务场景——客服工单分析，便于直接比较不同方法的结果。
"""

import json
from typing import Any, Literal

import anthropic
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field, ValidationError, model_validator
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from common import AnthropicTokenTracker, interactive_menu, setup_logging

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

MODEL = "deepseek-v4-flash"

# ---------------------------------------------------------------------------
# Pydantic 模型——复杂度逐步提升
# ---------------------------------------------------------------------------


# 简单模式：扁平分类
class TicketClassification(BaseModel):
    """包含类别、优先级和情感倾向的基础工单分类。"""

    category: Literal["billing", "technical", "account", "feature_request", "general"]
    priority: Literal["critical", "high", "medium", "low"]
    sentiment: Literal["positive", "neutral", "negative", "frustrated"]
    summary: str = Field(description="用一句话概括工单")


# 复杂模式：嵌套提取
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

    # JSON Schema 无法表达的自定义业务验证
    @model_validator(mode="after")
    def check_escalation_consistency(self) -> "TicketAnalysis":
        """如果需要升级处理，则必须提供原因。"""
        if self.requires_escalation and not self.escalation_reason:
            raise ValueError("requires_escalation 为 True 时必须提供 escalation_reason")
        return self


class TicketBatch(BaseModel):
    """对多张工单进行批量分析。"""

    analyses: list[TicketAnalysis]
    batch_summary: str = Field(description="本批工单的总体摘要")
    priority_distribution: dict[str, int] = Field(description="各优先级对应的工单数量")


# ---------------------------------------------------------------------------
# 客服工单示例（简单 → 中等 → 困难）
# ---------------------------------------------------------------------------

SAMPLE_TICKETS = [
    (
        "主题：Pro 订阅被重复扣费\n"
        "您好，我的 Pro 订阅本月被扣了两次款——1 月 3 日扣了 49.99 美元，"
        "1 月 5 日又扣了一次。我的账户 ID 是 ACC-78234。这已经是第三次发生这种事了，"
        "我真的非常恼火。请尽快退还重复扣取的费用。若今天还不能解决，我会取消订阅。"
    ),
    (
        "主题：企业版套餐的 API 速率限制问题\n"
        "批量处理超过 50 个项目时，API 总是返回 429 错误。"
        "我使用的是企业版套餐，文档中说速率限制应为每分钟 1000 次。"
        "另外，能否在响应中添加 retry-after 标头？这会非常有帮助。"
        "当前使用 Python SDK v3.2.1。"
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
# 核心提取器类
# ---------------------------------------------------------------------------


class StructuredExtractor:
    """使用多种技术从非结构化文本中提取结构化数据。"""

    def __init__(self, model: str, token_tracker: AnthropicTokenTracker):
        self.client = anthropic.Anthropic()
        self.model = model
        self.token_tracker = token_tracker

    # -- 技术 1：将工具调用用作结构化输出 --

    def extract_with_tool_use(
        self,
        text: str,
        model_class: type[BaseModel] = TicketClassification,
        tool_name: str = "classify_ticket",
        tool_description: str = "对客服工单进行分类。",
    ) -> BaseModel | None:
        """使用 tool_choice 强制模型通过工具定义输出结构化数据。"""
        tool = self._pydantic_to_tool(tool_name, tool_description, model_class)
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=21333,
                system=SYSTEM_PROMPT,
                tools=[tool],
                messages=[{"role": "user", "content": f"分析以下工单：\n\n{text}"}],
            )
            self.token_tracker.track(response.usage)

            for block in response.content:
                if block.type == "tool_use":
                    return model_class(**block.input)
        except Exception as e:
            logger.error("工具调用提取失败：%s", e)
        return None

    # -- 技术 2：原生结构化输出（约束解码）--

    def extract_with_native_schema(self, text: str) -> TicketClassification | None:
        """使用 Anthropic 原生约束解码进行提取，保证结果有效。"""
        try:
            response = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=21333,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"分析以下工单：\n\n{text}"}],
                output_config={"format": TicketClassification},
            )
            self.token_tracker.track(response.usage)

            result: TicketClassification | None = response.parsed_output
            if result:
                return result
        except Exception as e:
            logger.error("原生模式提取失败：%s", e)
        return None

    # -- 技术 3：验证与重试（自修复）--

    def extract_with_validation_retry(
        self, text: str, max_retries: int = 3
    ) -> TicketAnalysis | None:
        """使用验证循环提取；失败时反馈错误并重试。"""
        tool = self._pydantic_to_tool(
            name="analyze_ticket",
            description="对客服工单进行完整分析。",
            model=TicketAnalysis,
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"执行完整分析：\n\n{text}"}
        ]

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=21333,
                    system=SYSTEM_PROMPT,
                    tools=[tool],
                    messages=messages,
                )
                self.token_tracker.track(response.usage)

                for block in response.content:
                    if block.type == "tool_use":
                        raw = block.input
                        # 使用 Pydantic 验证（包括自定义业务规则）
                        result = TicketAnalysis(**raw)
                        logger.info("第 %d 次尝试：验证通过", attempt)
                        return result

            except ValidationError as e:
                logger.warning("第 %d 次尝试：验证失败——%s", attempt, e)
                if attempt < max_retries:
                    # 将错误反馈给大语言模型进行修正
                    messages = [
                        {"role": "user", "content": f"执行完整分析：\n\n{text}"},
                        {"role": "assistant", "content": response.content},
                        {
                            "role": "user",
                            "content": (
                                f"输出未通过验证：\n{e}\n\n"
                                "请修复问题后重试。关键规则：\n"
                                "- requires_escalation 为 true 时，escalation_reason 必须"
                                "是非空字符串\n"
                                "- 所有枚举值必须完全匹配"
                            ),
                        },
                    ]
            except Exception as e:
                logger.error("第 %d 次尝试：发生意外错误——%s", attempt, e)
                break

        logger.error("全部 %d 次尝试均失败", max_retries)
        return None

    # -- 技术 4：批量提取 --

    def extract_batch(self, texts: list[str]) -> TicketBatch | None:
        """通过一次调用从多张工单中提取结构化数据。"""
        tool = self._pydantic_to_tool(
            name="batch_analyze",
            description="分析多张客服工单并提供批次摘要。",
            model=TicketBatch,
        )
        numbered_tickets = "\n\n".join(
            f"--- 工单 {i + 1} ---\n{text}" for i, text in enumerate(texts)
        )
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=21333,
                system=SYSTEM_PROMPT,
                tools=[tool],
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"分析下面全部 {len(texts)} 张工单，并提供"
                            f"批量分析：\n\n{numbered_tickets}"
                        ),
                    }
                ],
            )
            self.token_tracker.track(response.usage)

            for block in response.content:
                if block.type == "tool_use":
                    return TicketBatch(**block.input)
        except Exception as e:
            logger.error("批量提取失败：%s", e)
        return None

    # -- 辅助方法 --

    def _pydantic_to_tool(
        self, name: str, description: str, model: type[BaseModel]
    ) -> dict[str, Any]:
        """将任意 Pydantic 模型转换为 Anthropic 工具定义。"""
        return {
            "name": name,
            "description": description,
            "input_schema": model.model_json_schema(),
        }


# ---------------------------------------------------------------------------
# 显示辅助函数（Rich 界面）
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
# 菜单处理函数
# ---------------------------------------------------------------------------

TECHNIQUE_LABELS = [
    "1：将工具调用用作结构化输出（简单 + 复杂）",
    "2：原生结构化输出（约束解码）",
    "3：验证与重试（自修复）",
    "4：批量提取（多个项目）",
]


def _run_tool_use(console: Console, extractor: StructuredExtractor) -> None:
    """技术 1：使用 tool_choice 按简单和复杂模式进行提取。"""
    console.print(
        "[dim]定义一个 input_schema 即所需输出模式的工具，再通过 tool_choice "
        "调用它，从而强制生成结构化输出。[/dim]\n"
    )
    console.print(
        Markdown(
            "**工作原理：** 定义工具 → 设置 `tool_choice` 强制调用 → "
            "将 `block.input` 作为结构化数据提取。\n\n"
            "**关键认识：** 使用 `model.model_json_schema()` 从 Pydantic 模型生成工具模式，"
            "不要为复杂结构手写 JSON Schema。\n\n"
            "**可靠性：** 高——API 会按照模式验证工具输入。\n"
        )
    )

    # A 部分：简单扁平模式
    console.print("[bold]A 部分：简单模式[/bold]——`TicketClassification`（扁平，4 个字段）\n")
    ticket_simple = SAMPLE_TICKETS[0]
    console.print(Panel(ticket_simple, title="输入工单（简单）"))

    result_simple = extractor.extract_with_tool_use(ticket_simple)
    _display_result(console, "简单模式提取", result_simple)

    # B 部分：复杂嵌套模式
    console.print(
        "\n[bold]B 部分：复杂模式[/bold]——`TicketAnalysis` "
        "（嵌套：分类 + 实体 + 操作项，10 多个字段）\n"
    )
    ticket_complex = SAMPLE_TICKETS[2]
    console.print(Panel(ticket_complex, title="输入工单（复杂）"))

    result_complex = extractor.extract_with_tool_use(
        ticket_complex,
        model_class=TicketAnalysis,
        tool_name="analyze_ticket",
        tool_description="对客服工单进行完整分析。",
    )
    _display_result(console, "复杂模式提取", result_complex)


def _run_native_schema(console: Console, extractor: StructuredExtractor) -> None:
    """技术 2：使用原生约束解码进行提取。"""
    console.print(
        "[dim]使用 Anthropic 的原生结构化输出——模型从根本上无法生成无效 JSON。"
        "该方式在解码器层使用约束解码。[/dim]\n"
    )
    console.print(
        Markdown(
            "**工作原理：** 将 Pydantic 模型作为 `output_config` 传入 → API 保证"
            "响应与模式完全匹配。\n\n"
            "**模式：** `TicketClassification`（扁平，4 个字段）\n\n"
            "**可靠性：** 有保证——解码器层强制约束，不会发生解析错误。\n"
        )
    )

    ticket = SAMPLE_TICKETS[0]
    console.print(Panel(ticket, title="输入工单"))

    result = extractor.extract_with_native_schema(ticket)
    _display_result(console, "原生模式提取", result)


def _run_validation_retry(console: Console, extractor: StructuredExtractor) -> None:
    """技术 3：通过验证与重试实现自修复提取。"""
    console.print(
        "[dim]当模式验证还不够时，可添加自定义业务规则。失败后将验证错误反馈给"
        "大语言模型，让其自行修正。[/dim]\n"
    )
    console.print(
        Markdown(
            "**工作原理：** 提取 → 使用 Pydantic 验证（包括自定义 `@model_validator` "
            "规则）→ 失败时将错误反馈给大语言模型 → 重试。\n\n"
            "**自定义规则：** `requires_escalation` 为 True 时必须提供 "
            "`escalation_reason`（单靠 JSON Schema 无法表达）。\n\n"
            "**最大重试次数：** 3 次，并累积错误信息。\n"
        )
    )

    # 使用工单 3——它很可能需要升级处理（企业评估、阻断问题）
    ticket = SAMPLE_TICKETS[2]
    console.print(Panel(ticket, title="输入工单（需要升级处理）"))

    result = extractor.extract_with_validation_retry(ticket)
    _display_result(console, "验证与重试提取", result)


def _run_batch(console: Console, extractor: StructuredExtractor) -> None:
    """技术 4：从多个项目中批量提取。"""
    console.print(
        "[dim]通过一次 API 调用处理多张工单。模型会为每张工单提取结构化数据，"
        "并提供批次摘要。[/dim]\n"
    )
    console.print(
        Markdown(
            "**工作原理：** 在一个提示词中发送所有工单 → 提取包含 "
            "`list[TicketAnalysis]` + 摘要 + 优先级分布的 `TicketBatch`。\n\n"
            "**使用场景：** 处理工单队列的生产数据管道。\n\n"
            "**权衡：** 单次调用（成本更低）与逐项调用（更可靠）。批量处理适合 "
            "3～10 个项目；超过此数量时，应并行逐项调用。\n"
        )
    )

    for i, ticket in enumerate(SAMPLE_TICKETS):
        console.print(Panel(ticket, title=f"工单 {i + 1}"))

    result = extractor.extract_batch(SAMPLE_TICKETS)
    _display_result(console, "批量提取", result)


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main() -> None:
    """使用四种结构化输出技术运行客服工单分析。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()
    extractor = StructuredExtractor(MODEL, token_tracker)

    header = Panel(
        "[bold cyan]结构化输出与验证[/bold cyan]\n\n"
        "从 Claude 提取可靠结构化数据的四种技术：\n"
        "  1. 工具调用——通过 tool_choice 强制结构化输出（简单 + 复杂）\n"
        "  2. 原生模式——约束解码（保证有效）\n"
        "  3. 验证与重试——通过错误反馈实现自修复\n"
        "  4. 批量提取——通过一次调用处理多个项目\n\n"
        "[bold]场景：[/bold]客服工单分析（分类、实体、操作）",
        title="高级技术——Anthropic",
    )

    handlers = {
        TECHNIQUE_LABELS[0]: _run_tool_use,
        TECHNIQUE_LABELS[1]: _run_native_schema,
        TECHNIQUE_LABELS[2]: _run_validation_retry,
        TECHNIQUE_LABELS[3]: _run_batch,
    }

    try:
        while True:
            selection = interactive_menu(
                console,
                TECHNIQUE_LABELS,
                title="选择一种技术",
                header=header,
            )
            if not selection:
                break

            console.print(f"\n[bold yellow]━━━ {selection} ━━━[/bold yellow]\n")

            try:
                handlers[selection](console, extractor)
            except Exception as e:
                logger.error("技术示例出错：%s", e)

            token_tracker.report()
            token_tracker.reset()

            console.print("\n[dim]按 Enter 键继续……[/dim]")
            input()

    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")


if __name__ == "__main__":
    main()
