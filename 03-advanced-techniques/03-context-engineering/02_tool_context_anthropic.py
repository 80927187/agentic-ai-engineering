"""
工具输出上下文工程（DeepSeek，使用 Anthropic 兼容接口）

演示在智能体上下文窗口中管理工具输出的三种策略：原样注入、截断（限制字符数）
和摘要（由 LLM 提取）。程序使用会返回大量仿真 JSON 数据的模拟业务工具，
展示工具输出如何成为上下文的主要消耗来源。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from common import AnthropicTokenTracker, interactive_menu, setup_logging

# 从根目录的 .env 文件加载环境变量
load_dotenv(find_dotenv())

# 配置日志
logger = setup_logging(__name__)

# 模型配置
MODEL = "deepseek-v4-flash"

SYSTEM_PROMPT = (
    "你是一名业务数据助理，可以使用 CRM、订单和产品工具。"
    "请调用适当的工具回答用户问题。回答应简洁，并引用工具结果中的具体数据。"
)

# 人为降低预算，让演示能够快速触发压缩
MAX_CONTEXT_TOKENS = 4096
RESPONSE_RESERVE = 2048
RECENT_MESSAGES_TO_KEEP = 4
MAX_TOOL_TURNS = 100

# 策略常量
TRUNCATE_MAX_CHARS = 500

STRATEGIES = {
    "naive": "直接注入原始工具输出（基线策略——会快速填满上下文）",
    "truncate": f"将工具输出限制为 {TRUNCATE_MAX_CHARS} 个字符（免费、有损）",
    "summarize": "由 LLM 提取工具输出中的关键事实（额外调用 API，但保留含义）",
}

# --- 模拟业务工具 ---

TOOLS = [
    {
        "name": "lookup_customer",
        "description": (
            "按姓名查找客户。返回联系方式、地址、账户历史、偏好和近期支持工单。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "要搜索的客户姓名",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_order_history",
        "description": (
            "获取客户的订单历史。返回包含订单项、总额、日期和履约状态的订单列表。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "客户 ID（例如 CUST-1001）",
                },
            },
            "required": ["customer_id"],
        },
    },
    {
        "name": "search_products",
        "description": (
            "按关键词搜索产品目录。返回匹配产品的说明、规格、价格和供应状态。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或短语",
                },
            },
            "required": ["query"],
        },
    },
]


DB_PATH = Path(__file__).parent / "database.json"


def _estimate_tokens(*values: Any) -> int:
    """在网关不支持 count_tokens 时，按 UTF-8 字节数近似估算令牌数。"""
    serialized = json.dumps(values, ensure_ascii=False, default=str)
    return max(1, (len(serialized.encode("utf-8")) + 3) // 4)


class MockDatabaseService:
    """从 JSON 加载的模拟业务数据库。"""

    def __init__(self, db_path: Path) -> None:
        self.data: dict[str, Any] = json.loads(db_path.read_text(encoding="utf-8"))
        logger.info("已从 %s 加载模拟数据库", db_path.name)

    def get_customer(self, name: str) -> dict:
        """按姓名查找客户，并返回第一个匹配项。"""
        name_lower = name.lower()
        for customer in self.data["customers"].values():
            if name_lower in customer["name"].lower():
                match: dict = customer
                return match
        return {"error": f"未找到客户“{name}”"}

    def get_orders(self, customer_id: str) -> dict:
        """按客户 ID 获取订单历史。"""
        if customer_id in self.data["orders"]:
            orders: dict = self.data["orders"][customer_id]
            return orders
        return {"error": f"未找到客户“{customer_id}”的订单"}

    def search_products(self, query: str) -> dict:
        """按关键词搜索产品目录。"""
        query_lower = query.lower()
        matches = [
            p
            for p in self.data["products"]
            if query_lower in json.dumps(p, ensure_ascii=False).lower()
        ]
        # 如果没有具体匹配项，则返回所有产品（模拟宽泛搜索）
        results = matches if matches else self.data["products"]
        return {"query": query, "total_results": len(results), "products": results}


# --- 数据类（自包含，与脚本 01 使用相同模式）---


@dataclass
class ContextBudget:
    """上下文各组成部分的令牌预算分配。"""

    max_context: int
    system_tokens: int = 0
    response_reserve: int = RESPONSE_RESERVE

    @property
    def history_budget(self) -> int:
        """可供对话历史使用的令牌数。"""
        return self.max_context - self.system_tokens - self.response_reserve


@dataclass
class TokenSnapshot:
    """用于预算面板的令牌用量快照。"""

    system: int = 0
    history: int = 0
    history_budget: int = 0
    reserve: int = 0
    message_count: int = 0
    compression_count: int = 0


# --- 核心智能体 ---


class ToolContextAgent:
    """演示工具输出上下文管理策略的智能体。"""

    def __init__(
        self,
        model: str,
        strategy: str,
        max_context: int,
        token_tracker: AnthropicTokenTracker,
        db: MockDatabaseService,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.strategy = strategy
        self.token_tracker = token_tracker
        self.db = db
        self.messages: list[dict] = []
        self.budget = ContextBudget(max_context=max_context)
        self.compression_count = 0

        # 工具名称 → 服务方法映射
        self.tool_handlers: dict[str, Any] = {
            "lookup_customer": lambda **kw: self.db.get_customer(kw["name"]),
            "get_order_history": lambda **kw: self.db.get_orders(kw["customer_id"]),
            "search_products": lambda **kw: self.db.search_products(kw["query"]),
        }

        # 初始化时只估算一次系统提示词和工具定义的令牌数
        self.budget.system_tokens = self._count_tokens([])
        logger.info(
            "上下文预算——系统+工具：%d，历史记录：%d，预留：%d，策略：%s",
            self.budget.system_tokens,
            self.budget.history_budget,
            self.budget.response_reserve,
            self.strategy,
        )

    def chat(self, user_input: str) -> str:
        """智能体循环：发送 → 检测工具调用 → 执行 → 处理结果 → 继续循环。"""
        self.messages.append({"role": "user", "content": user_input})

        # 如果历史记录超出预算，则在发送前压缩
        self._compress_if_needed()

        for _turn in range(MAX_TOOL_TURNS):
            logger.info(
                "正在发送请求（消息数：%d，历史记录令牌数：约 %d/%d）",
                len(self.messages),
                self._count_tokens(self.messages) - self.budget.system_tokens,
                self.budget.history_budget,
            )

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.budget.response_reserve,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self.messages,
            )

            self.token_tracker.track(response.usage)

            # 收集工具调用块；文本由 _extract_text 统一提取，以兼容思考块
            tool_uses = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_uses.append(block)

            # 将助手响应添加到历史记录
            self.messages.append({"role": "assistant", "content": response.content})

            # 如果没有工具调用，则返回文本响应
            if response.stop_reason != "tool_use" or not tool_uses:
                return _extract_text(response)

            # 执行工具并对结果应用所选策略
            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use.name
                tool_input = tool_use.input

                logger.info(
                    "正在执行工具：%s(%s)",
                    tool_name,
                    json.dumps(tool_input, ensure_ascii=False),
                )

                # 通过数据库服务执行工具
                raw_result = json.dumps(
                    self.tool_handlers[tool_name](**tool_input), indent=2, ensure_ascii=False
                )

                # 对工具输出应用上下文策略
                processed_result = self._process_tool_result(tool_name, raw_result)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": processed_result,
                    }
                )

            # 添加工具结果，并进入下一轮响应循环
            self.messages.append({"role": "user", "content": tool_results})

            # 如果工具结果使上下文超出预算，则再次压缩
            self._compress_if_needed()

        raise RuntimeError(f"工具调用达到最大轮数（{MAX_TOOL_TURNS}），已停止继续调用。")

    def _process_tool_result(self, tool_name: str, raw_result: str) -> str:
        """将工具输出注入上下文前，应用所选策略。"""
        raw_chars = len(raw_result)

        if self.strategy == "naive":
            logger.info("[原样注入] 工具 %s：直接注入 %d 个字符", tool_name, raw_chars)
            return raw_result

        if self.strategy == "truncate":
            processed = self._truncate_result(raw_result)
            logger.info("[截断] 工具 %s：%d → %d 个字符", tool_name, raw_chars, len(processed))
            return processed

        if self.strategy == "summarize":
            processed = self._summarize_result(tool_name, raw_result)
            logger.info("[摘要] 工具 %s：%d → %d 个字符", tool_name, raw_chars, len(processed))
            return processed

        return raw_result

    def _truncate_result(self, result: str) -> str:
        """将结果限制为 TRUNCATE_MAX_CHARS 个字符，并添加截断标记。"""
        if len(result) <= TRUNCATE_MAX_CHARS:
            return result
        return result[:TRUNCATE_MAX_CHARS] + "\n……[已截断——输出超出限制]"

    def _summarize_result(self, tool_name: str, result: str) -> str:
        """调用 LLM 从工具输出中提取关键事实。"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=21333,
            system=(
                "从这份工具输出中提取关键事实，并生成简洁摘要。"
                "保留所有姓名、ID、数字、日期和状态。"
                "使用单层项目符号列表。内容应简短但完整。"
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"工具：{tool_name}\n\n输出：\n{result}",
                }
            ],
        )

        self.token_tracker.track(response.usage)
        return _extract_text(response)

    def _count_tokens(self, messages: list[dict]) -> int:
        """在 DeepSeek 兼容网关未提供计数端点时，本地估算令牌数。"""
        msgs = messages if messages else [{"role": "user", "content": "."}]
        return _estimate_tokens(SYSTEM_PROMPT, TOOLS, msgs)

    def _compress_if_needed(self) -> None:
        """如果历史记录超出预算，则总结最早的消息。"""
        history_tokens = self._count_tokens(self.messages) - self.budget.system_tokens

        if history_tokens <= self.budget.history_budget:
            return

        logger.info(
            "历史记录（%d 个令牌）超出预算（%d 个令牌）——正在压缩",
            history_tokens,
            self.budget.history_budget,
        )

        # 拆分消息：近期消息保留原文，其余消息生成摘要
        keep_count = min(RECENT_MESSAGES_TO_KEEP, len(self.messages))
        old_messages = self.messages[:-keep_count] if keep_count > 0 else self.messages
        recent_messages = self.messages[-keep_count:] if keep_count > 0 else []

        if not old_messages:
            logger.warning("没有可压缩的消息——预算可能过小")
            return

        old_tokens = self._count_tokens(old_messages) - self.budget.system_tokens

        # 总结较早的消息
        summary = self._summarize_messages(old_messages)

        # 用摘要替换较早的消息
        summary_message = {
            "role": "user",
            "content": (
                f"[先前对话摘要]\n{summary}\n"
                "[摘要结束——请从这里继续对话]"
            ),
        }

        # 确保角色交替：摘要（用户消息）之后接近期消息
        if recent_messages and recent_messages[0]["role"] == "user":
            self.messages = [
                summary_message,
                {"role": "assistant", "content": "明白，我已经掌握了对话上下文。"},
                *recent_messages,
            ]
        else:
            self.messages = [summary_message, *recent_messages]

        new_tokens = self._count_tokens(self.messages) - self.budget.system_tokens
        self.compression_count += 1

        logger.info(
            "已压缩 %d 条消息：%d → %d 个令牌（节省 %d 个令牌）",
            len(old_messages),
            old_tokens,
            new_tokens,
            old_tokens - new_tokens,
        )

    def _summarize_messages(self, messages: list[dict]) -> str:
        """使用 LLM 总结一组包含工具交互的消息。"""
        parts = []
        for m in messages:
            role = "用户" if m["role"] == "user" else "助手"
            content = m["content"]
            # 处理工具结果消息（字典列表）
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        texts.append(f"[工具结果：{item.get('content', '')[:200]}……]")
                    elif isinstance(item, dict) and hasattr(item, "text"):
                        texts.append(str(item))
                    else:
                        texts.append(str(item))
                content = "\n".join(texts)
            elif not isinstance(content, str):
                content = str(content)
            parts.append(f"{role}：{content}")

        transcript = "\n".join(parts)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=21333,
            system=(
                "简洁地总结以下对话。"
                "保留关键事实、数据点、客户姓名、订单 ID 和工具结果。"
                "使用第三人称和过去时态。文字应简短但全面。"
            ),
            messages=[{"role": "user", "content": transcript}],
        )

        self.token_tracker.track(response.usage)
        return _extract_text(response)

    def get_token_snapshot(self) -> TokenSnapshot:
        """用于可视化的预算状态。"""
        history_tokens = 0
        if self.messages:
            history_tokens = self._count_tokens(self.messages) - self.budget.system_tokens

        return TokenSnapshot(
            system=self.budget.system_tokens,
            history=history_tokens,
            history_budget=self.budget.history_budget,
            reserve=self.budget.response_reserve,
            message_count=len(self.messages),
            compression_count=self.compression_count,
        )


# --- 用户界面 ---


def _extract_text(response: Any) -> str:
    """跳过思考块，只提取模型响应中的文本块。"""
    text_parts = [block.text for block in response.content if block.type == "text"]
    if not text_parts:
        block_types = [block.type for block in response.content]
        raise ValueError(
            f"模型响应中没有文本内容（stop_reason={response.stop_reason}，"
            f"内容块={block_types}，output_tokens={response.usage.output_tokens}）。"
        )
    return "\n\n".join(text_parts)


def _render_budget_display(console: Console, snapshot: TokenSnapshot) -> None:
    """渲染上下文预算可视化面板。"""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("组成部分", style="dim")
    table.add_column("令牌数", justify="right")
    table.add_column("用量", min_width=30)

    usage_ratio = snapshot.history / snapshot.history_budget if snapshot.history_budget > 0 else 0
    bar_width = 25
    filled = int(usage_ratio * bar_width)
    bar_color = "green" if usage_ratio < 0.7 else "yellow" if usage_ratio < 0.9 else "red"
    bar = f"[{bar_color}]{'█' * filled}[/{bar_color}][dim]{'░' * (bar_width - filled)}[/dim]"

    table.add_row("系统+工具", f"[cyan]{snapshot.system:,}[/cyan]", "[dim]固定[/dim]")
    table.add_row(
        "历史记录",
        f"[{bar_color}]{snapshot.history:,}[/{bar_color}] / {snapshot.history_budget:,}",
        bar,
    )
    table.add_row("响应预留", f"[cyan]{snapshot.reserve:,}[/cyan]", "[dim]max_tokens[/dim]")

    footer = f"消息数：{snapshot.message_count}"
    if snapshot.compression_count > 0:
        footer += f" │ 压缩次数：{snapshot.compression_count}"

    console.print(
        Panel(table, title="上下文预算", subtitle=footer, border_style="dim", padding=(0, 1))
    )


def main() -> None:
    """工具上下文工程演示的主编排函数。"""
    console = Console()

    # 选择策略
    strategy_items = [f"{name} — {desc}" for name, desc in STRATEGIES.items()]
    header = Panel(
        "[bold cyan]工具输出上下文工程[/bold cyan]\n\n"
        "工具输出是智能体系统中最大的上下文消耗来源。\n"
        "一次 API 调用就可能返回超过 1,000 个令牌的 JSON。\n\n"
        "请选择一种策略，观察它如何影响上下文用量：",
        border_style="cyan",
    )

    selected = interactive_menu(
        console,
        items=strategy_items,
        title="上下文策略",
        header=header,
    )

    if selected is None:
        console.print("[yellow]正在退出。[/yellow]")
        return

    # 从选择结果中提取策略名称
    strategy = selected.split(" — ")[0]
    console.clear()

    token_tracker = AnthropicTokenTracker()
    db = MockDatabaseService(DB_PATH)
    agent = ToolContextAgent(MODEL, strategy, MAX_CONTEXT_TOKENS, token_tracker, db)

    # 针对不同策略显示相应的欢迎说明
    strategy_hints = {
        "naive": (
            "原始工具输出将直接注入上下文。\n"
            "观察 2～3 次工具调用如何填满全部预算！"
        ),
        "truncate": (
            f"工具输出将限制为 {TRUNCATE_MAX_CHARS} 个字符。\n"
            "此策略没有额外成本，但可能丢失结果末尾的重要数据。"
        ),
        "summarize": (
            "注入前，LLM 会从每份工具输出中提取关键事实。\n"
            "每次使用工具都要额外调用一次 API，但可以保留含义。"
        ),
    }

    console.print(
        Panel(
            f"[bold cyan]策略：{strategy.upper()}[/bold cyan]\n\n"
            f"{strategy_hints[strategy]}\n\n"
            f"上下文预算：总计 {MAX_CONTEXT_TOKENS:,} 个令牌，"
            f"约 {agent.budget.history_budget:,} 个用于历史记录。\n\n"
            "可以尝试先输入“查找客户 Alice Johnson”，再输入“显示她的订单历史”\n"
            "输入 [bold]'quit'[/bold] 或 [bold]'exit'[/bold] 结束程序。",
            title="业务数据智能体",
        )
    )

    # 显示初始预算
    _render_budget_display(console, agent.get_token_snapshot())

    while True:
        console.print("\n[bold green]你：[/bold green] ", end="")
        user_input = input().strip()

        if user_input.lower() in ["quit", "exit", ""]:
            console.print("\n[yellow]正在结束会话……[/yellow]")
            break

        try:
            response = agent.chat(user_input)

            console.print("\n[bold blue]智能体：[/bold blue]")
            console.print(Markdown(response))

            # 每轮对话后显示预算
            console.print()
            _render_budget_display(console, agent.get_token_snapshot())

        except Exception as e:
            logger.error("聊天期间发生错误：%s", e)
            console.print(f"\n[red]错误：{e}[/red]")
            break

    # 最终报告
    console.print()
    token_tracker.report()
    console.print(
        f"\n[dim]消息数：{len(agent.messages)} │ "
        f"压缩次数：{agent.compression_count} │ "
        f"策略：{strategy}[/dim]"
    )


if __name__ == "__main__":
    main()
