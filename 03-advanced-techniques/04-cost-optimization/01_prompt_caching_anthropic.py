"""
提示词缓存（Anthropic）

通过一个以大型公司政策文档作为系统提示词的客户支持智能体，演示提示词缓存。
该政策超过 Anthropic 的 1024 词元缓存下限，因此重复调用可从缓存读取，节省 90% 的费用。

第一次调用：缓存未命中（cache_creation_input_tokens > 0）
后续调用：缓存命中（cache_read_input_tokens > 0）
"""

from dataclasses import dataclass, field

import anthropic
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from common import AnthropicTokenTracker, setup_logging

# 从根目录的 .env 文件加载环境变量
load_dotenv(find_dotenv())

# 配置日志记录
logger = setup_logging(__name__)

# 模型配置
MODEL = "deepseek-v4-flash"

# Anthropic 定价（美元/百万词元）——截至 2025 年
PRICING = {
    "input": 3.00,
    "output": 15.00,
    "cache_write": 3.75,  # 输入费率的 1.25 倍
    "cache_read": 0.30,  # 输入费率的 0.1 倍——主要节省来源
}

# 使用大型公司政策文档（约 1200～1500 个词元），确保超过 Sonnet
# 提示词缓存所要求的 1024 词元下限。
COMPANY_POLICY = """
# TechFlow Solutions——客户支持政策与常见问题

## 公司概况
TechFlow Solutions 是一家 B2B SaaS 公司，提供基于云的项目管理、团队协作和工作流
自动化工具。公司成立于 2019 年，为 40 个国家和地区的 15,000 多家企业客户提供服务。
产品套件包括 TechFlow Pro（项目管理）、TechFlow Connect（团队消息）和 TechFlow
Automate（工作流构建器）。

## 退货与退款政策

### 软件订阅
- 所有订阅方案均包含 14 天免费试用期，试用期间可使用全部功能。
- 月度订阅可随时取消；服务将持续至当前计费周期结束。未使用的天数不予部分退款。
- 年度订阅可在购买后 30 天内退款。超过 30 天后，剩余金额将转换为有效期 12 个月的账户余额。
- 企业合同（50 个以上席位）遵循服务协议中列明的定制条款。如需修改，请联系企业服务团队。

### 硬件与配件
- 实体产品（TechFlow Hub 设备及配件）可在送达后 30 天内以原包装退回，并获得全额退款。
- 有缺陷的硬件在保修范围内，可免费更换。
- 只有产品存在缺陷时，退货运费才由 TechFlow 承担。

## 配送与交付

### 数字产品
- 软件许可证和订阅激活信息会立即通过电子邮件发送。
- 企业部署包含一名专属实施专员，完整设置通常需要 5～10 个工作日。

### 实体产品
- 标准配送（5～7 个工作日）：订单满 50 美元免运费，否则收取 7.99 美元。
- 加急配送（2～3 个工作日）：14.99 美元。
- 次日达（1 个工作日）：24.99 美元——仅适用于美国地址。
- 国际配送（7～14 个工作日）：根据地区收取 19.99～39.99 美元。
- 所有货件均提供物流跟踪。订单金额超过 200 美元时需要签收。

## 保修条款
- TechFlow Hub 设备：制造商提供 2 年保修，涵盖材料和工艺缺陷。物理损坏、进水或未经授权的
  改装不在保修范围内。
- 软件：Pro 和 Enterprise 方案保证 99.9% 的正常运行时间 SLA。Basic 方案不包含 SLA。
  每低于 SLA 阈值一小时，停机补偿按该小时费用的 10 倍计算。

## 账户管理

### 方案等级
- **Basic**（12 美元/用户/月）：核心项目管理、5GB 存储空间、电子邮件支持。
- **Pro**（29 美元/用户/月）：高级分析、50GB 存储空间、优先支持、API 访问和自定义集成。
- **Enterprise**（49 美元/用户/月）：无限存储空间、专属客户经理、SSO/SAML、审计日志、
  自定义 SLA 和电话支持。

### 升级与降级
- 升级立即生效，并立即按比例收取差额。
- 降级在下一个计费周期生效。在此之前，较高等级的专属功能仍可使用。
- 降级处理前，必须导出或删除超出较低等级存储上限的数据。系统会提前 7 天自动发出警告。

### 账单
- 接受的付款方式：Visa、Mastercard、Amex 和电汇（仅限 Enterprise）。
- 年度方案的发票在每月 1 日生成；月度方案的发票在订阅周年日生成。
- 付款失败后，系统会在 9 天内重试 3 次。第三次失败后，账户将被暂停。暂停后数据保留 30 天。

## 常见问题

1. **如何重置密码？**
   前往“设置 > 安全 > 更改密码”，或使用登录页面上的“忘记密码”链接。重置链接将发送到注册邮箱。

2. **可以将许可证转让给其他用户吗？**
   可以。管理员可在团队管理控制台中免费重新分配席位。重新分配后，原用户会立即失去访问权限。

3. **支持哪些集成？**
   Pro 和 Enterprise 方案支持 Slack、Jira、GitHub、GitLab、Salesforce、HubSpot、Zapier，
   还可通过我们的 API 和 Zapier 连接器与其他 200 多种工具集成。

4. **我的数据是否经过加密？**
   是。所有静态数据均采用 AES-256 加密，传输中的数据采用 TLS 1.3 加密。Enterprise 方案支持
   客户管理的加密密钥（BYOK）。

5. **取消订阅后，我的数据会怎样？**
   取消后数据保留 30 天。在此期间，您可以随时通过“设置 > 数据导出”导出全部数据。30 天后，
   数据将按照我们的数据保留政策永久删除。

6. **是否提供教育机构或非营利组织折扣？**
   是。通过验证的教育机构和注册非营利组织可享受所有方案六折优惠。请在网站上提交有效文件申请。

7. **如何联系支持团队？**
   - Basic：电子邮件支持（24～48 小时内响应）
   - Pro：优先电子邮件支持（4～8 小时内响应）+ 在线聊天
   - Enterprise：专属客户经理 + 电话支持（1 小时响应 SLA）

8. **购买前可以查看演示吗？**
   可以。请访问 techflow.com/demo 预约个性化演示，也可以立即开始 14 天免费试用，无需信用卡。

9. **正常运行时间如何保证？**
   Pro 和 Enterprise 方案包含 99.9% 的正常运行时间 SLA。可在 status.techflow.com 查看实时状态。

10. **批量许可证如何计费？**
    购买 50 个以上席位可采用带批量折扣的 Enterprise 定价。请联系 sales@techflow.com 获取报价。

## 升级处理流程
- **第 1 级**（一线客服）：处理一般咨询、密码重置、账单问题和标准故障排除。
- **第 2 级**（高级客服）：处理超过 500 美元的退款请求、账户暂停、数据恢复和复杂技术问题。
- **第 3 级**（工程团队）：处理服务中断、安全事件、API 缺陷和基础设施问题。
- 升级前始终先尝试在当前级别解决问题。转交到下一级别前，记录所有已采取的步骤和客户沟通内容。
""".strip()

SYSTEM_INSTRUCTIONS = (
    "你是 TechFlow Solutions 的客户支持智能体。请根据下方的公司政策准确回答客户问题。"
    "回答应友好、专业且简洁。如果问题超出政策范围，请明确说明，并建议客户联系相应团队。"
    "适用时，请始终引用相关政策章节。"
)


@dataclass
class CacheMetrics:
    """跟踪多次 API 调用的缓存性能。"""

    call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_cache_read_tokens: int = 0
    per_call_history: list[dict] = field(default_factory=list)

    def record_call(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_write_tokens: int,
        cache_read_tokens: int,
    ) -> None:
        """记录单次 API 调用的指标。"""
        self.call_count += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cache_write_tokens += cache_write_tokens
        self.total_cache_read_tokens += cache_read_tokens
        self.per_call_history.append(
            {
                "call": self.call_count,
                "input": input_tokens,
                "output": output_tokens,
                "cache_write": cache_write_tokens,
                "cache_read": cache_read_tokens,
            }
        )

    def cost_with_caching(self) -> float:
        """使用缓存费率计算实际成本。"""
        uncached_input = (
            self.total_input_tokens - self.total_cache_write_tokens - (self.total_cache_read_tokens)
        )
        return (
            uncached_input * PRICING["input"]
            + self.total_cache_write_tokens * PRICING["cache_write"]
            + self.total_cache_read_tokens * PRICING["cache_read"]
            + self.total_output_tokens * PRICING["output"]
        ) / 1_000_000

    def cost_without_caching(self) -> float:
        """计算所有词元均按基础输入费率收费时的假设成本。"""
        return (
            self.total_input_tokens * PRICING["input"]
            + self.total_output_tokens * PRICING["output"]
        ) / 1_000_000

    def savings(self) -> float:
        """计算缓存节省的美元金额。"""
        return self.cost_without_caching() - self.cost_with_caching()

    def cache_hit_rate(self) -> float:
        """计算从缓存提供的可缓存词元百分比。"""
        total_cache = self.total_cache_write_tokens + self.total_cache_read_tokens
        if total_cache == 0:
            return 0.0
        return (self.total_cache_read_tokens / total_cache) * 100


class CachedSupportAgent:
    """演示提示词缓存的客户支持智能体。"""

    def __init__(
        self,
        model: str,
        token_tracker: AnthropicTokenTracker,
        use_cache: bool = True,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.token_tracker = token_tracker
        self.use_cache = use_cache
        self.messages: list[dict] = []
        self.metrics = CacheMetrics()

    def _build_system(self) -> str | list[dict]:
        """构建系统提示词——使用 cache_control 块或纯字符串。"""
        if not self.use_cache:
            return f"{SYSTEM_INSTRUCTIONS}\n\n{COMPANY_POLICY}"

        # 明确设置缓存断点：标记要缓存的大型政策块。
        # 指令很短且很少变化，但政策占提示词的大部分——缓存后可在重复调用时节省约 90%。
        return [
            {"type": "text", "text": SYSTEM_INSTRUCTIONS},
            {
                "type": "text",
                "text": COMPANY_POLICY,
                "cache_control": {"type": "ephemeral"},  # 缓存 5 分钟
            },
        ]
        # 另一种方案：调用 client.messages.create(...) 时不显式指定 cache_control，
        # 让 Anthropic 自动缓存超过 1024 个词元的前缀。
        # 明确的断点可以精确控制缓存哪些内容。

    def chat(self, user_input: str) -> tuple[str, dict]:
        """发送消息、跟踪缓存指标并返回（响应, 用量字典）。"""
        self.messages.append({"role": "user", "content": user_input})

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=21333,
                system=self._build_system(),
                messages=self.messages,
            )
        except Exception:
            self.messages.pop()
            raise

        self.token_tracker.track(response.usage)

        # 从用量信息中提取缓存指标
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cache_write = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0

        self.metrics.record_call(input_tokens, output_tokens, cache_write, cache_read)

        usage_dict = {
            "input": input_tokens,
            "output": output_tokens,
            "cache_write": cache_write,
            "cache_read": cache_read,
        }

        logger.info(
            "调用 %d——输入：%d，输出：%d，缓存写入：%d，缓存读取：%d",
            self.metrics.call_count,
            input_tokens,
            output_tokens,
            cache_write,
            cache_read,
        )

        text_parts = [block.text for block in response.content if block.type == "text"]
        if not text_parts:
            block_types = [block.type for block in response.content]
            raise ValueError(
                f"模型响应中没有文本内容（stop_reason={response.stop_reason}，"
                f"内容块={block_types}，output_tokens={response.usage.output_tokens}）。"
            )
        assistant_message = "\n\n".join(text_parts)
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message, usage_dict


def _render_call_metrics(console: Console, call_num: int, usage: dict) -> None:
    """呈现单次调用的缓存指标。"""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("指标", style="dim")
    table.add_column("值", justify="right")

    table.add_row("输入词元", f"[cyan]{usage['input']:,}[/cyan]")
    table.add_row("输出词元", f"[cyan]{usage['output']:,}[/cyan]")

    # 突出显示缓存行为
    if usage["cache_write"] > 0:
        table.add_row(
            "缓存写入",
            f"[yellow]{usage['cache_write']:,}[/yellow] [dim]（1.25 倍——首次调用填充缓存）[/dim]",
        )
    if usage["cache_read"] > 0:
        table.add_row(
            "缓存读取",
            f"[green]{usage['cache_read']:,}[/green] [dim]（0.1 倍——节省 90%！）[/dim]",
        )
    if usage["cache_write"] == 0 and usage["cache_read"] == 0:
        table.add_row("缓存", "[dim]没有可缓存的内容[/dim]")

    console.print(Panel(table, title=f"调用 {call_num}", border_style="dim", padding=(0, 1)))


def _render_savings_summary(console: Console, metrics: CacheMetrics) -> None:
    """呈现累计成本对比。"""
    cost_cached = metrics.cost_with_caching()
    cost_baseline = metrics.cost_without_caching()
    savings = metrics.savings()
    savings_pct = (savings / cost_baseline * 100) if cost_baseline > 0 else 0

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("指标", style="dim", min_width=20)
    table.add_column("值", justify="right")

    table.add_row("未使用缓存的成本", f"[red]${cost_baseline:.6f}[/red]")
    table.add_row("使用缓存的成本", f"[green]${cost_cached:.6f}[/green]")
    table.add_row("节省", f"[bold green]${savings:.6f} ({savings_pct:.1f}%)[/bold green]")
    table.add_row("缓存命中率", f"[cyan]{metrics.cache_hit_rate():.1f}%[/cyan]")
    table.add_row("调用总数", f"[cyan]{metrics.call_count}[/cyan]")

    console.print(
        Panel(
            table,
            title="累计节省",
            border_style="green" if savings > 0 else "dim",
            padding=(0, 1),
        )
    )


def main() -> None:
    """提示词缓存演示的主编排函数。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()
    agent = CachedSupportAgent(MODEL, token_tracker)

    console.print(
        Panel(
            "[bold cyan]提示词缓存演示[/bold cyan]\n\n"
            "此客户支持智能体使用一份大型公司政策文档（约 1500 个词元）作为系统提示词，\n"
            "并通过 Anthropic 的提示词缓存对其进行缓存。\n\n"
            "[bold]工作原理：[/bold]\n"
            "  1. 首次调用：缓存未命中——政策被写入缓存（1.25 倍成本）\n"
            "  2. 后续调用：缓存命中——从缓存读取政策（0.1 倍成本）\n"
            "  3. 缓存 TTL 为 5 分钟——每次命中都会刷新\n\n"
            "请就 TechFlow Solutions 提出支持问题，并观察节省金额的增长。\n"
            "输入 [bold]'quit'[/bold] 或 [bold]'exit'[/bold] 结束。\n\n"
            "[bold]可以尝试以下示例问题：[/bold]\n"
            "  1. TechFlow Solutions 是什么公司？\n"
            "  2. 年度订阅的退款政策是什么？\n"
            "  3. 配送需要多长时间？\n"
            "  4. 提供哪些方案等级，价格分别是多少？\n"
            "  5. 是否为非营利组织提供折扣？",
            title="TechFlow 客户支持",
        )
    )

    while True:
        console.print("\n[bold green]你：[/bold green] ", end="")
        user_input = input().strip()

        if user_input.lower() in ["quit", "exit", ""]:
            console.print("\n[yellow]正在结束会话……[/yellow]")
            break

        try:
            response, usage = agent.chat(user_input)

            console.print("\n[bold blue]支持智能体：[/bold blue]")
            console.print(Markdown(response))

            # 单次调用的缓存明细
            console.print()
            _render_call_metrics(console, agent.metrics.call_count, usage)

            # 累计节省（至少调用 2 次后才有意义）
            if agent.metrics.call_count >= 2:
                _render_savings_summary(console, agent.metrics)

        except Exception as e:
            logger.error("聊天期间发生错误：%s", e)
            console.print(f"\n[red]错误：{e}[/red]")
            break

    # 最终报告
    console.print()
    token_tracker.report()

    if agent.metrics.call_count >= 2:
        console.print()
        _render_savings_summary(console, agent.metrics)


if __name__ == "__main__":
    main()
