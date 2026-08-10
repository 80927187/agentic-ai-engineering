"""
模型路由（Anthropic）

演示如何通过智能模型路由优化成本。低成本分类器（Haiku）评估任务难度，并将简单任务
路由到 Haiku，将困难任务路由到 Sonnet。展示与全部使用 Sonnet 的基准方案相比实际节省的成本。
"""

from dataclasses import dataclass, field

import anthropic
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from common import AnthropicTokenTracker, setup_logging
from common.menu import interactive_menu

# 从根目录的 .env 文件加载环境变量
load_dotenv(find_dotenv())

# 配置日志记录
logger = setup_logging(__name__)

# 模型配置
MODEL_CLASSIFIER = "deepseek-v4-flash"
MODEL_EASY = "deepseek-v4-flash"
MODEL_HARD = "deepseek-v4-flash"

# 定价（美元/百万词元）
PRICING = {
    "haiku_input": 1.00,
    "haiku_output": 5.00,
    "sonnet_input": 3.00,
    "sonnet_output": 15.00,
}

# 混合简单和困难任务的示例
SAMPLE_TASKS = [
    "法国的首都是哪里？",
    "将 72 华氏度换算为摄氏度。",
    "为一款实时多人游戏设计微服务架构。该游戏需要支持 10 万名并发用户，延迟低于 50 毫秒。",
    "第一代 iPhone 是在哪一年发布的？",
    "分析事件溯源与传统 CRUD 在金融交易系统中的权衡；该系统需要完整的审计追踪并满足监管要求。",
    "一千米等于多少米？",
    "对于全球分布式电子商务平台，比较选择 PostgreSQL、Cassandra 和 CockroachDB 时，"
    "CAP 定理所带来的不同影响。",
    "金的化学符号是什么？",
]


@dataclass
class TaskResult:
    """路由任务的执行结果。"""

    task: str
    difficulty: str
    model_used: str
    response: str
    routed_cost: float
    baseline_cost: float  # 使用 Sonnet 时的成本


class ModelRouter:
    """根据复杂度将任务路由到合适的模型。"""

    def __init__(self, token_tracker: AnthropicTokenTracker):
        self.client = anthropic.Anthropic()
        self.token_tracker = token_tracker
        self.results: list[TaskResult] = field(default_factory=list)
        self.results = []

    @staticmethod
    def _extract_text(response: anthropic.types.Message) -> str:
        """跳过思考块并合并响应中的所有文本块。"""
        text_parts = [block.text for block in response.content if block.type == "text"]
        if not text_parts:
            block_types = [block.type for block in response.content]
            raise ValueError(
                f"模型响应中没有文本内容（stop_reason={response.stop_reason}，"
                f"内容块={block_types}，output_tokens={response.usage.output_tokens}）。"
            )
        return "\n\n".join(text_parts)

    def classify(self, task: str) -> str:
        """使用 Haiku 将任务难度分类为 'easy' 或 'hard'。"""
        response = self.client.messages.create(
            model=MODEL_CLASSIFIER,
            max_tokens=21333,
            system=(
                "将以下任务分类为 'easy' 或 'hard'。\n"
                "easy：简单的事实查询、单位换算、基础数学和定义。\n"
                "hard：分析、架构设计、多步推理、比较、创意写作和代码审查。\n"
                "只用一个单词回答：easy 或 hard。"
            ),
            messages=[{"role": "user", "content": task}],
        )
        self.token_tracker.track(response.usage)

        classification = self._extract_text(response).strip().lower()
        # 分类结果不明确时，默认按困难任务处理
        if classification not in ("easy", "hard"):
            logger.warning("分类结果 '%s' 不明确，默认按 hard 处理", classification)
            classification = "hard"

        logger.info("分类为 '%s'：%s", classification, task[:60])
        return classification

    def execute(self, task: str, model: str) -> tuple[str, int, int]:
        """在指定模型上运行任务，返回（响应, 输入词元数, 输出词元数）。"""
        response = self.client.messages.create(
            model=model,
            max_tokens=21333,
            messages=[{"role": "user", "content": task}],
        )
        self.token_tracker.track(response.usage)

        return (
            self._extract_text(response),
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """根据给定模型和词元数计算成本。"""
        if model == MODEL_HARD:
            return (
                input_tokens * PRICING["sonnet_input"] + output_tokens * PRICING["sonnet_output"]
            ) / 1_000_000
        return (
            input_tokens * PRICING["haiku_input"] + output_tokens * PRICING["haiku_output"]
        ) / 1_000_000

    def route_and_execute(self, task: str) -> TaskResult:
        """对任务进行分类、路由和执行，并跟踪成本。"""
        difficulty = self.classify(task)

        model = MODEL_EASY if difficulty == "easy" else MODEL_HARD
        response_text, input_tokens, output_tokens = self.execute(task, model)

        routed_cost = self._calculate_cost(model, input_tokens, output_tokens)
        baseline_cost = self._calculate_cost(MODEL_HARD, input_tokens, output_tokens)

        result = TaskResult(
            task=task,
            difficulty=difficulty,
            model_used=model,
            response=response_text,
            routed_cost=routed_cost,
            baseline_cost=baseline_cost,
        )
        self.results.append(result)

        return result

    def get_summary(self) -> dict:
        """汇总所有结果的成本对比。"""
        total_routed = sum(r.routed_cost for r in self.results)
        total_baseline = sum(r.baseline_cost for r in self.results)
        savings = total_baseline - total_routed
        savings_pct = (savings / total_baseline * 100) if total_baseline > 0 else 0
        easy_count = sum(1 for r in self.results if r.difficulty == "easy")
        hard_count = sum(1 for r in self.results if r.difficulty == "hard")

        return {
            "total_tasks": len(self.results),
            "easy_count": easy_count,
            "hard_count": hard_count,
            "total_routed_cost": total_routed,
            "total_baseline_cost": total_baseline,
            "savings": savings,
            "savings_pct": savings_pct,
        }


def _render_task_result(console: Console, result: TaskResult, index: int) -> None:
    """呈现单个任务结果及其路由信息。"""
    model_label = "Haiku" if result.model_used == MODEL_EASY else "Sonnet"
    diff_color = "green" if result.difficulty == "easy" else "yellow"
    savings = result.baseline_cost - result.routed_cost
    savings_pct = (savings / result.baseline_cost * 100) if result.baseline_cost > 0 else 0

    console.print(
        Panel(
            f"[dim]任务：[/dim] {result.task}\n"
            f"[dim]难度：[/dim] [{diff_color}]{result.difficulty}[/{diff_color}] → "
            f"[bold]{model_label}[/bold]\n"
            f"[dim]路由成本：[/dim] [green]${result.routed_cost:.6f}[/green]  "
            f"[dim]基准（Sonnet）：[/dim] [red]${result.baseline_cost:.6f}[/red]  "
            f"[dim]节省：[/dim] [bold green]${savings:.6f} ({savings_pct:.0f}%)[/bold green]",
            title=f"任务 {index}",
            border_style="dim",
            padding=(0, 1),
        )
    )


def _render_summary(console: Console, summary: dict) -> None:
    """呈现汇总成本信息。"""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("指标", style="dim", min_width=22)
    table.add_column("值", justify="right")

    table.add_row("已处理任务", f"[cyan]{summary['total_tasks']}[/cyan]")
    table.add_row(
        "路由明细",
        f"[green]{summary['easy_count']} 个简单任务[/green] / "
        f"[yellow]{summary['hard_count']} 个困难任务[/yellow]",
    )
    table.add_row(
        "成本（路由后）",
        f"[green]${summary['total_routed_cost']:.6f}[/green]",
    )
    table.add_row(
        "成本（全部使用 Sonnet 的基准）",
        f"[red]${summary['total_baseline_cost']:.6f}[/red]",
    )
    table.add_row(
        "节省总额",
        f"[bold green]${summary['savings']:.6f} ({summary['savings_pct']:.1f}%)[/bold green]",
    )

    console.print(
        Panel(
            table,
            title="成本汇总——路由方案与全部使用 Sonnet 的方案",
            border_style="green",
            padding=(0, 1),
        )
    )


def _run_demo(console: Console, router: ModelRouter) -> None:
    """运行所有示例任务并显示结果。"""
    console.print(f"\n[bold]正在运行 {len(SAMPLE_TASKS)} 个示例任务……[/bold]\n")

    for i, task in enumerate(SAMPLE_TASKS, 1):
        console.print(f"[dim]正在处理任务 {i}/{len(SAMPLE_TASKS)}……[/dim]")
        try:
            result = router.route_and_execute(task)
            _render_task_result(console, result, i)
            # 显示截断后的响应
            preview = (
                result.response[:200] + "..." if len(result.response) > 200 else result.response
            )
            console.print(Markdown(preview))
            console.print()
        except Exception as e:
            logger.error("处理任务 %d 时发生错误：%s", i, e)
            console.print(f"[red]错误：{e}[/red]\n")

    _render_summary(console, router.get_summary())


def _run_interactive(console: Console, router: ModelRouter) -> None:
    """交互模式——用户输入任务并实时查看分类结果。"""
    console.print(
        "\n[bold]交互模式[/bold]——输入任务以查看路由决策。\n"
        "输入 [bold]'summary'[/bold] 查看成本汇总，输入 [bold]'quit'[/bold] 退出。\n"
    )

    while True:
        console.print("[bold green]任务：[/bold green] ", end="")
        user_input = input().strip()

        if user_input.lower() in ["quit", "exit", ""]:
            break

        if user_input.lower() == "summary":
            if router.results:
                _render_summary(console, router.get_summary())
            else:
                console.print("[dim]尚未处理任何任务。[/dim]")
            continue

        try:
            result = router.route_and_execute(user_input)
            _render_task_result(console, result, len(router.results))

            console.print("\n[bold blue]响应：[/bold blue]")
            console.print(Markdown(result.response))
            console.print()

        except Exception as e:
            logger.error("处理任务时发生错误：%s", e)
            console.print(f"\n[red]错误：{e}[/red]")

    if router.results:
        _render_summary(console, router.get_summary())


def main() -> None:
    """模型路由演示的主编排函数。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()
    router = ModelRouter(token_tracker)

    header = Panel(
        "[bold cyan]模型路由演示[/bold cyan]\n\n"
        "低成本分类器（Haiku）会评估每项任务的难度，\n"
        "然后将简单任务路由到 [green]Haiku[/green]，困难任务路由到 [yellow]Sonnet[/yellow]。\n\n"
        "Haiku 的输入成本比 Sonnet 低约 73%——将简单任务路由到更便宜的模型，\n"
        "在大规模使用时可以真正节省费用。\n\n"
        "[bold]定价：[/bold]\n"
        f"  Haiku： ${PRICING['haiku_input']:.2f} 输入 / ${PRICING['haiku_output']:.2f} 输出（每百万词元）\n"
        f"  Sonnet：${PRICING['sonnet_input']:.2f} 输入 / ${PRICING['sonnet_output']:.2f} 输出（每百万词元）",
        title="智能模型路由",
    )

    mode = interactive_menu(
        console,
        items=[
            "演示——使用自动路由运行示例任务",
            "交互——输入你自己的任务",
        ],
        title="选择模式",
        header=header,
    )

    if mode is None:
        return

    if mode.startswith("演示"):
        _run_demo(console, router)
    else:
        _run_interactive(console, router)

    # 最终词元报告
    console.print()
    token_tracker.report()


if __name__ == "__main__":
    main()
