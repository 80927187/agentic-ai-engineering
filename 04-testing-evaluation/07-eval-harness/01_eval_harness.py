"""
评测工具——综合项目

这条完整的评测流水线将单元测试模式、评测、追踪、红队测试和基准测试
整合为一套用于研究助手智能体的统一工具。

本综合项目融合了模块 05 的全部五项技术：
1. 使用依赖注入实现可测试的智能体设计
2. 使用基于代码的评分和复合评分评测黄金数据集
3. 将执行追踪与评测结果关联
4. 对抗性安全测试套件
5. 使用帕累托分析进行模型基准测试

支持两种模式：
- 模拟模式（默认）：使用预定义回答，不调用 API，立即得到结果
- 实时模式：通过工具调用智能体循环发起真实的模型 API 请求
"""

import json
import os
from pathlib import Path
from typing import Any

from common import AnthropicTokenTracker, interactive_menu, setup_logging
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel

from eval_harness import (
    BenchmarkRunner,
    CompositeGrader,
    EvalReport,
    EvalResult,
    EvalTask,
    EvalTrial,
    ResearchAgent,
    SafetyTester,
    SimulatedResearchAgent,
)
from eval_harness.red_team import load_adversarial_tasks
from eval_harness.reporter import EvalReporter
from eval_harness.tracer import SimpleTracer

load_dotenv(find_dotenv())

logger = setup_logging(__name__)


MODE_OPTIONS = [
    "模拟模式——使用预定义回答，不调用 API",
    "实时模式——发起真实的模型 API 请求",
]

AVAILABLE_MODELS = [
    "deepseek-v4-flash",
    "glm-5.2",
    "glm-4.7",
]


def select_mode_and_create_agent(console: Console, header: Panel) -> Any:
    """以交互方式选择模式和模型，并返回配置好的智能体。"""
    mode = interactive_menu(console, MODE_OPTIONS, title="选择运行模式", header=header)
    if mode is None:
        raise SystemExit(0)

    if mode.startswith("实时模式"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            console.print("[red bold]实时模式要求设置 ANTHROPIC_API_KEY[/red bold]")
            raise SystemExit(1)

        model = interactive_menu(console, AVAILABLE_MODELS, title="选择模型", header=header)
        if model is None:
            raise SystemExit(0)

        import anthropic

        client = anthropic.Anthropic()
        console.print(
            f"\n[green bold]正在以实时模式运行[/green bold]——向 {model} 发起真实 API 请求\n"
            "[dim]评测试验和安全测试使用实时智能体。"
            "基准测试仍为模拟运行（用于多模型比较）。[/dim]\n"
        )
        return ResearchAgent(client=client, model=model)

    console.print("\n[dim]正在以模拟模式运行——使用预定义回答，不调用 API。[/dim]\n")
    return SimulatedResearchAgent()


def load_golden_tasks(path: Path) -> list[EvalTask]:
    """从黄金数据集 JSON 文件加载评测任务。"""
    with Path.open(path, encoding="utf-8") as f:
        data = json.load(f)
    tasks = [EvalTask(**t) for t in data["tasks"]]
    logger.info("已加载 %d 项黄金任务（v%s）", len(tasks), data["version"])
    return tasks


def run_eval_trials(
    agent: ResearchAgent | SimulatedResearchAgent,
    tasks: list[EvalTask],
    tracer: SimpleTracer,
) -> list[EvalTrial]:
    """让智能体执行每项任务，并在追踪的同时收集试验结果。"""
    trials: list[EvalTrial] = []

    for task in tasks:
        question_preview = task.question[:50] + "…" if len(task.question) > 50 else task.question
        logger.info("正在评测任务 %s：%s", task.id, question_preview)

        # 追踪评测执行过程
        span = tracer.start_span(f"eval_{task.id}", "eval_trial")

        response = agent.answer(task.question, task_id=task.id)

        tracer.end_span(span)

        trial = EvalTrial(
            task_id=task.id,
            answer=response["answer"],
            tool_calls=response.get("tool_calls", []),
            trace=tracer.get_spans()[-1:],
            latency_ms=response.get("latency_ms", 0.0),
            input_tokens=response.get("input_tokens", 0),
            output_tokens=response.get("output_tokens", 0),
        )
        trials.append(trial)

    return trials


def grade_trials(
    trials: list[EvalTrial],
    tasks: list[EvalTask],
    grader: CompositeGrader,
) -> list[EvalResult]:
    """为所有试验评分并生成评测结果。"""
    task_map = {t.id: t for t in tasks}
    results: list[EvalResult] = []

    for trial in trials:
        task = task_map[trial.task_id]
        scores = grader.grade(trial, task)

        # 根据复合分数计算通过率
        composite = next((s for s in scores if s.grader_name == "composite"), None)
        pass_rate = 1.0 if (composite and composite.passed) else 0.0
        avg_score = composite.score if composite else 0.0

        result = EvalResult(
            task_id=trial.task_id,
            trials=[trial],
            grader_scores=scores,
            pass_rate=pass_rate,
            avg_score=avg_score,
        )
        results.append(result)

    return results


def main() -> None:
    """运行完整的评测流水线。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()

    header = Panel(
        "[bold cyan]评测工具——综合项目[/bold cyan]\n\n"
        "完整的评测流水线融合了：\n"
        "  1. 可测试的智能体设计（依赖注入）\n"
        "  2. 黄金数据集评测（关键词评分 + 引用评分）\n"
        "  3. 执行追踪（将 span 与结果关联）\n"
        "  4. 对抗性安全测试（红队测试套件）\n"
        "  5. 模型基准测试（帕累托分析）",
        title="教程 06",
    )

    # 以交互方式选择模式和模型
    agent = select_mode_and_create_agent(console, header)

    # 第 1 步：加载数据集
    base_dir = Path(__file__).parent
    tasks = load_golden_tasks(base_dir / "datasets" / "golden_tasks.json")
    adversarial_attacks = load_adversarial_tasks(base_dir / "datasets" / "adversarial_tasks.json")
    console.print(
        f"[bold]第 1 步：[/bold]已加载 {len(tasks)} 项任务和 {len(adversarial_attacks)} 次攻击\n"
    )

    # 第 2 步：初始化组件
    tracer = SimpleTracer()
    grader = CompositeGrader(keyword_weight=0.5, citation_weight=0.5)
    console.print("[bold]第 2 步：[/bold]已初始化追踪器和评分器\n")

    # 第 3 步：运行评测试验并进行追踪
    console.print("[bold]第 3 步：[/bold]正在运行评测试验……\n")
    trials = run_eval_trials(agent, tasks, tracer)
    logger.info("已完成 %d 次试验，收集了 %d 个 span", len(trials), tracer.get_span_count())

    # 第 4 步：使用复合评分器评分
    console.print("[bold]第 4 步：[/bold]正在为回答评分……\n")
    eval_results = grade_trials(trials, tasks, grader)

    # 第 5 步：运行安全测试
    console.print("[bold]第 5 步：[/bold]正在运行安全测试……\n")
    safety_tester = SafetyTester()
    safety_results = safety_tester.run_safety_suite(agent, adversarial_attacks)

    # 第 6 步：运行基准测试（模拟）
    console.print("[bold]第 6 步：[/bold]正在运行基准测试……\n")
    benchmark_runner = BenchmarkRunner()
    # 仅使用部分任务进行基准测试，以保持输出简洁
    benchmark_tasks = tasks[:5]
    benchmark_entries = benchmark_runner.run_benchmark(benchmark_tasks)
    pareto_configs = benchmark_runner.find_pareto_optimal(benchmark_entries)

    # 第 7 步：组装并输出报告
    total_latency = sum(t.latency_ms for t in trials)
    total_cost = sum(e.cost_usd for e in benchmark_entries)
    passed_count = sum(1 for r in eval_results if r.pass_rate >= 0.5)
    blocked_count = sum(1 for r in safety_results if r.blocked)

    report = EvalReport(
        agent_name="研究助手",
        eval_results=eval_results,
        safety_results=safety_results,
        benchmark_entries=benchmark_entries,
        overall_pass_rate=passed_count / len(eval_results) if eval_results else 0.0,
        overall_safety_score=blocked_count / len(safety_results) if safety_results else 0.0,
        total_cost_usd=total_cost,
        total_latency_ms=total_latency,
    )

    console.print("[bold]第 7 步：[/bold]正在生成报告……\n")
    reporter = EvalReporter(console)
    reporter.print_report(report)

    # 帕累托分析摘要
    console.print(
        Panel(
            f"[bold]帕累托最优配置：[/bold]{', '.join(pareto_configs)}\n\n"
            "这些配置在任何一个维度（准确率、延迟、成本）上\n"
            "都未被另一项配置完全支配。",
            title="帕累托分析",
        )
    )

    # 追踪摘要
    console.print(
        f"\n[dim]追踪：{tracer.get_span_count()} 个 span，"
        f"总持续时间 {tracer.get_total_duration_ms():.0f} 毫秒[/dim]"
    )

    token_tracker.report()


if __name__ == "__main__":
    main()
