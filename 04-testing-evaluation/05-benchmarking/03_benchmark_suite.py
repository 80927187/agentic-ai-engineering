"""
完整基准测试套件

将模型对比和提示词对比组合成配置矩阵。
运行所有“模型 × 提示词”组合、计算聚合统计数据、执行帕累托分析
（识别非支配配置）并生成汇总报告。使用模拟结果时可独立运行。
"""

import json
import os
import time
from typing import Any

import anthropic
import openai
from common import AnthropicTokenTracker, OpenAITokenTracker, setup_logging
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared.knowledge_base import (
    BENCHMARK_TASKS,
    TOOLS_ANTHROPIC,
    TOOLS_OPENAI,
    score_answer,
    search_knowledge_base,
)
from shared.models import MODEL_CONFIGS, BenchmarkConfig, BenchmarkResult, ModelConfig

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# 提示词策略
# ---------------------------------------------------------------------------

PROMPT_STRATEGIES = {
    "zero_shot": (
        "你是一名研究助手。只能使用工具所提供的搜索结果来回答问题，并标注信息来源。"
    ),
    "few_shot": (
        "你是一名研究助手。只能使用工具所提供的搜索结果来回答问题，并标注信息来源。\n\n"
        "示例：\n"
        "问题：什么是 REST？\n"
        "回答：REST（表述性状态转移）是一种 API 架构风格（doc_002）。"
        "它采用面向资源的设计，并使用 HTTP 方法表示操作。\n\n"
        "现在请用相同格式回答用户的问题。"
    ),
    "chain_of_thought": (
        "你是一名研究助手。只能使用工具所提供的搜索结果来回答问题，并标注信息来源。\n\n"
        "请逐步思考：\n"
        "1. 搜索相关文档\n"
        "2. 从每份文档中提取关键事实\n"
        "3. 整合出全面的回答\n"
        "4. 标注使用的所有信息来源"
    ),
}

PROMPT_STRATEGY_LABELS = {
    "zero_shot": "零样本",
    "few_shot": "少样本",
    "chain_of_thought": "思维链",
}

# ---------------------------------------------------------------------------
# 演示模式的模拟结果（3 个模型 × 3 种提示词 × 5 个任务 = 45 条结果）
# ---------------------------------------------------------------------------


def _build_simulated_results() -> dict[str, list[BenchmarkResult]]:
    """为完整配置矩阵构建模拟结果。"""
    # (配置名称, 任务 ID) -> (得分, 延迟, 输入 Token, 输出 Token, 成本)
    sim_data: dict[str, list[tuple[str, float, float, int, int, float]]] = {
        "Claude Sonnet + 零样本": [
            ("bench_001", 0.8, 1100, 150, 180, 0.0031),
            ("bench_002", 0.8, 1080, 145, 175, 0.0031),
            ("bench_003", 0.7, 1150, 155, 185, 0.0032),
            ("bench_004", 0.6, 1090, 148, 178, 0.0031),
            ("bench_005", 0.7, 1120, 146, 176, 0.0031),
        ],
        "Claude Sonnet + 少样本": [
            ("bench_001", 0.9, 1250, 190, 200, 0.0036),
            ("bench_002", 1.0, 1230, 185, 195, 0.0035),
            ("bench_003", 0.9, 1300, 195, 210, 0.0037),
            ("bench_004", 0.8, 1240, 188, 198, 0.0036),
            ("bench_005", 0.9, 1260, 186, 196, 0.0035),
        ],
        "Claude Sonnet + 思维链": [
            ("bench_001", 0.9, 1500, 200, 260, 0.0045),
            ("bench_002", 1.0, 1480, 195, 255, 0.0044),
            ("bench_003", 0.9, 1550, 205, 270, 0.0047),
            ("bench_004", 1.0, 1490, 198, 258, 0.0045),
            ("bench_005", 0.9, 1520, 196, 256, 0.0045),
        ],
        "Claude Haiku + 零样本": [
            ("bench_001", 0.6, 420, 125, 130, 0.0006),
            ("bench_002", 0.6, 410, 120, 125, 0.0006),
            ("bench_003", 0.5, 440, 128, 135, 0.0006),
            ("bench_004", 0.4, 415, 122, 128, 0.0006),
            ("bench_005", 0.5, 430, 124, 130, 0.0006),
        ],
        "Claude Haiku + 少样本": [
            ("bench_001", 0.7, 480, 165, 155, 0.0008),
            ("bench_002", 0.8, 470, 160, 150, 0.0007),
            ("bench_003", 0.7, 500, 168, 160, 0.0008),
            ("bench_004", 0.6, 475, 162, 152, 0.0007),
            ("bench_005", 0.7, 490, 164, 154, 0.0008),
        ],
        "Claude Haiku + 思维链": [
            ("bench_001", 0.8, 560, 175, 200, 0.0009),
            ("bench_002", 0.8, 550, 170, 195, 0.0009),
            ("bench_003", 0.7, 580, 178, 210, 0.0010),
            ("bench_004", 0.7, 555, 172, 198, 0.0009),
            ("bench_005", 0.7, 570, 174, 202, 0.0009),
        ],
        "GPT-4.1 mini + 零样本": [
            ("bench_001", 0.7, 780, 130, 160, 0.0003),
            ("bench_002", 0.7, 760, 125, 155, 0.0003),
            ("bench_003", 0.6, 800, 135, 165, 0.0003),
            ("bench_004", 0.5, 770, 128, 158, 0.0003),
            ("bench_005", 0.6, 790, 126, 156, 0.0003),
        ],
        "GPT-4.1 mini + 少样本": [
            ("bench_001", 0.8, 850, 170, 180, 0.0004),
            ("bench_002", 0.9, 840, 165, 175, 0.0003),
            ("bench_003", 0.8, 880, 175, 185, 0.0004),
            ("bench_004", 0.7, 845, 168, 178, 0.0004),
            ("bench_005", 0.8, 860, 166, 176, 0.0004),
        ],
        "GPT-4.1 mini + 思维链": [
            ("bench_001", 0.8, 1000, 180, 230, 0.0004),
            ("bench_002", 0.9, 980, 175, 225, 0.0004),
            ("bench_003", 0.8, 1020, 185, 240, 0.0005),
            ("bench_004", 0.8, 990, 178, 228, 0.0004),
            ("bench_005", 0.8, 1010, 176, 232, 0.0004),
        ],
    }

    results: dict[str, list[BenchmarkResult]] = {}
    for config_name, tasks in sim_data.items():
        results[config_name] = [
            BenchmarkResult(
                task_id=tid,
                config_name=config_name,
                answer=f"使用 {config_name} 为 {tid} 生成的模拟回答。",
                keyword_score=score,
                latency_ms=lat,
                input_tokens=inp,
                output_tokens=out,
                cost_usd=cost,
                tool_calls=1,
            )
            for tid, score, lat, inp, out, cost in tasks
        ]
    return results


SIMULATED_SUITE_RESULTS = _build_simulated_results()

# ---------------------------------------------------------------------------
# 基准测试套件类
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """包含配置矩阵和帕累托分析的完整基准测试套件。"""

    def __init__(self) -> None:
        self.anthropic_tracker = AnthropicTokenTracker()
        self.openai_tracker = OpenAITokenTracker()

    def build_config_matrix(
        self, models: list[ModelConfig], prompts: dict[str, str]
    ) -> list[BenchmarkConfig]:
        """构建所有“模型 × 提示词”组合。"""
        configs: list[BenchmarkConfig] = []
        for model in models:
            for prompt_name, system_prompt in prompts.items():
                prompt_label = PROMPT_STRATEGY_LABELS.get(prompt_name, prompt_name)
                name = f"{model.name} + {prompt_label}"
                configs.append(BenchmarkConfig(name, model, prompt_name, system_prompt))
        logger.info(
            "已构建 %d 个配置（%d 个模型 × %d 种提示词）",
            len(configs),
            len(models),
            len(prompts),
        )
        return configs

    def _run_anthropic(self, task: dict, config: BenchmarkConfig) -> BenchmarkResult:
        """使用 Anthropic API 运行任务。"""
        client = anthropic.Anthropic()
        messages: list[dict[str, Any]] = [{"role": "user", "content": task["question"]}]
        tool_call_count = 0

        start = time.perf_counter()
        while True:
            response = client.messages.create(
                model=config.model.model_id,
                max_tokens=1024,
                system=config.system_prompt,
                tools=TOOLS_ANTHROPIC,
                messages=messages,
            )
            self.anthropic_tracker.track(response.usage)

            if response.stop_reason != "tool_use":
                answer = "".join(b.text for b in response.content if hasattr(b, "text"))
                break

            messages.append({"role": "assistant", "content": response.content})
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_call_count += 1
                    result = search_knowledge_base(**block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
            messages.append({"role": "user", "content": tool_results})

        latency_ms = (time.perf_counter() - start) * 1000
        inp = response.usage.input_tokens
        out = response.usage.output_tokens
        cost = (
            inp * config.model.cost_per_input_token + out * config.model.cost_per_output_token
        ) / 1_000_000

        return BenchmarkResult(
            task_id=task["id"],
            config_name=config.name,
            answer=answer,
            keyword_score=score_answer(answer, task["expected_keywords"]),
            latency_ms=latency_ms,
            input_tokens=inp,
            output_tokens=out,
            cost_usd=cost,
            tool_calls=tool_call_count,
        )

    def _run_openai(self, task: dict, config: BenchmarkConfig) -> BenchmarkResult:
        """使用 OpenAI API 运行任务。"""
        client = openai.OpenAI()
        messages: list[dict[str, Any]] = [{"role": "user", "content": task["question"]}]
        tool_call_count = 0

        start = time.perf_counter()
        while True:
            response = client.responses.create(
                model=config.model.model_id,
                instructions=config.system_prompt,
                max_output_tokens=1024,
                tools=TOOLS_OPENAI,
                input=messages,
            )
            self.openai_tracker.track(response.usage)

            function_calls = [o for o in response.output if o.type == "function_call"]
            if not function_calls:
                answer = response.output_text or ""
                break

            messages.extend(response.output)
            for func_call in function_calls:
                tool_call_count += 1
                args = json.loads(func_call.arguments)
                result = search_knowledge_base(**args)
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": func_call.call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )

        latency_ms = (time.perf_counter() - start) * 1000
        inp = response.usage.input_tokens
        out = response.usage.output_tokens
        cost = (
            inp * config.model.cost_per_input_token + out * config.model.cost_per_output_token
        ) / 1_000_000

        return BenchmarkResult(
            task_id=task["id"],
            config_name=config.name,
            answer=answer,
            keyword_score=score_answer(answer, task["expected_keywords"]),
            latency_ms=latency_ms,
            input_tokens=inp,
            output_tokens=out,
            cost_usd=cost,
            tool_calls=tool_call_count,
        )

    def run_suite(
        self,
        configs: list[BenchmarkConfig],
        tasks: list[dict],
        num_trials: int = 1,
    ) -> dict[str, list[BenchmarkResult]]:
        """使用所有配置和任务运行完整基准测试套件。"""
        all_results: dict[str, list[BenchmarkResult]] = {}

        for config in configs:
            logger.info("配置：%s", config.name)
            config_results: list[BenchmarkResult] = []

            for trial in range(num_trials):
                for task in tasks:
                    logger.info("  第 %d 次试验，任务 %s", trial + 1, task["id"])
                    try:
                        if config.model.provider == "anthropic":
                            result = self._run_anthropic(task, config)
                        elif config.model.provider == "openai":
                            result = self._run_openai(task, config)
                        else:
                            logger.error("未知服务商：%s", config.model.provider)
                            continue
                        config_results.append(result)
                    except Exception as e:
                        logger.error("    错误：%s", e)

            all_results[config.name] = config_results

        return all_results

    def compute_summary(self, results: dict[str, list[BenchmarkResult]]) -> list[dict[str, Any]]:
        """计算每个配置的聚合统计数据。"""
        summaries: list[dict[str, Any]] = []
        for config_name, res_list in results.items():
            n = len(res_list)
            if n == 0:
                continue
            summaries.append(
                {
                    "config": config_name,
                    "accuracy": sum(r.keyword_score for r in res_list) / n,
                    "avg_latency_ms": sum(r.latency_ms for r in res_list) / n,
                    "avg_tokens": sum(r.input_tokens + r.output_tokens for r in res_list) / n,
                    "avg_cost": sum(r.cost_usd for r in res_list) / n,
                    "total_cost": sum(r.cost_usd for r in res_list),
                    "tasks": n,
                }
            )
        return summaries

    def find_pareto_optimal(self, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """找出帕累托最优配置（在准确率和成本上不受其他配置支配）。"""
        pareto: list[dict[str, Any]] = []

        for candidate in summaries:
            dominated = False
            for other in summaries:
                if other["config"] == candidate["config"]:
                    continue
                # 如果 other 在所有维度上都至少与 candidate 一样好，且至少一个维度严格更好，
                # 则 other 支配 candidate
                better_or_equal_acc = other["accuracy"] >= candidate["accuracy"]
                better_or_equal_cost = other["avg_cost"] <= candidate["avg_cost"]
                strictly_better = (
                    other["accuracy"] > candidate["accuracy"]
                    or other["avg_cost"] < candidate["avg_cost"]
                )
                if better_or_equal_acc and better_or_equal_cost and strictly_better:
                    dominated = True
                    break

            if not dominated:
                pareto.append(candidate)

        logger.info(
            "帕累托最优：%d/%d 个配置",
            len(pareto),
            len(summaries),
        )
        return pareto

    def generate_report(self, summaries: list[dict[str, Any]], pareto: list[dict[str, Any]]) -> str:
        """生成基准测试结果的文本汇总报告。"""
        lines: list[str] = ["基准测试报告", "=" * 60, ""]

        # 整体统计数据
        lines.append(f"已测试配置数：{len(summaries)}")
        lines.append(f"帕累托最优配置数：{len(pareto)}")
        lines.append("")

        # 各维度最佳配置
        best_acc = max(summaries, key=lambda s: s["accuracy"])
        best_cost = min(summaries, key=lambda s: s["avg_cost"])
        best_lat = min(summaries, key=lambda s: s["avg_latency_ms"])
        lines.append("各维度最佳配置：")
        lines.append(f"  准确率：{best_acc['config']}（{best_acc['accuracy']:.0%}）")
        lines.append(f"  成本：  {best_cost['config']}（${best_cost['avg_cost']:.4f}）")
        lines.append(f"  延迟：  {best_lat['config']}（{best_lat['avg_latency_ms']:.0f}ms）")
        lines.append("")

        # 帕累托集合
        lines.append("帕累托最优配置：")
        for p in pareto:
            lines.append(
                f"  {p['config']}：准确率={p['accuracy']:.0%}，"
                f"成本=${p['avg_cost']:.4f}，延迟={p['avg_latency_ms']:.0f}ms"
            )
        lines.append("")

        # 建议
        lines.append("建议：")
        if pareto:
            cheapest_pareto = min(pareto, key=lambda p: p["avg_cost"])
            best_pareto = max(pareto, key=lambda p: p["accuracy"])
            lines.append(f"  预算友好：{cheapest_pareto['config']}")
            lines.append(f"  质量最佳：{best_pareto['config']}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main() -> None:
    """运行包含矩阵对比和帕累托分析的完整基准测试套件。"""
    console = Console()
    console.print(
        Panel(
            "[bold cyan]完整基准测试套件[/bold cyan]\n\n"
            "带有帕累托分析的“模型 × 提示词”配置矩阵。\n"
            "识别在准确率和成本维度上不受支配的配置。",
            title="基准测试教程 3",
        )
    )

    # 确定运行模式
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    live_mode = has_anthropic and has_openai

    suite = BenchmarkSuite()
    configs = suite.build_config_matrix(MODEL_CONFIGS, PROMPT_STRATEGIES)

    console.print(
        f"配置矩阵：{len(MODEL_CONFIGS)} 个模型 × "
        f"{len(PROMPT_STRATEGIES)} 种提示词 = {len(configs)} 个配置\n"
    )

    if live_mode:
        console.print("[green]已找到 API 密钥——正在运行实时基准测试套件[/green]\n")
        results = suite.run_suite(configs, BENCHMARK_TASKS)
    else:
        console.print("[yellow]缺少 API 密钥——使用模拟结果进行演示[/yellow]\n")
        results = SIMULATED_SUITE_RESULTS

    # 计算汇总数据
    summaries = suite.compute_summary(results)
    pareto = suite.find_pareto_optimal(summaries)

    # 完整矩阵结果表
    matrix_table = Table(title="配置矩阵结果", show_lines=True)
    matrix_table.add_column("配置", style="bold", width=32)
    matrix_table.add_column("准确率", justify="center", width=10)
    matrix_table.add_column("平均延迟", justify="right", width=12)
    matrix_table.add_column("平均 Token", justify="right", width=12)
    matrix_table.add_column("平均成本", justify="right", width=10)
    matrix_table.add_column("帕累托最优", justify="center", width=12)

    pareto_names = {p["config"] for p in pareto}
    for s in summaries:
        acc = s["accuracy"]
        acc_color = "green" if acc >= 0.8 else ("yellow" if acc >= 0.6 else "red")
        is_pareto = "是" if s["config"] in pareto_names else ""
        pareto_style = "[bold green]是[/bold green]" if is_pareto else "[dim]-[/dim]"
        matrix_table.add_row(
            s["config"],
            f"[{acc_color}]{acc:.0%}[/{acc_color}]",
            f"{s['avg_latency_ms']:.0f}ms",
            f"{s['avg_tokens']:.0f}",
            f"${s['avg_cost']:.4f}",
            pareto_style,
        )

    console.print(matrix_table)
    console.print()

    # 帕累托最优配置面板
    pareto_lines: list[str] = []
    for p in pareto:
        pareto_lines.append(
            f"  [bold]{p['config']}[/bold]: "
            f"准确率={p['accuracy']:.0%}，"
            f"成本=${p['avg_cost']:.4f}，"
            f"延迟={p['avg_latency_ms']:.0f}ms"
        )
    console.print(
        Panel(
            "\n".join(pareto_lines) if pareto_lines else "未找到帕累托最优配置。",
            title="帕累托最优配置",
            subtitle="在准确率和成本维度上不受支配",
        )
    )

    # 准确率与成本散点图（文本形式）
    console.print("\n[bold]准确率与成本（文本图）[/bold]")
    sorted_by_cost = sorted(summaries, key=lambda s: s["avg_cost"])
    for s in sorted_by_cost:
        bar_len = int(s["accuracy"] * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        pareto_marker = " *" if s["config"] in pareto_names else ""
        console.print(
            f"  ${s['avg_cost']:.4f} |{bar}| {s['accuracy']:.0%}  "
            f"[dim]{s['config']}[/dim]{pareto_marker}"
        )
    console.print("  [dim]（* = 帕累托最优）[/dim]")

    # 报告
    report = suite.generate_report(summaries, pareto)
    console.print()
    console.print(Panel(report, title="基准测试报告"))

    # Token 用量（实时模式）
    if live_mode:
        console.print()
        suite.anthropic_tracker.report()
        suite.openai_tracker.report()


if __name__ == "__main__":
    main()
