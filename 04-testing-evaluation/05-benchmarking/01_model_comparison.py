"""
模型对比基准测试

使用多个模型和服务商对相同的研究助手任务进行基准测试。
测量准确率（关键词匹配）、延迟、Token 用量和单次查询成本。
既支持实时 API 调用，也支持无需 API 密钥的演示模拟模式。
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
    SYSTEM_PROMPT,
    TOOLS_ANTHROPIC,
    TOOLS_OPENAI,
    score_answer,
    search_knowledge_base,
)
from shared.models import MODEL_CONFIGS, BenchmarkResult, ModelConfig

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# 演示模式的模拟结果
# ---------------------------------------------------------------------------

SIMULATED_RESULTS = [
    # bench_001——微服务
    BenchmarkResult(
        "bench_001",
        "Claude Sonnet",
        "微服务提供可扩展性、故障隔离和独立部署能力（doc_001）。",
        0.9,
        1200,
        150,
        200,
        0.0035,
        1,
    ),
    BenchmarkResult(
        "bench_001",
        "Claude Haiku",
        "其优势包括故障隔离和可扩展性（doc_001）。",
        0.7,
        450,
        130,
        150,
        0.0007,
        1,
    ),
    BenchmarkResult(
        "bench_001",
        "GPT-4.1 mini",
        "主要优势包括可扩展性、故障隔离和独立服务（doc_001）。",
        0.8,
        800,
        140,
        180,
        0.0003,
        1,
    ),
    # bench_002——REST API
    BenchmarkResult(
        "bench_002",
        "Claude Sonnet",
        (
            "REST API 的端点使用名词，操作使用 HTTP 方法，"
            "结果使用状态码（doc_002）。"
        ),
        1.0,
        1150,
        145,
        190,
        0.0033,
        1,
    ),
    BenchmarkResult(
        "bench_002",
        "Claude Haiku",
        "使用名词和 HTTP 方法，并配以适当的状态码（doc_002）。",
        0.8,
        420,
        125,
        140,
        0.0007,
        1,
    ),
    BenchmarkResult(
        "bench_002",
        "GPT-4.1 mini",
        "使用名词设计端点，并使用 HTTP 方法和状态码（doc_002）。",
        0.9,
        780,
        135,
        170,
        0.0003,
        1,
    ),
    # bench_003——数据库索引
    BenchmarkResult(
        "bench_003",
        "Claude Sonnet",
        (
            "B 树索引处理等值查询，复合索引可提升"
            "多列查询性能（doc_003）。"
        ),
        0.9,
        1300,
        155,
        210,
        0.0036,
        1,
    ),
    BenchmarkResult(
        "bench_003",
        "Claude Haiku",
        "数据库索引使用 B 树结构提高查询性能（doc_003）。",
        0.6,
        440,
        128,
        145,
        0.0007,
        1,
    ),
    BenchmarkResult(
        "bench_003",
        "GPT-4.1 mini",
        "B 树索引和复合索引可以提高查询性能（doc_003）。",
        0.8,
        820,
        138,
        175,
        0.0003,
        1,
    ),
    # bench_004——身份认证与授权
    BenchmarkResult(
        "bench_004",
        "Claude Sonnet",
        (
            "身份认证用于验证身份，而授权用于控制访问权限。"
            "JWT 和 OAuth 2.0 是关键机制（doc_004）。"
        ),
        0.9,
        1250,
        148,
        205,
        0.0035,
        1,
    ),
    BenchmarkResult(
        "bench_004",
        "Claude Haiku",
        "身份认证确认身份，授权控制访问，可使用 JWT 令牌（doc_004）。",
        0.6,
        430,
        122,
        138,
        0.0006,
        1,
    ),
    BenchmarkResult(
        "bench_004",
        "GPT-4.1 mini",
        (
            "身份认证验证身份，授权则通过 JWT 和 OAuth 控制"
            "访问权限（doc_004）。"
        ),
        0.8,
        810,
        132,
        172,
        0.0003,
        1,
    ),
    # bench_005——CI/CD
    BenchmarkResult(
        "bench_005",
        "Claude Sonnet",
        (
            "CI 通过自动化测试和快速反馈循环"
            "实现持续集成（doc_005）。"
        ),
        0.8,
        1180,
        142,
        195,
        0.0034,
        1,
    ),
    BenchmarkResult(
        "bench_005",
        "Claude Haiku",
        "持续集成包括自动构建和快速反馈（doc_005）。",
        0.7,
        460,
        126,
        142,
        0.0007,
        1,
    ),
    BenchmarkResult(
        "bench_005",
        "GPT-4.1 mini",
        "CI/CD 的关键实践包括持续自动化测试和反馈循环（doc_005）。",
        0.7,
        790,
        130,
        168,
        0.0003,
        1,
    ),
]

# ---------------------------------------------------------------------------
# 模型基准测试类
# ---------------------------------------------------------------------------


class ModelBenchmark:
    """使用多个模型对相同任务进行基准测试。"""

    def __init__(self) -> None:
        self.anthropic_tracker = AnthropicTokenTracker()
        self.openai_tracker = OpenAITokenTracker()

    def run_task_anthropic(self, task: dict, config: ModelConfig) -> BenchmarkResult:
        """使用 Anthropic API 运行单个基准测试任务。"""
        client = anthropic.Anthropic()
        messages: list[dict[str, Any]] = [{"role": "user", "content": task["question"]}]
        tool_call_count = 0

        start = time.perf_counter()

        # 智能体循环——处理工具调用，直至获得最终响应
        for _turn in range(100):
            response = client.messages.create(
                model=config.model_id,
                max_tokens=21333,
                system=SYSTEM_PROMPT,
                tools=TOOLS_ANTHROPIC,
                messages=messages,
            )
            self.anthropic_tracker.track(response.usage)

            if response.stop_reason != "tool_use":
                text_parts = [b.text for b in response.content if b.type == "text"]
                if not text_parts:
                    block_types = [b.type for b in response.content]
                    raise ValueError(
                        f"模型响应中没有文本内容（stop_reason={response.stop_reason}，"
                        f"内容块={block_types}，output_tokens={response.usage.output_tokens}）。"
                    )
                answer = "\n\n".join(text_parts)
                break

            # 处理工具调用
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

        else:
            raise RuntimeError("工具调用循环超过 100 轮")

        latency_ms = (time.perf_counter() - start) * 1000
        input_tok = response.usage.input_tokens
        output_tok = response.usage.output_tokens
        cost = (
            input_tok * config.cost_per_input_token + output_tok * config.cost_per_output_token
        ) / 1_000_000

        return BenchmarkResult(
            task_id=task["id"],
            config_name=config.name,
            answer=answer,
            keyword_score=score_answer(answer, task["expected_keywords"]),
            latency_ms=latency_ms,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=cost,
            tool_calls=tool_call_count,
        )

    def run_task_openai(self, task: dict, config: ModelConfig) -> BenchmarkResult:
        """使用 OpenAI API 运行单个基准测试任务。"""
        client = openai.OpenAI()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": task["question"]},
        ]
        tool_call_count = 0

        start = time.perf_counter()

        # 智能体循环——处理函数调用，直至获得最终响应
        for _turn in range(100):
            response = client.responses.create(
                model=config.model_id,
                instructions=SYSTEM_PROMPT,
                max_output_tokens=21333,
                tools=TOOLS_OPENAI,
                input=messages,
            )
            self.openai_tracker.track(response.usage)

            function_calls = [o for o in response.output if o.type == "function_call"]

            if not function_calls:
                answer = response.output_text or ""
                break

            # 处理函数调用
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

        else:
            raise RuntimeError("工具调用循环超过 100 轮")

        latency_ms = (time.perf_counter() - start) * 1000
        input_tok = response.usage.input_tokens
        output_tok = response.usage.output_tokens
        cost = (
            input_tok * config.cost_per_input_token + output_tok * config.cost_per_output_token
        ) / 1_000_000

        return BenchmarkResult(
            task_id=task["id"],
            config_name=config.name,
            answer=answer,
            keyword_score=score_answer(answer, task["expected_keywords"]),
            latency_ms=latency_ms,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=cost,
            tool_calls=tool_call_count,
        )

    def run_benchmark(self, tasks: list[dict], configs: list[ModelConfig]) -> list[BenchmarkResult]:
        """使用所有模型配置运行全部任务。"""
        results: list[BenchmarkResult] = []
        for config in configs:
            logger.info("正在对模型进行基准测试：%s（%s）", config.name, config.model_id)
            for task in tasks:
                logger.info("  任务 %s：%s", task["id"], task["question"][:50])
                try:
                    if config.provider == "anthropic":
                        result = self.run_task_anthropic(task, config)
                    elif config.provider == "openai":
                        result = self.run_task_openai(task, config)
                    else:
                        logger.error("未知服务商：%s", config.provider)
                        continue
                    results.append(result)
                    logger.info(
                        "    得分=%.2f，延迟=%dms，成本=$%.4f",
                        result.keyword_score,
                        result.latency_ms,
                        result.cost_usd,
                    )
                except Exception as e:
                    logger.error("    错误：%s", e)
        return results


# ---------------------------------------------------------------------------
# 聚合辅助函数
# ---------------------------------------------------------------------------


def aggregate_by_model(results: list[BenchmarkResult]) -> dict[str, dict[str, float]]:
    """计算每个模型在所有任务上的平均值。"""
    model_results: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        model_results.setdefault(r.config_name, []).append(r)

    summaries: dict[str, dict[str, float]] = {}
    for model, mrs in model_results.items():
        n = len(mrs)
        summaries[model] = {
            "accuracy": sum(r.keyword_score for r in mrs) / n,
            "avg_latency_ms": sum(r.latency_ms for r in mrs) / n,
            "avg_tokens": sum(r.input_tokens + r.output_tokens for r in mrs) / n,
            "avg_cost": sum(r.cost_usd for r in mrs) / n,
            "tasks": n,
        }
    return summaries


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main() -> None:
    """运行模型对比基准测试并显示结果。"""
    console = Console()
    console.print(
        Panel(
            "[bold cyan]模型对比基准测试[/bold cyan]\n\n"
            "使用多个模型比较相同的研究助手任务。\n"
            "测量指标：准确率（关键词匹配）、延迟、Token 用量和成本。",
            title="基准测试教程 1",
        )
    )

    # 确定运行模式：实时或模拟
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    live_mode = has_anthropic and has_openai

    if live_mode:
        console.print("[green]已找到 API 密钥——正在运行实时基准测试[/green]\n")
        benchmark = ModelBenchmark()
        results = benchmark.run_benchmark(BENCHMARK_TASKS, MODEL_CONFIGS)
    else:
        console.print("[yellow]缺少 API 密钥——使用模拟结果进行演示[/yellow]\n")
        results = SIMULATED_RESULTS

    # 各任务明细表
    detail_table = Table(title="各任务结果", show_lines=True)
    detail_table.add_column("任务", style="cyan", width=10)
    detail_table.add_column("模型", width=14)
    detail_table.add_column("得分", justify="center", width=7)
    detail_table.add_column("延迟", justify="right", width=9)
    detail_table.add_column("Token", justify="right", width=8)
    detail_table.add_column("成本", justify="right", width=9)
    detail_table.add_column("工具", justify="center", width=6)

    for r in results:
        score_color = (
            "green" if r.keyword_score >= 0.8 else ("yellow" if r.keyword_score >= 0.5 else "red")
        )
        detail_table.add_row(
            r.task_id,
            r.config_name,
            f"[{score_color}]{r.keyword_score:.0%}[/{score_color}]",
            f"{r.latency_ms:.0f}ms",
            str(r.input_tokens + r.output_tokens),
            f"${r.cost_usd:.4f}",
            str(r.tool_calls),
        )

    console.print(detail_table)
    console.print()

    # 聚合对比表
    summaries = aggregate_by_model(results)

    summary_table = Table(title="模型对比汇总", show_lines=True)
    summary_table.add_column("模型", style="bold", width=14)
    summary_table.add_column("准确率", justify="center", width=10)
    summary_table.add_column("平均延迟", justify="right", width=12)
    summary_table.add_column("平均 Token", justify="right", width=12)
    summary_table.add_column("平均成本", justify="right", width=10)

    for model, stats in summaries.items():
        acc = stats["accuracy"]
        acc_color = "green" if acc >= 0.8 else ("yellow" if acc >= 0.6 else "red")
        summary_table.add_row(
            model,
            f"[{acc_color}]{acc:.0%}[/{acc_color}]",
            f"{stats['avg_latency_ms']:.0f}ms",
            f"{stats['avg_tokens']:.0f}",
            f"${stats['avg_cost']:.4f}",
        )

    console.print(summary_table)

    # 突出显示各维度上的最佳模型
    console.print("\n[bold]各维度最佳模型[/bold]")
    best_acc = max(summaries.items(), key=lambda x: x[1]["accuracy"])
    best_lat = min(summaries.items(), key=lambda x: x[1]["avg_latency_ms"])
    best_cost = min(summaries.items(), key=lambda x: x[1]["avg_cost"])
    console.print(f"  准确率：{best_acc[0]}（{best_acc[1]['accuracy']:.0%}）")
    console.print(f"  延迟：  {best_lat[0]}（{best_lat[1]['avg_latency_ms']:.0f}ms）")
    console.print(f"  成本：  {best_cost[0]}（${best_cost[1]['avg_cost']:.4f}）")

    # Token 用量报告（实时模式）
    if live_mode:
        console.print()
        benchmark.anthropic_tracker.report()
        benchmark.openai_tracker.report()


if __name__ == "__main__":
    main()
