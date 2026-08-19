"""
提示词策略对比基准测试

在同一模型和相同任务上对三种提示词策略（零样本、少样本、思维链）进行基准测试。
测量提示工程对准确率、回答详尽程度和成本的影响。
既支持实时 API 调用，也支持无需 API 密钥的演示模拟模式。
"""

import json
import os
import time
from typing import Any

import anthropic
from common import AnthropicTokenTracker, setup_logging
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared.knowledge_base import (
    BENCHMARK_TASKS,
    TOOLS_ANTHROPIC,
    score_answer,
    search_knowledge_base,
)
from shared.models import BenchmarkResult

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# 提示词策略——测试中的核心变量
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

# 提示词对比的默认模型——隔离提示词变量
DEFAULT_MODEL = "deepseek-v4-flash"
COST_PER_INPUT = 3.0  # 每 100 万输入 Token 的美元价格
COST_PER_OUTPUT = 15.0  # 每 100 万输出 Token 的美元价格

# ---------------------------------------------------------------------------
# 演示模式的模拟结果
# ---------------------------------------------------------------------------

SIMULATED_RESULTS: dict[str, list[BenchmarkResult]] = {
    "zero_shot": [
        BenchmarkResult(
            "bench_001",
            "zero_shot",
            "微服务提供可扩展性和故障隔离能力（doc_001）。",
            0.7,
            1100,
            140,
            120,
            0.0022,
            1,
        ),
        BenchmarkResult(
            "bench_002",
            "zero_shot",
            "端点使用名词，操作使用 HTTP 方法（doc_002）。",
            0.7,
            1050,
            135,
            115,
            0.0021,
            1,
        ),
        BenchmarkResult(
            "bench_003",
            "zero_shot",
            "B 树索引可以提高查询性能（doc_003）。",
            0.6,
            1120,
            142,
            118,
            0.0022,
            1,
        ),
        BenchmarkResult(
            "bench_004",
            "zero_shot",
            "身份认证用于验证身份，授权用于控制访问权限（doc_004）。",
            0.5,
            1080,
            138,
            122,
            0.0022,
            1,
        ),
        BenchmarkResult(
            "bench_005",
            "zero_shot",
            "CI/CD 包括自动化测试和持续部署（doc_005）。",
            0.6,
            1090,
            136,
            116,
            0.0022,
            1,
        ),
    ],
    "few_shot": [
        BenchmarkResult(
            "bench_001",
            "few_shot",
            (
                "微服务架构提供可扩展性、故障隔离和"
                "独立部署能力（doc_001）。"
            ),
            0.9,
            1250,
            185,
            160,
            0.0030,
            1,
        ),
        BenchmarkResult(
            "bench_002",
            "few_shot",
            (
                "REST API 端点使用名词，操作使用 HTTP 方法，"
                "结果使用状态码（doc_002）。"
            ),
            1.0,
            1200,
            180,
            155,
            0.0029,
            1,
        ),
        BenchmarkResult(
            "bench_003",
            "few_shot",
            (
                "数据库索引包括 B 树索引和复合索引，"
                "可以提高查询性能（doc_003）。"
            ),
            0.9,
            1280,
            188,
            162,
            0.0030,
            1,
        ),
        BenchmarkResult(
            "bench_004",
            "few_shot",
            (
                "身份认证用于验证身份，授权用于控制访问权限。"
                "其中会使用 JWT 和 OAuth 2.0（doc_004）。"
            ),
            0.8,
            1220,
            182,
            158,
            0.0029,
            1,
        ),
        BenchmarkResult(
            "bench_005",
            "few_shot",
            (
                "CI/CD 的关键实践包括持续集成、"
                "自动化测试和快速反馈循环（doc_005）。"
            ),
            0.9,
            1240,
            184,
            156,
            0.0029,
            1,
        ),
    ],
    "chain_of_thought": [
        BenchmarkResult(
            "bench_001",
            "chain_of_thought",
            (
                "步骤 1：搜索微服务。步骤 2：关键事实包括可扩展性、"
                "故障隔离和独立部署。步骤 3：微服务支持独立扩缩容"
                "和故障隔离（doc_001）。"
            ),
            0.9,
            1500,
            195,
            250,
            0.0043,
            1,
        ),
        BenchmarkResult(
            "bench_002",
            "chain_of_thought",
            (
                "步骤 1：搜索 REST API。步骤 2：端点使用名词，并使用"
                "HTTP 方法和状态码。步骤 3：REST API 应使用名词、"
                "HTTP 方法和状态码（doc_002）。"
            ),
            1.0,
            1450,
            190,
            245,
            0.0042,
            1,
        ),
        BenchmarkResult(
            "bench_003",
            "chain_of_thought",
            (
                "步骤 1：搜索索引。步骤 2：关注 B 树、复合索引和查询性能。"
                "步骤 3：B 树索引和复合索引可提高查询性能（doc_003）。"
            ),
            0.9,
            1520,
            198,
            255,
            0.0044,
            1,
        ),
        BenchmarkResult(
            "bench_004",
            "chain_of_thought",
            (
                "步骤 1：搜索身份认证与授权。步骤 2：关注身份、访问权限、"
                "JWT 和 OAuth。步骤 3：身份认证用于验证身份，授权则使用 JWT 和"
                "OAuth 控制访问权限（doc_004）。"
            ),
            1.0,
            1480,
            192,
            248,
            0.0043,
            1,
        ),
        BenchmarkResult(
            "bench_005",
            "chain_of_thought",
            (
                "步骤 1：搜索 CI/CD。步骤 2：关注持续、自动化和反馈。"
                "步骤 3：CI/CD 依靠持续自动构建和快速反馈循环（doc_005）。"
            ),
            0.9,
            1510,
            196,
            252,
            0.0044,
            1,
        ),
    ],
}

# ---------------------------------------------------------------------------
# 提示词基准测试类
# ---------------------------------------------------------------------------


class PromptBenchmark:
    """在同一模型和相同任务上对不同提示词策略进行基准测试。"""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self.token_tracker = AnthropicTokenTracker()

    def run_with_prompt(self, task: dict, prompt_name: str, system_prompt: str) -> BenchmarkResult:
        """使用指定的提示词策略运行单个任务。"""
        client = anthropic.Anthropic()
        messages: list[dict[str, Any]] = [{"role": "user", "content": task["question"]}]
        tool_call_count = 0

        start = time.perf_counter()

        for _turn in range(100):
            response = client.messages.create(
                model=self.model,
                max_tokens=21333,
                system=system_prompt,
                tools=TOOLS_ANTHROPIC,
                messages=messages,
            )
            self.token_tracker.track(response.usage)

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
        cost = (input_tok * COST_PER_INPUT + output_tok * COST_PER_OUTPUT) / 1_000_000

        return BenchmarkResult(
            task_id=task["id"],
            config_name=prompt_name,
            answer=answer,
            keyword_score=score_answer(answer, task["expected_keywords"]),
            latency_ms=latency_ms,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=cost,
            tool_calls=tool_call_count,
        )

    def run_comparison(self, tasks: list[dict]) -> dict[str, list[BenchmarkResult]]:
        """使用每种提示词策略运行全部任务。"""
        all_results: dict[str, list[BenchmarkResult]] = {}

        for prompt_name, system_prompt in PROMPT_STRATEGIES.items():
            logger.info("正在运行提示词策略：%s", prompt_name)
            strategy_results: list[BenchmarkResult] = []

            for task in tasks:
                logger.info("  任务 %s：%s", task["id"], task["question"][:50])
                try:
                    result = self.run_with_prompt(task, prompt_name, system_prompt)
                    strategy_results.append(result)
                    logger.info(
                        "    得分=%.2f，延迟=%dms，Token=%d",
                        result.keyword_score,
                        result.latency_ms,
                        result.input_tokens + result.output_tokens,
                    )
                except Exception as e:
                    logger.error("    错误：%s", e)

            all_results[prompt_name] = strategy_results

        return all_results


# ---------------------------------------------------------------------------
# 聚合辅助函数
# ---------------------------------------------------------------------------


def aggregate_by_strategy(
    results: dict[str, list[BenchmarkResult]],
) -> dict[str, dict[str, float]]:
    """计算各策略的平均值。"""
    summaries: dict[str, dict[str, float]] = {}
    for strategy, res_list in results.items():
        n = len(res_list)
        if n == 0:
            continue
        summaries[strategy] = {
            "accuracy": sum(r.keyword_score for r in res_list) / n,
            "avg_latency_ms": sum(r.latency_ms for r in res_list) / n,
            "avg_output_tokens": sum(r.output_tokens for r in res_list) / n,
            "avg_cost": sum(r.cost_usd for r in res_list) / n,
            "tasks": n,
        }
    return summaries


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main() -> None:
    """运行提示词策略对比并显示结果。"""
    console = Console()
    console.print(
        Panel(
            "[bold cyan]提示词策略对比[/bold cyan]\n\n"
            "同一模型，三种提示词策略：零样本、少样本和思维链。\n"
            "测量提示工程对准确率、回答详尽程度和成本的影响。",
            title="基准测试教程 2",
        )
    )

    # 确定运行模式
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if has_api_key:
        console.print(
            f"[green]已找到 API 密钥——正在使用 {DEFAULT_MODEL} 运行实时基准测试[/green]\n"
        )
        benchmark = PromptBenchmark()
        results = benchmark.run_comparison(BENCHMARK_TASKS)
    else:
        console.print("[yellow]没有 API 密钥——使用模拟结果进行演示[/yellow]\n")
        results = SIMULATED_RESULTS

    # 按提示词策略展示各任务明细
    detail_table = Table(title="各提示词策略的任务结果", show_lines=True)
    detail_table.add_column("任务", style="cyan", width=10)
    detail_table.add_column("策略", width=16)
    detail_table.add_column("得分", justify="center", width=7)
    detail_table.add_column("延迟", justify="right", width=9)
    detail_table.add_column("输出 Token", justify="right", width=10)
    detail_table.add_column("成本", justify="right", width=9)

    for strategy, res_list in results.items():
        for r in res_list:
            score_color = (
                "green"
                if r.keyword_score >= 0.8
                else ("yellow" if r.keyword_score >= 0.5 else "red")
            )
            detail_table.add_row(
                r.task_id,
                PROMPT_STRATEGY_LABELS.get(strategy, strategy),
                f"[{score_color}]{r.keyword_score:.0%}[/{score_color}]",
                f"{r.latency_ms:.0f}ms",
                str(r.output_tokens),
                f"${r.cost_usd:.4f}",
            )

    console.print(detail_table)
    console.print()

    # 汇总对比表
    summaries = aggregate_by_strategy(results)

    summary_table = Table(title="提示词策略对比汇总", show_lines=True)
    summary_table.add_column("策略", style="bold", width=18)
    summary_table.add_column("准确率", justify="center", width=10)
    summary_table.add_column("平均延迟", justify="right", width=12)
    summary_table.add_column("平均输出 Token", justify="right", width=14)
    summary_table.add_column("平均成本", justify="right", width=10)

    for strategy, stats in summaries.items():
        acc = stats["accuracy"]
        acc_color = "green" if acc >= 0.8 else ("yellow" if acc >= 0.6 else "red")
        summary_table.add_row(
            PROMPT_STRATEGY_LABELS.get(strategy, strategy),
            f"[{acc_color}]{acc:.0%}[/{acc_color}]",
            f"{stats['avg_latency_ms']:.0f}ms",
            f"{stats['avg_output_tokens']:.0f}",
            f"${stats['avg_cost']:.4f}",
        )

    console.print(summary_table)

    # 分析
    console.print("\n[bold]分析[/bold]")
    best_acc = max(summaries.items(), key=lambda x: x[1]["accuracy"])
    cheapest = min(summaries.items(), key=lambda x: x[1]["avg_cost"])
    most_verbose = max(summaries.items(), key=lambda x: x[1]["avg_output_tokens"])
    console.print(
        f"  准确率最高：{PROMPT_STRATEGY_LABELS.get(best_acc[0], best_acc[0])}"
        f"（{best_acc[1]['accuracy']:.0%}）"
    )
    console.print(
        f"  成本最低：  {PROMPT_STRATEGY_LABELS.get(cheapest[0], cheapest[0])}"
        f"（${cheapest[1]['avg_cost']:.4f}）"
    )
    console.print(
        f"  回答最详尽：{PROMPT_STRATEGY_LABELS.get(most_verbose[0], most_verbose[0])} "
        f"（{most_verbose[1]['avg_output_tokens']:.0f} Token）"
    )

    # 权衡分析
    console.print(
        Panel(
            "少样本提示词通常通过提供输出格式示例来提高准确率。\n"
            "思维链会增加 Token 用量（成本），但可能提升推理质量。\n"
            "零样本成本最低，但可能遗漏细节。请根据准确率与成本预算做出选择。",
            title="关键洞见",
            style="dim",
        )
    )

    # Token 用量报告（实时模式）
    if has_api_key:
        console.print()
        benchmark.token_tracker.report()


if __name__ == "__main__":
    main()
