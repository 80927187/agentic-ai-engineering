"""
Langfuse——链路追踪与评测平台。

演示如何使用 Langfuse 实现智能体的可观测性和评测。Langfuse 是一个开源且可自行托管的
平台，提供链路追踪（层次化跨度）、评分（数值、分类、布尔值）和实验跟踪功能。

本脚本将：
1. 展示基于装饰器的链路追踪模式（@observe）
2. 演示如何以编程方式为追踪评分
3. 使用数据集条目运行一个小型评测实验
4. 在没有 Langfuse 服务器时以模拟模式运行

安装：pip install langfuse
需要：LANGFUSE_SECRET_KEY、LANGFUSE_PUBLIC_KEY、LANGFUSE_BASE_URL
也可自行托管：在 Langfuse 仓库中运行 docker compose up
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any

from common import setup_logging
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared.knowledge_base import EVAL_TASKS, get_agent_response

load_dotenv(find_dotenv())

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# 模拟的 Langfuse 追踪收集器（用于在没有 Langfuse 服务器时演示）
# ---------------------------------------------------------------------------


@dataclass
class SimulatedSpan:
    """模拟的 Langfuse 观测/跨度。"""

    name: str
    span_type: str
    start_time: float = 0.0
    end_time: float = 0.0
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    scores: list[dict[str, Any]] = field(default_factory=list)
    children: list["SimulatedSpan"] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@dataclass
class SimulatedTrace:
    """支持评分的模拟 Langfuse 追踪。"""

    trace_id: str
    name: str
    spans: list[SimulatedSpan] = field(default_factory=list)
    scores: list[dict[str, Any]] = field(default_factory=list)


class SimulatedLangfuse:
    """模拟 Langfuse 的链路追踪和评分功能，以便演示。"""

    def __init__(self) -> None:
        self.traces: list[SimulatedTrace] = []
        self._current_trace: SimulatedTrace | None = None

    def start_trace(self, name: str, trace_id: str) -> SimulatedTrace:
        """开始新的追踪。"""
        trace = SimulatedTrace(trace_id=trace_id, name=name)
        self.traces.append(trace)
        self._current_trace = trace
        return trace

    def start_span(self, name: str, span_type: str = "span") -> SimulatedSpan:
        """在当前追踪中开始新的跨度。"""
        span = SimulatedSpan(name=name, span_type=span_type, start_time=time.perf_counter())
        if self._current_trace:
            self._current_trace.spans.append(span)
        return span

    def end_span(self, span: SimulatedSpan, output: dict[str, Any] | None = None) -> None:
        """结束跨度并记录输出。"""
        span.end_time = time.perf_counter()
        if output:
            span.output_data = output

    def score_trace(
        self,
        trace: SimulatedTrace,
        name: str,
        value: float | str | bool,
        data_type: str = "NUMERIC",
        comment: str = "",
    ) -> None:
        """为追踪添加评分（对应 langfuse.create_score）。"""
        trace.scores.append(
            {
                "name": name,
                "value": value,
                "data_type": data_type,
                "comment": comment,
            }
        )

    def end_trace(self, trace: SimulatedTrace) -> None:
        """结束模拟追踪（真实客户端需要关闭根观测）。"""

    def flush(self) -> None:
        """模拟客户端无需刷新。"""


class RealLangfuse:
    """将本示例使用的接口适配到真实 Langfuse SDK。"""

    def __init__(self) -> None:
        from langfuse import Langfuse

        self.client = Langfuse()
        self.traces: list[SimulatedTrace] = []
        self._current_trace: SimulatedTrace | None = None

    def start_trace(self, name: str, trace_id: str) -> SimulatedTrace:
        # 保持根观测上下文，后续 start_as_current_observation 会自动成为子跨度。
        context = self.client.start_as_current_observation(name=name, as_type="chain")
        observation = context.__enter__()
        trace = SimulatedTrace(trace_id=observation.trace_id, name=name)
        trace._context = context
        trace._observation = observation
        self.traces.append(trace)
        self._current_trace = trace
        return trace

    def start_span(self, name: str, span_type: str = "span") -> SimulatedSpan:
        span = SimulatedSpan(name=name, span_type=span_type, start_time=time.perf_counter())
        as_type = "generation" if span_type == "generation" else "span"
        context = self.client.start_as_current_observation(name=name, as_type=as_type)
        span._context = context
        span._observation = context.__enter__()
        if self._current_trace is not None:
            self._current_trace.spans.append(span)
        return span

    def end_span(self, span: SimulatedSpan, output: dict[str, Any] | None = None) -> None:
        span.end_time = time.perf_counter()
        if output:
            span.output_data = output
            span._observation.update(output=output)
        span._context.__exit__(None, None, None)

    def score_trace(
        self,
        trace: SimulatedTrace,
        name: str,
        value: float | str | bool,
        data_type: str = "NUMERIC",
        comment: str = "",
    ) -> None:
        self.client.create_score(
            trace_id=trace.trace_id,
            name=name,
            value=value,
            data_type=data_type,
            comment=comment,
        )
        trace.scores.append({"name": name, "value": value, "data_type": data_type, "comment": comment})

    def end_trace(self, trace: SimulatedTrace) -> None:
        trace._context.__exit__(None, None, None)
        self._current_trace = None

    def flush(self) -> None:
        self.client.flush()

    def shutdown(self) -> None:
        """等待后台 OTEL 导出线程完成，避免进程退出时丢失追踪。"""
        self.client.shutdown()


# ---------------------------------------------------------------------------
# 使用链路追踪和评分进行评测
# ---------------------------------------------------------------------------


def run_traced_eval(
    langfuse_client: SimulatedLangfuse,
    tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """使用 Langfuse 风格的链路追踪和评分运行评测任务。"""
    results: list[dict[str, Any]] = []

    for task in tasks:
        # 为当前评测任务开始一条追踪
        trace = langfuse_client.start_trace(
            name=f"eval_{task['id']}",
            trace_id=f"trace_{task['id']}",
        )

        # 跨度：执行智能体
        agent_span = langfuse_client.start_span("agent_call", span_type="generation")
        agent_span.input_data = {"question": task["question"]}

        response = get_agent_response(task["id"])

        langfuse_client.end_span(agent_span, output={"answer": response["answer"]})

        # 跨度：评分
        grading_span = langfuse_client.start_span("grading", span_type="span")

        # 评分：关键词覆盖率（NUMERIC）
        answer_lower = response["answer"].lower()
        keywords = task["expected_keywords"]
        if keywords:
            found = sum(1 for kw in keywords if kw.lower() in answer_lower)
            keyword_score = found / len(keywords)
        else:
            has_refusal = "无法" in response["answer"] or "没有相关" in response["answer"]
            keyword_score = 1.0 if has_refusal else 0.0

        langfuse_client.score_trace(
            trace,
            name="keyword_coverage",
            value=keyword_score,
            data_type="NUMERIC",
            comment=f"找到 {found if keywords else '不适用'}/{len(keywords)} 个关键词",
        )

        # 评分：来源依据（BOOLEAN）
        expected_sources = task.get("expected_source_ids", [])
        if expected_sources:
            all_cited = all(sid in response["answer"] for sid in expected_sources)
        else:
            all_cited = "无法" in response["answer"] or "没有相关" in response["answer"]
        langfuse_client.score_trace(
            trace,
            name="source_grounded",
            value=all_cited,
            data_type="BOOLEAN",
            comment="已引用所有预期来源" if all_cited else "缺少来源引用",
        )

        # 评分：质量类别（CATEGORICAL）
        if keyword_score >= 0.8 and all_cited:
            quality = "优秀"
        elif keyword_score >= 0.5:
            quality = "合格"
        else:
            quality = "较差"
        langfuse_client.score_trace(
            trace,
            name="quality_tier",
            value=quality,
            data_type="CATEGORICAL",
            comment=f"关键词={keyword_score:.0%}，有来源依据={all_cited}",
        )

        langfuse_client.end_span(grading_span)
        langfuse_client.end_trace(trace)

        results.append(
            {
                "task_id": task["id"],
                "trace_id": trace.trace_id,
                "keyword_score": keyword_score,
                "grounded": all_cited,
                "quality": quality,
                "duration_ms": sum(s.duration_ms for s in trace.spans),
            }
        )

    return results


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main() -> None:
    """对研究助手运行 Langfuse 风格的链路追踪评测。"""
    console = Console()
    console.print(
        Panel(
            "[bold cyan]Langfuse——链路追踪与评测平台[/bold cyan]\n\n"
            "演示用于智能体评测的 Langfuse 模式：\n"
            "  - 基于装饰器的链路追踪（@observe）\n"
            "  - 程序化评分（NUMERIC、BOOLEAN、CATEGORICAL）\n"
            "  - 使用数据集跟踪实验\n\n"
            "开源且可自行托管。安装：pip install langfuse",
            title="03 - Langfuse",
        )
    )

    # 检查 Langfuse SDK 和凭据
    has_langfuse = False
    try:
        import langfuse  # noqa: F401

        has_langfuse = True
    except ImportError:
        pass

    has_langfuse_keys = bool(
        os.environ.get("LANGFUSE_SECRET_KEY") and os.environ.get("LANGFUSE_PUBLIC_KEY")
    )

    if has_langfuse and has_langfuse_keys:
        console.print("[green]已找到 Langfuse SDK 和密钥——追踪数据将发送到服务器[/green]")
    elif has_langfuse:
        console.print(
            "[yellow]已安装 Langfuse SDK，但未设置密钥——以模拟模式运行[/yellow]"
        )
    else:
        console.print("[yellow]未安装 Langfuse——运行模拟演示[/yellow]")
    console.print()

    # 有凭据时使用真实 SDK，否则使用本地模拟客户端。
    langfuse_client = RealLangfuse() if has_langfuse and has_langfuse_keys else SimulatedLangfuse()
    results = run_traced_eval(langfuse_client, EVAL_TASKS)
    langfuse_client.flush()
    if hasattr(langfuse_client, "shutdown"):
        langfuse_client.shutdown()

    # 结果表格
    table = Table(title="Langfuse 链路追踪评测结果", show_lines=True)
    table.add_column("任务", style="cyan", width=12)
    table.add_column("追踪 ID", width=16)
    table.add_column("关键词", width=10, justify="center")
    table.add_column("有依据", width=10, justify="center")
    table.add_column("质量", width=12, justify="center")
    table.add_column("耗时", width=10, justify="right")

    for r in results:
        kw_color = "green" if r["keyword_score"] >= 0.7 else "yellow"
        grounded_str = "[green]是[/green]" if r["grounded"] else "[red]否[/red]"
        quality_color = {
            "优秀": "green",
            "合格": "yellow",
            "较差": "red",
        }.get(r["quality"], "dim")

        table.add_row(
            r["task_id"],
            r["trace_id"],
            f"[{kw_color}]{r['keyword_score']:.0%}[/{kw_color}]",
            grounded_str,
            f"[{quality_color}]{r['quality']}[/{quality_color}]",
            f"{r['duration_ms']:.1f}ms",
        )

    console.print(table)

    # 追踪汇总
    console.print(
        f"\n[bold]已收集追踪数：[/bold] {len(langfuse_client.traces)}\n"
        f"[bold]评分总数：[/bold] "
        f"{sum(len(t.scores) for t in langfuse_client.traces)}\n"
        f"[bold]跨度总数：[/bold] "
        f"{sum(len(t.spans) for t in langfuse_client.traces)}"
    )

    # 评分类型明细
    score_types = {"NUMERIC": 0, "BOOLEAN": 0, "CATEGORICAL": 0}
    for trace in langfuse_client.traces:
        for score in trace.scores:
            score_types[score["data_type"]] = score_types.get(score["data_type"], 0) + 1

    console.print("\n[bold]使用的评分类型：[/bold]")
    for dtype, count in score_types.items():
        console.print(f"  {dtype}: {count}")

    # 展示 Langfuse 代码模式
    console.print("\n[bold]Langfuse SDK 模式：[/bold]\n")
    from rich.syntax import Syntax

    decorator_code = (
        "from langfuse import observe, get_client\n\n"
        "@observe()  # 自动创建追踪\n"
        "def my_agent(question: str) -> str:\n"
        "    result = search_and_answer(question)\n"
        "    return result\n\n"
        '@observe(name="llm-call", as_type="generation")\n'
        "def search_and_answer(question: str) -> str:\n"
        "    # 自动捕获嵌套跨度\n"
        "    return call_llm(question)\n"
    )
    console.print(Syntax(decorator_code, "python", theme="monokai", line_numbers=True))

    scoring_code = (
        "langfuse = get_client()\n\n"
        "# 执行完成后评分\n"
        "langfuse.create_score(\n"
        "    trace_id=trace_id,\n"
        '    name="correctness",\n'
        "    value=0.95,\n"
        '    data_type="NUMERIC",\n'
        '    comment="事实准确",\n'
        ")\n"
    )
    console.print(Syntax(scoring_code, "python", theme="monokai", line_numbers=True))


if __name__ == "__main__":
    main()
