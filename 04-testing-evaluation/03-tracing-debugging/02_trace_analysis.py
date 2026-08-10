"""
追踪分析

演示如何加载已记录的追踪、计算汇总指标、检测反模式并比较追踪。
使用示例追踪数据即可完全离线运行，无需 API 密钥。

核心概念：
- 汇总指标：令牌总数、成本估算、按跨度类型拆分的延迟
- 反模式检测：调用过多、重复搜索、令牌用量过高和错误
- 追踪比较：比较同一任务的两次追踪，以发现性能退化
"""

import json
from typing import Any

from common import setup_logging
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared.tracer import collect_all_spans

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

# 每个令牌的成本（近似值，仅用于教学）
COST_PER_INPUT_TOKEN = 3.0 / 1_000_000  # 每 100 万个输入令牌 3 美元
COST_PER_OUTPUT_TOKEN = 15.0 / 1_000_000  # 每 100 万个输出令牌 15 美元


# ---------------------------------------------------------------------------
# 示例追踪——内容自包含，无需 API 密钥
# ---------------------------------------------------------------------------

SAMPLE_TRACE_GOOD = {
    "trace_id": "trace_good",
    "question": "微服务有哪些优势？",
    "spans": [
        {
            "name": "answer_question",
            "span_type": "agent_step",
            "start_time": 1000.0,
            "end_time": 1003.5,
            "duration_ms": 3500.0,
            "inputs": {"question": "微服务有哪些优势？"},
            "outputs": {"answer": "微服务具有可扩展性和故障隔离能力……"},
            "tokens": {},
            "error": None,
            "children": [
                {
                    "name": "llm_call_1",
                    "span_type": "llm_call",
                    "start_time": 1000.1,
                    "end_time": 1001.2,
                    "duration_ms": 1100.0,
                    "inputs": {"message_count": 1},
                    "outputs": {"stop_reason": "tool_use"},
                    "tokens": {"input": 150, "output": 80},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "tool_search_knowledge_base",
                    "span_type": "tool_call",
                    "start_time": 1001.2,
                    "end_time": 1001.22,
                    "duration_ms": 20.0,
                    "inputs": {
                        "tool": "search_knowledge_base",
                        "input": {"query": "微服务 优势"},
                    },
                    "outputs": {"result": "[{'id': 'doc_001'}]"},
                    "tokens": {},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "llm_call_2",
                    "span_type": "llm_call",
                    "start_time": 1001.3,
                    "end_time": 1002.5,
                    "duration_ms": 1200.0,
                    "inputs": {"message_count": 3},
                    "outputs": {"stop_reason": "tool_use"},
                    "tokens": {"input": 280, "output": 60},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "tool_get_document",
                    "span_type": "tool_call",
                    "start_time": 1002.5,
                    "end_time": 1002.51,
                    "duration_ms": 10.0,
                    "inputs": {"tool": "get_document", "input": {"doc_id": "doc_001"}},
                    "outputs": {"result": "{'id': 'doc_001', 'title': '微服务架构'}"},
                    "tokens": {},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "llm_call_3",
                    "span_type": "llm_call",
                    "start_time": 1002.6,
                    "end_time": 1003.4,
                    "duration_ms": 800.0,
                    "inputs": {"message_count": 5},
                    "outputs": {"stop_reason": "end_turn"},
                    "tokens": {"input": 450, "output": 120},
                    "error": None,
                    "children": [],
                },
            ],
        }
    ],
}

SAMPLE_TRACE_ANTI_PATTERNS = {
    "trace_id": "trace_anti",
    "question": "介绍一下缓存",
    "spans": [
        {
            "name": "answer_question",
            "span_type": "agent_step",
            "start_time": 2000.0,
            "end_time": 2018.0,
            "duration_ms": 18000.0,
            "inputs": {"question": "介绍一下缓存"},
            "outputs": {"answer": "缓存是……"},
            "tokens": {},
            "error": None,
            "children": [
                {
                    "name": "llm_call_1",
                    "span_type": "llm_call",
                    "start_time": 2000.1,
                    "end_time": 2001.5,
                    "duration_ms": 1400.0,
                    "inputs": {"message_count": 1},
                    "outputs": {"stop_reason": "tool_use"},
                    "tokens": {"input": 200, "output": 90},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "tool_search_knowledge_base",
                    "span_type": "tool_call",
                    "start_time": 2001.5,
                    "end_time": 2001.52,
                    "duration_ms": 20.0,
                    "inputs": {"tool": "search_knowledge_base", "input": {"query": "缓存"}},
                    "outputs": {"result": "[{'id': 'doc_008'}]"},
                    "tokens": {},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "llm_call_2",
                    "span_type": "llm_call",
                    "start_time": 2001.6,
                    "end_time": 2003.0,
                    "duration_ms": 1400.0,
                    "inputs": {"message_count": 3},
                    "outputs": {"stop_reason": "tool_use"},
                    "tokens": {"input": 350, "output": 70},
                    "error": None,
                    "children": [],
                },
                # 重复搜索——再次使用同一查询（反模式）
                {
                    "name": "tool_search_knowledge_base",
                    "span_type": "tool_call",
                    "start_time": 2003.0,
                    "end_time": 2003.02,
                    "duration_ms": 20.0,
                    "inputs": {"tool": "search_knowledge_base", "input": {"query": "缓存"}},
                    "outputs": {"result": "[{'id': 'doc_008'}]"},
                    "tokens": {},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "llm_call_3",
                    "span_type": "llm_call",
                    "start_time": 2003.1,
                    "end_time": 2005.0,
                    "duration_ms": 1900.0,
                    "inputs": {"message_count": 5},
                    "outputs": {"stop_reason": "tool_use"},
                    "tokens": {"input": 500, "output": 100},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "tool_get_document",
                    "span_type": "tool_call",
                    "start_time": 2005.0,
                    "end_time": 2005.01,
                    "duration_ms": 10.0,
                    "inputs": {"tool": "get_document", "input": {"doc_id": "doc_008"}},
                    "outputs": {"result": "{'id': 'doc_008'}"},
                    "tokens": {},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "llm_call_4",
                    "span_type": "llm_call",
                    "start_time": 2005.1,
                    "end_time": 2007.0,
                    "duration_ms": 1900.0,
                    "inputs": {"message_count": 7},
                    "outputs": {"stop_reason": "tool_use"},
                    "tokens": {"input": 700, "output": 110},
                    "error": None,
                    "children": [],
                },
                # 重复搜索——第三次使用同一查询（反模式）
                {
                    "name": "tool_search_knowledge_base",
                    "span_type": "tool_call",
                    "start_time": 2007.0,
                    "end_time": 2007.02,
                    "duration_ms": 20.0,
                    "inputs": {"tool": "search_knowledge_base", "input": {"query": "缓存"}},
                    "outputs": {"result": "[{'id': 'doc_008'}]"},
                    "tokens": {},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "llm_call_5",
                    "span_type": "llm_call",
                    "start_time": 2007.1,
                    "end_time": 2009.0,
                    "duration_ms": 1900.0,
                    "inputs": {"message_count": 9},
                    "outputs": {"stop_reason": "tool_use"},
                    "tokens": {"input": 900, "output": 130},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "tool_get_document",
                    "span_type": "tool_call",
                    "start_time": 2009.0,
                    "end_time": 2009.01,
                    "duration_ms": 10.0,
                    "inputs": {"tool": "get_document", "input": {"doc_id": "doc_008"}},
                    "outputs": {"result": "{'id': 'doc_008'}"},
                    "tokens": {},
                    "error": None,
                    "children": [],
                },
                # 缓慢的 LLM 调用（反模式：超过 10 秒）
                {
                    "name": "llm_call_6",
                    "span_type": "llm_call",
                    "start_time": 2009.1,
                    "end_time": 2017.5,
                    "duration_ms": 8400.0,
                    "inputs": {"message_count": 11},
                    "outputs": {"stop_reason": "end_turn"},
                    "tokens": {"input": 1100, "output": 250},
                    "error": None,
                    "children": [],
                },
            ],
        }
    ],
}

SAMPLE_TRACE_ERROR = {
    "trace_id": "trace_error",
    "question": "什么是 GraphQL？",
    "spans": [
        {
            "name": "answer_question",
            "span_type": "agent_step",
            "start_time": 3000.0,
            "end_time": 3004.0,
            "duration_ms": 4000.0,
            "inputs": {"question": "什么是 GraphQL？"},
            "outputs": {},
            "tokens": {},
            "error": "未找到相关文档",
            "children": [
                {
                    "name": "llm_call_1",
                    "span_type": "llm_call",
                    "start_time": 3000.1,
                    "end_time": 3001.3,
                    "duration_ms": 1200.0,
                    "inputs": {"message_count": 1},
                    "outputs": {"stop_reason": "tool_use"},
                    "tokens": {"input": 150, "output": 70},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "tool_search_knowledge_base",
                    "span_type": "tool_call",
                    "start_time": 3001.3,
                    "end_time": 3001.32,
                    "duration_ms": 20.0,
                    "inputs": {"tool": "search_knowledge_base", "input": {"query": "GraphQL"}},
                    "outputs": {"result": "[]"},
                    "tokens": {},
                    "error": "未找到结果",
                    "children": [],
                },
                {
                    "name": "llm_call_2",
                    "span_type": "llm_call",
                    "start_time": 3001.4,
                    "end_time": 3003.8,
                    "duration_ms": 2400.0,
                    "inputs": {"message_count": 3},
                    "outputs": {"stop_reason": "end_turn"},
                    "tokens": {"input": 300, "output": 180},
                    "error": None,
                    "children": [],
                },
            ],
        }
    ],
}

ALL_SAMPLE_TRACES = {
    "good": SAMPLE_TRACE_GOOD,
    "anti_patterns": SAMPLE_TRACE_ANTI_PATTERNS,
    "error": SAMPLE_TRACE_ERROR,
}

TRACE_NAME_LABELS = {
    "good": "良好",
    "anti_patterns": "含反模式",
    "error": "错误",
}

PATTERN_LABELS = {
    "excessive_llm_calls": "LLM 调用过多",
    "repeated_tool_call": "重复工具调用",
    "high_token_usage": "令牌用量过高",
    "unretried_failure": "失败后未重试",
    "slow_span": "跨度过慢",
}

SEVERITY_LABELS = {
    "error": "错误",
    "warning": "警告",
    "info": "信息",
}


# ---------------------------------------------------------------------------
# 追踪分析
# ---------------------------------------------------------------------------


class TraceAnalyzer:
    """分析执行追踪，以检测模式并计算指标。"""

    def load_trace(self, path: str) -> dict[str, Any]:
        """从 JSON 文件加载追踪。"""
        from pathlib import Path

        with Path(path).open(encoding="utf-8") as f:
            result: dict[str, Any] = json.load(f)
            return result

    def load_trace_from_dict(self, trace_data: dict[str, Any]) -> dict[str, Any]:
        """从内存字典加载追踪。"""
        return trace_data

    def compute_metrics(self, trace: dict[str, Any]) -> dict[str, Any]:
        """计算汇总指标：令牌总数、成本、步骤数和延迟明细。"""
        all_spans = collect_all_spans(trace.get("spans", []))

        total_input_tokens = 0
        total_output_tokens = 0
        llm_latency_ms = 0.0
        tool_latency_ms = 0.0
        llm_call_count = 0
        tool_call_count = 0
        error_count = 0

        for span in all_spans:
            tokens = span.get("tokens", {})
            total_input_tokens += tokens.get("input", 0)
            total_output_tokens += tokens.get("output", 0)

            duration = span.get("duration_ms", 0.0)
            span_type = span.get("span_type", "")

            if span_type == "llm_call":
                llm_latency_ms += duration
                llm_call_count += 1
            elif span_type == "tool_call":
                tool_latency_ms += duration
                tool_call_count += 1

            if span.get("error"):
                error_count += 1

        total_tokens = total_input_tokens + total_output_tokens
        estimated_cost = (
            total_input_tokens * COST_PER_INPUT_TOKEN + total_output_tokens * COST_PER_OUTPUT_TOKEN
        )

        # 根据根跨度计算总耗时
        total_duration_ms = sum(s.get("duration_ms", 0.0) for s in trace.get("spans", []))

        return {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
            "llm_call_count": llm_call_count,
            "tool_call_count": tool_call_count,
            "error_count": error_count,
            "total_duration_ms": round(total_duration_ms, 2),
            "llm_latency_ms": round(llm_latency_ms, 2),
            "tool_latency_ms": round(tool_latency_ms, 2),
            "total_spans": len(all_spans),
        }

    def detect_anti_patterns(self, trace: dict[str, Any]) -> list[dict[str, str]]:
        """检测工具调用过多、循环和工具失败等反模式。"""
        issues: list[dict[str, str]] = []
        all_spans = collect_all_spans(trace.get("spans", []))

        # 检查 1：LLM 调用过多（单个问题超过 5 次）
        llm_calls = [s for s in all_spans if s.get("span_type") == "llm_call"]
        if len(llm_calls) > 5:
            issues.append(
                {
                    "pattern": "excessive_llm_calls",
                    "severity": "warning",
                    "message": f"发现 {len(llm_calls)} 次 LLM 调用——请考虑简化提示词",
                }
            )

        # 检查 2：重复的相同工具调用
        tool_calls = [s for s in all_spans if s.get("span_type") == "tool_call"]
        seen_calls: dict[str, int] = {}
        for tc in tool_calls:
            tool_input = tc.get("inputs", {}).get("input", {})
            key = f"{tc.get('inputs', {}).get('tool', '')}:{json.dumps(tool_input, sort_keys=True)}"
            seen_calls[key] = seen_calls.get(key, 0) + 1

        for key, count in seen_calls.items():
            if count > 1:
                issues.append(
                    {
                        "pattern": "repeated_tool_call",
                        "severity": "warning",
                        "message": f"工具调用“{key}”重复了 {count} 次——智能体可能陷入循环",
                    }
                )

        # 检查 3：令牌消耗过高（简单任务总计超过 2000 个）
        total_tokens = sum(
            s.get("tokens", {}).get("input", 0) + s.get("tokens", {}).get("output", 0)
            for s in all_spans
        )
        if total_tokens > 2000:
            issues.append(
                {
                    "pattern": "high_token_usage",
                    "severity": "info",
                    "message": f"令牌总用量为 {total_tokens}——请检查任务是否确实需要这么多令牌",
                }
            )

        # 检查 4：失败后未重试的工具调用
        failed_tools = [s for s in tool_calls if s.get("error")]
        for ft in failed_tools:
            tool_name = ft.get("inputs", {}).get("tool", "unknown")
            retried = any(
                s.get("inputs", {}).get("tool") == tool_name
                for s in tool_calls
                if s is not ft and not s.get("error")
            )
            if not retried:
                issues.append(
                    {
                        "pattern": "unretried_failure",
                        "severity": "error",
                        "message": f"工具“{tool_name}”调用失败，但未重试",
                    }
                )

        # 检查 5：跨度耗时过长（单个操作超过 10 秒）
        for span in all_spans:
            duration = span.get("duration_ms", 0.0)
            if duration > 10000 and span.get("span_type") != "agent_step":
                issues.append(
                    {
                        "pattern": "slow_span",
                        "severity": "warning",
                        "message": (
                            f"跨度“{span['name']}”耗时 {duration:.0f} 毫秒（超过 10000 毫秒阈值）"
                        ),
                    }
                )

        return issues

    def compare_traces(self, trace_a: dict[str, Any], trace_b: dict[str, Any]) -> dict[str, Any]:
        """比较同一任务的两次追踪。"""
        metrics_a = self.compute_metrics(trace_a)
        metrics_b = self.compute_metrics(trace_b)

        comparison: dict[str, Any] = {}
        for key in metrics_a:
            val_a = metrics_a[key]
            val_b = metrics_b[key]
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                diff = val_b - val_a
                pct = (diff / val_a * 100) if val_a != 0 else 0.0
                comparison[key] = {
                    "trace_a": val_a,
                    "trace_b": val_b,
                    "diff": round(diff, 4),
                    "pct_change": round(pct, 1),
                }

        return comparison


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """分析示例追踪：计算指标、检测反模式并进行比较。"""
    console = Console()

    console.print(
        Panel(
            "[bold cyan]追踪分析[/bold cyan]\n\n"
            "加载已记录的追踪、计算汇总指标、检测反模式并比较追踪。\n"
            "完全支持离线运行。\n\n"
            "概念：指标汇总、反模式检测、追踪比较",
            title="02 - 追踪分析",
        )
    )

    analyzer = TraceAnalyzer()

    # --- 指标表 ---
    console.print("\n[bold]追踪指标[/bold]\n")

    metrics_table = Table(title="各追踪的指标")
    metrics_table.add_column("指标", style="cyan")
    for name in ALL_SAMPLE_TRACES:
        metrics_table.add_column(TRACE_NAME_LABELS[name], justify="right")

    all_metrics: dict[str, dict[str, Any]] = {}
    for name, trace in ALL_SAMPLE_TRACES.items():
        all_metrics[name] = analyzer.compute_metrics(trace)

    metric_labels = {
        "total_tokens": "令牌总数",
        "total_input_tokens": "输入令牌",
        "total_output_tokens": "输出令牌",
        "estimated_cost_usd": "估算成本（美元）",
        "llm_call_count": "LLM 调用次数",
        "tool_call_count": "工具调用次数",
        "error_count": "错误数",
        "total_duration_ms": "总耗时（毫秒）",
        "llm_latency_ms": "LLM 延迟（毫秒）",
        "tool_latency_ms": "工具延迟（毫秒）",
        "total_spans": "跨度总数",
    }

    for key, label in metric_labels.items():
        row = [label]
        for name in ALL_SAMPLE_TRACES:
            val = all_metrics[name].get(key, 0)
            if key == "estimated_cost_usd":
                row.append(f"${val:.6f}")
            elif isinstance(val, float):
                row.append(f"{val:.1f}")
            else:
                row.append(str(val))
        metrics_table.add_row(*row)

    console.print(metrics_table)

    # --- 反模式检测 ---
    console.print("\n[bold]反模式检测[/bold]\n")

    for name, trace in ALL_SAMPLE_TRACES.items():
        issues = analyzer.detect_anti_patterns(trace)
        if issues:
            issue_table = Table(title=f"“{TRACE_NAME_LABELS[name]}”追踪中的问题")
            issue_table.add_column("严重程度", style="bold")
            issue_table.add_column("模式")
            issue_table.add_column("消息")
            for issue in issues:
                severity = issue["severity"]
                style = {"error": "red", "warning": "yellow", "info": "blue"}.get(severity, "")
                issue_table.add_row(
                    f"[{style}]{SEVERITY_LABELS[severity]}[/{style}]",
                    PATTERN_LABELS.get(issue["pattern"], issue["pattern"]),
                    issue["message"],
                )
            console.print(issue_table)
        else:
            console.print(f"  [green]在“{TRACE_NAME_LABELS[name]}”追踪中未检测到问题[/green]")
        console.print()

    # --- 追踪比较 ---
    console.print("[bold]追踪比较：良好追踪与含反模式追踪[/bold]\n")

    comparison = analyzer.compare_traces(SAMPLE_TRACE_GOOD, SAMPLE_TRACE_ANTI_PATTERNS)
    comp_table = Table(title="比较结果")
    comp_table.add_column("指标", style="cyan")
    comp_table.add_column("良好追踪", justify="right")
    comp_table.add_column("含反模式追踪", justify="right")
    comp_table.add_column("差值", justify="right")
    comp_table.add_column("变化百分比", justify="right")

    for key, vals in comparison.items():
        label = metric_labels.get(key, key)
        pct = vals["pct_change"]
        pct_style = "red" if pct > 0 else "green" if pct < 0 else ""
        comp_table.add_row(
            label,
            str(vals["trace_a"]),
            str(vals["trace_b"]),
            str(vals["diff"]),
            f"[{pct_style}]{pct:+.1f}%[/{pct_style}]" if pct_style else f"{pct:+.1f}%",
        )

    console.print(comp_table)


if __name__ == "__main__":
    main()
