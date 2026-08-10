"""
追踪收集器

演示如何为智能体执行构建纯 Python 追踪系统。使用分层跨度记录每一次 LLM 调用、
工具调用和智能体步骤，捕获耗时、令牌用量、输入和输出——这是智能体可观测性的基础。

核心概念：
- 基于跨度的追踪：将操作嵌套成树，以查看完整执行过程
- 上下文管理器跨度：正确嵌套并自动记录开始和结束时间
- 基于装饰器的追踪：无需修改函数体即可添加追踪
- 追踪序列化：将追踪导出为 JSON，供后续分析和调试
"""

import os
from typing import Any

import anthropic
from common import setup_logging
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from shared.agent import TracedResearchAssistant
from shared.tracer import TraceCollector

load_dotenv(find_dotenv())

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# 可视化辅助函数
# ---------------------------------------------------------------------------


def build_span_tree(span_data: dict[str, Any], tree: Tree) -> None:
    """根据跨度数据递归构建 Rich 树。"""
    duration = span_data.get("duration_ms", 0)
    tokens = span_data.get("tokens", {})
    error = span_data.get("error")

    label = f"[bold]{span_data['name']}[/bold] [{span_data['span_type']}]"
    label += f"  {duration:.1f}ms"
    if tokens:
        label += f"  令牌: {tokens.get('input', 0)}入/{tokens.get('output', 0)}出"
    if error:
        label += f"  [red]错误: {error}[/red]"

    branch = tree.add(label)
    for child in span_data.get("children", []):
        build_span_tree(child, branch)


# ---------------------------------------------------------------------------
# 离线模式的示例追踪
# ---------------------------------------------------------------------------

SAMPLE_TRACE = {
    "trace_id": "sample_001",
    "question": "微服务有哪些优势？",
    "spans": [
        {
            "name": "answer_question",
            "span_type": "agent_step",
            "start_time": 1000.0,
            "end_time": 1003.5,
            "duration_ms": 3500.0,
            "inputs": {
                "question": "微服务有哪些优势？",
            },
            "outputs": {
                "answer": "微服务具有可扩展性……",
                "llm_calls": 3,
            },
            "metadata": {},
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
                    "metadata": {},
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
                    "outputs": {
                        "result": "[{'id': 'doc_001', 'title': '微服务架构'}]",
                    },
                    "metadata": {},
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
                    "metadata": {},
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
                    "inputs": {
                        "tool": "get_document",
                        "input": {"doc_id": "doc_001"},
                    },
                    "outputs": {
                        "result": "{'id': 'doc_001', 'title': '微服务架构'}",
                    },
                    "metadata": {},
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
                    "metadata": {},
                    "tokens": {"input": 450, "output": 120},
                    "error": None,
                    "children": [],
                },
            ],
        }
    ],
}


def main() -> None:
    """运行带追踪的研究助手，并将执行追踪可视化。"""
    console = Console()

    console.print(
        Panel(
            "[bold cyan]追踪收集器[/bold cyan]\n\n"
            "使用分层跨度记录智能体执行，捕获每个操作的耗时、\n"
            "令牌用量、输入和输出。\n\n"
            "概念：跨度层次结构、上下文管理器追踪、追踪序列化",
            title="01 - 追踪收集器",
        )
    )

    # 确定运行模式
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if has_api_key:
        console.print("\n[green]已找到 API 密钥——正在运行实时追踪智能体[/green]\n")
        client = anthropic.Anthropic()
        tracer = TraceCollector()
        assistant = TracedResearchAssistant(client, tracer)

        questions = [
            "微服务有哪些优势？",
            "应该如何设计 REST API？",
        ]

        for question in questions:
            console.print(f"\n[bold yellow]问题：[/bold yellow] {question}")
            try:
                result = assistant.answer(question)
                console.print(f"[dim]回答：{result['answer'][:150]}……[/dim]")
                console.print(f"[dim]LLM 调用次数：{result['llm_calls']}[/dim]")
            except Exception as e:
                logger.error("回答问题时出错：%s", e)

        trace_data = tracer.to_dict()
        trace_path = "trace_output.json"
        tracer.save(trace_path)
        console.print(f"\n[green]追踪已保存到 {trace_path}[/green]")

    else:
        console.print("\n[yellow]未找到 API 密钥——使用示例追踪数据[/yellow]\n")
        trace_data = SAMPLE_TRACE

    # 将追踪可视化为树
    console.print("\n[bold]追踪可视化[/bold]\n")

    tree = Tree(f"[bold magenta]追踪 {trace_data.get('trace_id', '未知')}[/bold magenta]")
    for span_data in trace_data.get("spans", []):
        build_span_tree(span_data, tree)

    console.print(tree)

    # 汇总统计
    total_tokens = {"input": 0, "output": 0}
    span_count = 0

    def count_spans(spans: list[dict[str, Any]]) -> None:
        nonlocal span_count
        for s in spans:
            span_count += 1
            tokens = s.get("tokens", {})
            total_tokens["input"] += tokens.get("input", 0)
            total_tokens["output"] += tokens.get("output", 0)
            count_spans(s.get("children", []))

    count_spans(trace_data.get("spans", []))

    console.print(
        f"\n[bold]汇总：[/bold] {span_count} 个跨度，"
        f"{total_tokens['input']} 个输入令牌，"
        f"{total_tokens['output']} 个输出令牌"
    )


if __name__ == "__main__":
    main()
