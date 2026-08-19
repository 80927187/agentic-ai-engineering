"""
追踪调试

演示基于追踪的调试工作流：给定一次失败的智能体执行，逐步检查已记录的追踪，
找出故障点、提取决策路径、提出修复建议，并从检查点重放。

核心概念：
- 故障点检测：遍历跨度树，找到第一个错误或异常输出
- 决策路径提取：重建智能体所做的一系列选择
- 修复建议：将故障类型映射到可执行的修复步骤
- 追踪重放：列出检查点，并模拟从选定位置重新执行
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
from rich.tree import Tree

from shared.tracer import collect_all_spans

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

MODEL = "deepseek-v4-flash"


# ---------------------------------------------------------------------------
# 失败追踪示例
# ---------------------------------------------------------------------------

# 故障 1：智能体使用了错误的搜索词，未找到结果
TRACE_WRONG_SEARCH = {
    "trace_id": "fail_wrong_search",
    "question": "Kubernetes 如何处理自动扩缩容？",
    "expected_answer_contains": "auto-scaling",
    "spans": [
        {
            "name": "answer_question",
            "span_type": "agent_step",
            "start_time": 1000.0,
            "end_time": 1005.0,
            "duration_ms": 5000.0,
            "inputs": {"question": "Kubernetes 如何处理自动扩缩容？"},
            "outputs": {"answer": "我找不到相关信息。"},
            "tokens": {},
            "error": None,
            "children": [
                {
                    "name": "llm_call_1",
                    "span_type": "llm_call",
                    "start_time": 1000.1,
                    "end_time": 1001.5,
                    "duration_ms": 1400.0,
                    "inputs": {"message_count": 1},
                    "outputs": {"stop_reason": "tool_use"},
                    "tokens": {"input": 160, "output": 70},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "tool_search_knowledge_base",
                    "span_type": "tool_call",
                    "start_time": 1001.5,
                    "end_time": 1001.52,
                    "duration_ms": 20.0,
                    "inputs": {
                        "tool": "search_knowledge_base",
                        "input": {"query": "horizontal pod autoscaler HPA"},
                    },
                    "outputs": {"result": "[]"},
                    "tokens": {},
                    "error": "查询过于具体，未找到结果",
                    "children": [],
                },
                {
                    "name": "llm_call_2",
                    "span_type": "llm_call",
                    "start_time": 1001.6,
                    "end_time": 1003.0,
                    "duration_ms": 1400.0,
                    "inputs": {"message_count": 3},
                    "outputs": {"stop_reason": "tool_use"},
                    "tokens": {"input": 280, "output": 65},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "tool_search_knowledge_base",
                    "span_type": "tool_call",
                    "start_time": 3001.0,
                    "end_time": 3001.02,
                    "duration_ms": 20.0,
                    "inputs": {
                        "tool": "search_knowledge_base",
                        "input": {"query": "HPA metrics CPU"},
                    },
                    "outputs": {"result": "[]"},
                    "tokens": {},
                    "error": "未找到结果——对于该知识库而言，查询过于具体",
                    "children": [],
                },
                {
                    "name": "llm_call_3",
                    "span_type": "llm_call",
                    "start_time": 3001.1,
                    "end_time": 3004.8,
                    "duration_ms": 3700.0,
                    "inputs": {"message_count": 5},
                    "outputs": {"stop_reason": "end_turn"},
                    "tokens": {"input": 380, "output": 100},
                    "error": None,
                    "children": [],
                },
            ],
        }
    ],
}

# 故障 2：智能体找到了结果，但捏造了文档中不存在的信息
TRACE_HALLUCINATION = {
    "trace_id": "fail_hallucination",
    "question": "有哪些可用的缓存策略？",
    "expected_answer_contains": "cache-aside",
    "spans": [
        {
            "name": "answer_question",
            "span_type": "agent_step",
            "start_time": 2000.0,
            "end_time": 2004.0,
            "duration_ms": 4000.0,
            "inputs": {"question": "有哪些可用的缓存策略？"},
            "outputs": {
                "answer": (
                    "主要缓存策略包括旁路缓存、写穿、写回，以及采用一致性哈希的分布式缓存。"
                    "对于静态资源，还应考虑使用 Cloudflare 进行 CDN 级缓存。"
                ),
            },
            "tokens": {},
            "error": "检测到幻觉",
            "children": [
                {
                    "name": "llm_call_1",
                    "span_type": "llm_call",
                    "start_time": 2000.1,
                    "end_time": 2001.3,
                    "duration_ms": 1200.0,
                    "inputs": {"message_count": 1},
                    "outputs": {"stop_reason": "tool_use"},
                    "tokens": {"input": 150, "output": 60},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "tool_search_knowledge_base",
                    "span_type": "tool_call",
                    "start_time": 2001.3,
                    "end_time": 2001.32,
                    "duration_ms": 20.0,
                    "inputs": {
                        "tool": "search_knowledge_base",
                        "input": {"query": "缓存策略"},
                    },
                    "outputs": {"result": "[{'id': 'doc_008', 'title': '缓存策略'}]"},
                    "tokens": {},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "llm_call_2",
                    "span_type": "llm_call",
                    "start_time": 2001.4,
                    "end_time": 2002.8,
                    "duration_ms": 1400.0,
                    "inputs": {"message_count": 3},
                    "outputs": {"stop_reason": "tool_use"},
                    "tokens": {"input": 300, "output": 50},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "tool_get_document",
                    "span_type": "tool_call",
                    "start_time": 2002.8,
                    "end_time": 2002.81,
                    "duration_ms": 10.0,
                    "inputs": {"tool": "get_document", "input": {"doc_id": "doc_008"}},
                    "outputs": {
                        "result": (
                            "缓存可以降低延迟……策略包括旁路缓存、写穿和写回。"
                            "可以使用 Redis 或 Memcached。"
                        ),
                    },
                    "tokens": {},
                    "error": None,
                    "children": [],
                },
                {
                    "name": "llm_call_3",
                    "span_type": "llm_call",
                    "start_time": 2002.9,
                    "end_time": 2003.9,
                    "duration_ms": 1000.0,
                    "inputs": {"message_count": 5},
                    "outputs": {
                        "stop_reason": "end_turn",
                        "answer_includes_hallucination": True,
                        "hallucinated_claims": [
                            "采用一致性哈希的分布式缓存",
                            "使用 Cloudflare 的 CDN 级缓存",
                        ],
                    },
                    "tokens": {"input": 500, "output": 150},
                    "error": "LLM 添加了检索文档中不存在的说法",
                    "children": [],
                },
            ],
        }
    ],
}

# 故障 3：智能体因重复调用而陷入循环
TRACE_LOOP = {
    "trace_id": "fail_loop",
    "question": "比较微服务架构和事件驱动架构",
    "expected_answer_contains": "微服务",
    "spans": [
        {
            "name": "answer_question",
            "span_type": "agent_step",
            "start_time": 3000.0,
            "end_time": 3020.0,
            "duration_ms": 20000.0,
            "inputs": {"question": "比较微服务架构和事件驱动架构"},
            "outputs": {"answer": "已达到最大迭代次数"},
            "tokens": {},
            "error": "已达到最大迭代次数",
            "children": [
                {
                    "name": f"llm_call_{i}",
                    "span_type": "llm_call",
                    "start_time": 3000.0 + i * 2,
                    "end_time": 3001.5 + i * 2,
                    "duration_ms": 1500.0,
                    "inputs": {"message_count": 1 + i * 2},
                    "outputs": {"stop_reason": "tool_use"},
                    "tokens": {"input": 200 + i * 100, "output": 60},
                    "error": None,
                    "children": [],
                }
                for i in range(8)
            ]
            + [
                {
                    "name": f"tool_search_knowledge_base_{i}",
                    "span_type": "tool_call",
                    "start_time": 3001.5 + i * 2,
                    "end_time": 3001.52 + i * 2,
                    "duration_ms": 20.0,
                    "inputs": {
                        "tool": "search_knowledge_base",
                        "input": {"query": "微服务" if i % 2 == 0 else "事件驱动"},
                    },
                    "outputs": {
                        "result": ("[{'id': 'doc_001'}]" if i % 2 == 0 else "[{'id': 'doc_007'}]"),
                    },
                    "tokens": {},
                    "error": None,
                    "children": [],
                }
                for i in range(8)
            ],
        }
    ],
}

ALL_FAILING_TRACES = {
    "wrong_search": TRACE_WRONG_SEARCH,
    "hallucination": TRACE_HALLUCINATION,
    "loop": TRACE_LOOP,
}

TRACE_NAME_LABELS = {
    "wrong_search": "错误搜索",
    "hallucination": "幻觉",
    "loop": "循环",
}


# ---------------------------------------------------------------------------
# 调试工具
# ---------------------------------------------------------------------------


class TraceDebugger:
    """使用执行追踪调试智能体故障。"""

    def find_failure_point(self, trace: dict[str, Any]) -> dict[str, Any] | None:
        """遍历追踪，找到第一个发生错误的跨度。"""
        all_spans = collect_all_spans(trace.get("spans", []))
        for span in all_spans:
            if span.get("error"):
                return {
                    "span_name": span["name"],
                    "span_type": span.get("span_type", "未知"),
                    "error": span["error"],
                    "inputs": span.get("inputs", {}),
                    "outputs": span.get("outputs", {}),
                    "duration_ms": span.get("duration_ms", 0),
                }
        return None

    def get_decision_path(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        """提取智能体所做的一系列决策。"""
        all_spans = collect_all_spans(trace.get("spans", []))
        decisions: list[dict[str, Any]] = []

        for span in all_spans:
            span_type = span.get("span_type", "")
            if span_type == "agent_step":
                continue  # 跳过根包装跨度

            decision: dict[str, Any] = {
                "step": len(decisions) + 1,
                "name": span["name"],
                "type": span_type,
                "duration_ms": span.get("duration_ms", 0),
            }

            if span_type == "llm_call":
                decision["action"] = "LLM 决策"
                decision["outcome"] = span.get("outputs", {}).get("stop_reason", "未知")
            elif span_type == "tool_call":
                tool_name = span.get("inputs", {}).get("tool", "未知")
                tool_input = span.get("inputs", {}).get("input", {})
                decision["action"] = f"调用了 {tool_name}"
                decision["detail"] = json.dumps(tool_input, ensure_ascii=False)
                decision["outcome"] = "错误" if span.get("error") else "成功"

            if span.get("error"):
                decision["error"] = span["error"]

            decisions.append(decision)

        return decisions

    def suggest_fixes(self, failure: dict[str, Any]) -> list[str]:
        """根据故障类型提出可能的修复建议。"""
        suggestions: list[str] = []
        error = failure.get("error", "")
        span_type = failure.get("span_type", "")

        # 搜索错误或无结果
        if "未找到结果" in error or "未找到" in error:
            suggestions.append("扩大搜索范围——使用更少、更通用的词语")
            suggestions.append("添加回退逻辑：结果为空时使用更简单的关键词重试")
            suggestions.append("扩充知识库以覆盖更多主题")

        # 幻觉
        if "幻觉" in error or "不存在" in error:
            suggestions.append(
                "添加明确的依据约束：‘只能使用检索到的文档中的信息’"
            )
            suggestions.append(
                "实现生成后检查，对照源文档验证各项说法"
            )
            suggestions.append("降低 temperature，减少创造性生成")

        # 循环或达到最大迭代次数
        if "最大迭代次数" in error or "循环" in error:
            suggestions.append("添加已查询集合，防止重复执行相同搜索")
            suggestions.append("降低 max_iterations，并添加‘汇总已有信息’的回退逻辑")
            suggestions.append(
                "改进系统提示词，要求智能体在搜索 2～3 次后综合已有信息"
            )

        # 缓慢跨度
        if span_type == "llm_call" and failure.get("duration_ms", 0) > 10000:
            suggestions.append("检查提示词是否过长——汇总之前的上下文")
            suggestions.append("考虑在中间步骤使用速度更快的模型")

        # 工具执行错误
        if span_type == "tool_call" and error:
            suggestions.append("为临时错误添加指数退避重试逻辑")
            suggestions.append("执行前验证工具输入")

        # 通用建议
        if not suggestions:
            suggestions.append("检查完整决策路径，以理解智能体的推理过程")
            suggestions.append("在失败跨度周围添加更详细的日志")

        return suggestions


class TraceReplay:
    """从已记录追踪中的检查点重放智能体执行过程。"""

    def list_checkpoints(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        """列出追踪中可用的检查点（决策点）。"""
        all_spans = collect_all_spans(trace.get("spans", []))
        checkpoints: list[dict[str, Any]] = []

        for i, span in enumerate(all_spans):
            if span.get("span_type") in ("llm_call", "tool_call"):
                checkpoints.append(
                    {
                        "index": len(checkpoints),
                        "span_index": i,
                        "name": span["name"],
                        "type": span.get("span_type"),
                        "inputs": span.get("inputs", {}),
                        "had_error": bool(span.get("error")),
                    }
                )

        return checkpoints

    def replay_from(
        self,
        trace: dict[str, Any],
        checkpoint_index: int,
        client: anthropic.Anthropic | None = None,
    ) -> dict[str, Any]:
        """从指定检查点重放，也可以选择使用实时 LLM。"""
        checkpoints = self.list_checkpoints(trace)

        if checkpoint_index < 0 or checkpoint_index >= len(checkpoints):
            return {"error": f"无效的检查点索引：{checkpoint_index}"}

        checkpoint = checkpoints[checkpoint_index]
        preceding = checkpoints[:checkpoint_index]

        # 根据之前的步骤构建上下文
        context: list[dict[str, Any]] = []
        for cp in preceding:
            context.append(
                {
                    "step": cp["name"],
                    "type": cp["type"],
                    "inputs": cp["inputs"],
                }
            )

        result: dict[str, Any] = {
            "checkpoint": checkpoint,
            "preceding_steps": len(preceding),
            "context_summary": context,
        }

        if client is not None:
            # 实时重放：使用截至检查点的上下文重新运行 LLM 调用
            logger.info("从检查点 %d 实时重放：%s", checkpoint_index, checkpoint["name"])
            question = trace.get("question", "")

            system_prompt = (
                "你是一名研究助手。上一次智能体执行失败了。现在你正从某个检查点重放。"
                "请使用所提供的上下文回答原始问题，回答应简洁并以事实为依据。"
            )

            context_text = f"原始问题：{question}\n\n"
            context_text += "失败前的执行上下文：\n"
            for step in context:
                context_text += f"- {step['step']}：{json.dumps(step['inputs'], ensure_ascii=False)}\n"
            context_text += f"\n失败位置：{checkpoint['name']}\n"
            context_text += "请提供修正后的回答。"

            token_tracker = AnthropicTokenTracker()
            start = time.time()
            response = client.messages.create(
                model=MODEL,
                max_tokens=21333,
                system=system_prompt,
                messages=[{"role": "user", "content": context_text}],
            )
            elapsed = (time.time() - start) * 1000
            token_tracker.track(response.usage)

            text_parts = [block.text for block in response.content if block.type == "text"]
            if not text_parts:
                block_types = [block.type for block in response.content]
                raise ValueError(
                    f"模型响应中没有文本内容（stop_reason={response.stop_reason}，"
                    f"内容块={block_types}，output_tokens={response.usage.output_tokens}）。"
                )
            answer = "\n\n".join(text_parts)

            result["replayed_answer"] = answer
            result["replay_duration_ms"] = round(elapsed, 2)
            result["replay_tokens"] = {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            }
        else:
            result["replayed_answer"] = (
                f"[试运行] 将从“{checkpoint['name']}”重新执行，"
                f"并使用之前的 {len(preceding)} 个步骤作为上下文"
            )

        return result


# ---------------------------------------------------------------------------
# 可视化辅助函数
# ---------------------------------------------------------------------------


def _build_decision_tree(decisions: list[dict[str, Any]], tree: Tree) -> None:
    """将决策步骤添加到 Rich 树。"""
    for d in decisions:
        label = f"[bold]步骤 {d['step']}：[/bold] {d['name']}"
        if d.get("action"):
            label += f" — {d['action']}"
        if d.get("outcome"):
            style = "red" if d["outcome"] == "错误" else "green"
            label += f" [{style}]({d['outcome']})[/{style}]"
        if d.get("error"):
            label += f"\n  [red]错误：{d['error']}[/red]"
        if d.get("detail"):
            label += f"\n  [dim]{d['detail']}[/dim]"
        tree.add(label)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """演示基于追踪的调试工作流。"""
    console = Console()

    console.print(
        Panel(
            "[bold cyan]追踪调试[/bold cyan]\n\n"
            "针对失败的智能体执行，逐步检查追踪以找出故障点、\n"
            "提取决策路径、提出修复建议并列出重放检查点。\n\n"
            "概念：故障检测、决策路径、修复建议、追踪重放",
            title="03 - 追踪调试",
        )
    )

    debugger = TraceDebugger()
    replayer = TraceReplay()

    for trace_name, trace in ALL_FAILING_TRACES.items():
        console.print(f"\n{'=' * 80}")
        console.print(
            f"\n[bold magenta]正在调试追踪：{TRACE_NAME_LABELS[trace_name]}[/bold magenta]"
            f"\n[dim]问题：{trace.get('question', '不适用')}[/dim]\n"
        )

        # 步骤 1：查找故障点
        failure = debugger.find_failure_point(trace)
        if failure:
            console.print(
                Panel(
                    f"[bold red]故障点[/bold red]\n\n"
                    f"跨度：{failure['span_name']}（{failure['span_type']}）\n"
                    f"错误：{failure['error']}\n"
                    f"耗时：{failure['duration_ms']:.0f} 毫秒",
                    title="检测到故障",
                    border_style="red",
                )
            )
        else:
            console.print("[green]追踪中未找到明确故障[/green]")

        # 步骤 2：显示决策路径
        decisions = debugger.get_decision_path(trace)
        decision_tree = Tree(f"[bold]决策路径（{len(decisions)} 个步骤）[/bold]")
        _build_decision_tree(decisions, decision_tree)
        console.print(decision_tree)

        # 步骤 3：提出修复建议
        if failure:
            suggestions = debugger.suggest_fixes(failure)
            console.print("\n[bold yellow]修复建议：[/bold yellow]")
            for i, suggestion in enumerate(suggestions, 1):
                console.print(f"  {i}. {suggestion}")

        # 步骤 4：列出重放检查点
        checkpoints = replayer.list_checkpoints(trace)
        if checkpoints:
            cp_table = Table(title="重放检查点")
            cp_table.add_column("索引", justify="center")
            cp_table.add_column("名称")
            cp_table.add_column("类型")
            cp_table.add_column("是否出错", justify="center")
            for cp in checkpoints:
                error_marker = "[red]是[/red]" if cp["had_error"] else "[green]否[/green]"
                cp_table.add_row(
                    str(cp["index"]),
                    cp["name"],
                    cp["type"],
                    error_marker,
                )
            console.print()
            console.print(cp_table)

        # 步骤 5：从第一个出错的检查点开始试运行重放
        errored_cps = [cp for cp in checkpoints if cp["had_error"]]
        if errored_cps:
            first_error_cp = errored_cps[0]["index"]
            console.print(f"\n[bold]从检查点 {first_error_cp} 开始试运行重放：[/bold]")

            has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
            client = anthropic.Anthropic() if has_api_key else None

            replay_result = replayer.replay_from(trace, first_error_cp, client=client)
            console.print(f"  [dim]{replay_result['replayed_answer']}[/dim]")

            if replay_result.get("replay_tokens"):
                tokens = replay_result["replay_tokens"]
                console.print(
                    f"  [dim]重放令牌：{tokens['input']} 入 / {tokens['output']} 出，"
                    f"耗时：{replay_result.get('replay_duration_ms', 0):.0f} 毫秒[/dim]"
                )

    console.print(f"\n{'=' * 80}")
    console.print("\n[bold green]调试工作流已完成。[/bold green]")


if __name__ == "__main__":
    main()
