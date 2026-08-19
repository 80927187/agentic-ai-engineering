"""
带追踪的共享研究助手智能体。

封装具有完整执行追踪的智能体循环，供需要实时智能体的教程（01、03）使用。
"""

import json
from typing import Any

import anthropic
from common import AnthropicTokenTracker, setup_logging

from shared.knowledge_base import SYSTEM_PROMPT, TOOLS, execute_tool
from shared.tracer import TraceCollector

logger = setup_logging(__name__)

MODEL = "deepseek-v4-flash"


class TracedResearchAssistant:
    """具有完整执行追踪功能的研究助手。"""

    def __init__(
        self,
        client: anthropic.Anthropic,
        tracer: TraceCollector,
    ) -> None:
        self.client = client
        self.tracer = tracer
        self.token_tracker = AnthropicTokenTracker()

    def answer(self, question: str) -> dict[str, Any]:
        """在完整追踪下回答问题。"""
        with self.tracer.span("answer_question", "agent_step", {"question": question}) as root:
            messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
            llm_call_count = 0

            while True:
                llm_call_count += 1
                with self.tracer.span(
                    f"llm_call_{llm_call_count}",
                    "llm_call",
                    {"message_count": len(messages)},
                ) as llm_span:
                    response = self.client.messages.create(
                        model=MODEL,
                        max_tokens=21333,
                        system=SYSTEM_PROMPT,
                        tools=TOOLS,
                        messages=messages,
                    )
                    self.token_tracker.track(response.usage)
                    llm_span.tokens = {
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                    }
                    llm_span.outputs = {"stop_reason": response.stop_reason}

                # 处理响应
                tool_uses = []
                text_parts: list[str] = []
                for block in response.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_uses.append(block)

                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason != "tool_use" or not tool_uses:
                    if not text_parts:
                        block_types = [block.type for block in response.content]
                        raise ValueError(
                            f"模型响应中没有文本内容（stop_reason={response.stop_reason}，"
                            f"内容块={block_types}，output_tokens={response.usage.output_tokens}）。"
                        )
                    answer_text = "\n\n".join(text_parts)
                    root.outputs = {"answer": answer_text[:200], "llm_calls": llm_call_count}
                    return {
                        "answer": answer_text,
                        "llm_calls": llm_call_count,
                        "trace": self.tracer.to_dict(),
                    }

                # 在追踪下执行工具
                tool_results = []
                for tool_use in tool_uses:
                    with self.tracer.span(
                        f"tool_{tool_use.name}",
                        "tool_call",
                        {"tool": tool_use.name, "input": tool_use.input},
                    ) as tool_span:
                        result = execute_tool(tool_use.name, tool_use.input)
                        tool_span.outputs = {"result": str(result)[:200]}

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": json.dumps(result, default=str),
                        }
                    )

                messages.append({"role": "user", "content": tool_results})

                if llm_call_count >= 100:
                    root.error = "已达到最大迭代次数"
                    return {"answer": "已达到最大迭代次数", "llm_calls": llm_call_count}
