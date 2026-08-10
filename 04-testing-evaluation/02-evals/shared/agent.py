"""评估教程使用的研究助手智能体。

实现一个能够搜索知识库、综合答案并引用来源的工具调用智能体，
作为评估流水线中的被测系统。
"""

import json
from typing import Any

import anthropic
from common import AnthropicTokenTracker, setup_logging

from shared.knowledge_base import KNOWLEDGE_BASE, SYSTEM_PROMPT, TOOLS, search_knowledge_base

logger = setup_logging(__name__)


class ResearchAssistant:
    """搜索知识库并综合答案的研究助手。"""

    def __init__(
        self,
        client: anthropic.Anthropic,
        knowledge_base: list[dict[str, Any]] | None = None,
        model: str = "claude-sonnet-4-5-20250929",
    ) -> None:
        self.client = client
        self.knowledge_base = knowledge_base if knowledge_base is not None else KNOWLEDGE_BASE
        self.model = model
        self.token_tracker = AnthropicTokenTracker()

    def answer(self, question: str) -> dict[str, Any]:
        """使用知识库回答问题。"""
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        tool_calls_made: list[dict[str, Any]] = []

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
            self.token_tracker.track(response.usage)

            if response.stop_reason != "tool_use":
                answer_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        answer_text += block.text
                return {
                    "answer": answer_text,
                    "tool_calls": tool_calls_made,
                    "sources": [tc["results"] for tc in tool_calls_made],
                }

            # 处理工具调用
            messages.append({"role": "assistant", "content": response.content})
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "tool_use":
                    result = search_knowledge_base(**block.input, corpus=self.knowledge_base)
                    tool_calls_made.append(
                        {"name": block.name, "input": block.input, "results": result}
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
