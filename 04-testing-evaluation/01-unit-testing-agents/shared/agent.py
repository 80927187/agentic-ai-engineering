"""
通过依赖注入实现可测试性的工具调用代理。

封装代理的核心循环：向大语言模型发送消息、解析工具调用、执行工具并返回结果，
不断重复，直至收到文本响应或达到最大迭代次数。
"""

import json
from typing import Any

from common import AnthropicTokenTracker, setup_logging

from shared.tools import TOOLS, execute_tool

logger = setup_logging(__name__)


class ToolUseAgent:
    """支持依赖注入和迭代次数限制的工具调用代理。"""

    def __init__(
        self,
        client: Any,
        model: str = "claude-sonnet-4-5-20250929",
        max_iterations: int = 10,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.max_iterations = max_iterations
        self.tools = tools if tools is not None else TOOLS
        self.messages: list[dict[str, Any]] = []
        self.token_tracker = AnthropicTokenTracker()

    def send_message(self, user_message: str) -> str:
        """发送消息并执行代理循环，直至收到文本响应。"""
        self.messages.append({"role": "user", "content": user_message})
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            logger.info(
                "第 %d/%d 次迭代（消息数：%d）",
                iterations,
                self.max_iterations,
                len(self.messages),
            )

            # 核心概念：在此调用注入的客户端，因此很容易进行模拟
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                tools=self.tools,
                messages=self.messages,
            )

            self.token_tracker.track(response.usage)

            tool_uses = []
            text_content = []

            for block in response.content:
                if hasattr(block, "text"):
                    text_content.append(block.text)
                elif hasattr(block, "name") and hasattr(block, "input"):
                    tool_uses.append(block)

            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use" or not tool_uses:
                return "\n".join(text_content) if text_content else ""

            # 执行每个工具并返回结果
            tool_results = []
            for tool_use in tool_uses:
                result = execute_tool(tool_use.name, tool_use.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": json.dumps(result),
                    }
                )

            self.messages.append({"role": "user", "content": tool_results})

        # 契约：达到 max_iterations 后，代理必须停止
        logger.warning("已达到最大迭代次数（%d），正在停止代理", self.max_iterations)
        return "[代理已停止：达到最大迭代次数]"
