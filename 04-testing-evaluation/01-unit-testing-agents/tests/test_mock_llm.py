"""
模拟大语言模型测试

在不发起真实 API 调用的情况下测试代理的工具调用循环。
使用 unittest.mock 模拟 Anthropic 响应，验证代理能否正确解析工具调用、
执行工具、返回结果并处理错误。
"""

import json
from unittest.mock import MagicMock

import pytest

from shared.agent import ToolUseAgent
from shared.mock_helpers import create_mock_response, make_text_block, make_tool_use_block


class TestToolUseAgent:
    """使用模拟大语言模型响应测试 ToolUseAgent。"""

    def setup_method(self) -> None:
        """为每个测试创建带有模拟客户端的新代理。"""
        self.mock_client = MagicMock()
        self.agent = ToolUseAgent(client=self.mock_client)

    def test_agent_calls_calculator_tool(self) -> None:
        """测试代理收到模拟的工具调用响应后执行计算器。"""
        # 第一次响应：大语言模型要求使用计算器
        tool_block = make_tool_use_block(
            "call_1",
            "calculator",
            {
                "operation": "multiply",
                "a": 6,
                "b": 7,
            },
        )
        tool_response = create_mock_response([tool_block], stop_reason="tool_use")

        # 第二次响应：大语言模型返回最终文本
        text_block = make_text_block("结果是 42。")
        text_response = create_mock_response([text_block], stop_reason="end_turn")

        self.mock_client.messages.create.side_effect = [tool_response, text_response]

        result = self.agent.send_message("6 * 7 等于多少？")

        assert result == "结果是 42。"
        assert self.mock_client.messages.create.call_count == 2

        # 验证工具结果已返回给大语言模型
        tool_result_msg = self.agent.messages[2]  # 用户 -> 助手 -> 工具结果
        assert tool_result_msg["role"] == "user"
        tool_result_data = json.loads(tool_result_msg["content"][0]["content"])
        assert tool_result_data["result"] == 42

    def test_agent_handles_text_response(self) -> None:
        """验证未调用工具时，代理会直接返回文本。"""
        text_block = make_text_block("你好！有什么可以帮你？")
        response = create_mock_response([text_block], stop_reason="end_turn")
        self.mock_client.messages.create.return_value = response

        result = self.agent.send_message("你好")

        assert result == "你好！有什么可以帮你？"
        # 只调用一次 API，不进入工具循环
        assert self.mock_client.messages.create.call_count == 1

    def test_agent_handles_multi_turn_tool_use(self) -> None:
        """验证代理按工具调用 -> 结果 -> 文本的顺序循环。"""
        # 第 1 轮：大语言模型请求计算器
        tool_block = make_tool_use_block(
            "call_1",
            "calculator",
            {
                "operation": "add",
                "a": 10,
                "b": 20,
            },
        )
        # 第 2 轮：大语言模型返回最终答案
        text_block = make_text_block("10 + 20 = 30")

        self.mock_client.messages.create.side_effect = [
            create_mock_response([tool_block], stop_reason="tool_use"),
            create_mock_response([text_block], stop_reason="end_turn"),
        ]

        result = self.agent.send_message("计算 10 加 20")

        assert result == "10 + 20 = 30"
        # 消息：用户、助手（工具调用）、用户（工具结果）、助手（文本）
        assert len(self.agent.messages) == 4

    def test_agent_sends_tool_results_back(self) -> None:
        """验证消息历史中的工具结果格式正确。"""
        tool_block = make_tool_use_block(
            "call_abc",
            "calculator",
            {
                "operation": "divide",
                "a": 100,
                "b": 4,
            },
        )
        text_block = make_text_block("25")

        self.mock_client.messages.create.side_effect = [
            create_mock_response([tool_block], stop_reason="tool_use"),
            create_mock_response([text_block], stop_reason="end_turn"),
        ]

        self.agent.send_message("计算 100 除以 4")

        # 查找 tool_result 消息
        tool_result_msg = self.agent.messages[2]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "call_abc"

    def test_agent_tracks_tokens(self) -> None:
        """验证令牌跟踪器会累积多次 API 调用的用量。"""
        text_block = make_text_block("完成")
        response = create_mock_response(
            [text_block], stop_reason="end_turn", input_tokens=150, output_tokens=75
        )
        self.mock_client.messages.create.return_value = response

        self.agent.send_message("你好")

        assert self.agent.token_tracker.total_input_tokens == 150
        assert self.agent.token_tracker.total_output_tokens == 75

    def test_agent_handles_api_error(self) -> None:
        """验证代理会向上抛出 API 错误。"""
        self.mock_client.messages.create.side_effect = Exception("超出 API 速率限制")

        with pytest.raises(Exception, match="超出 API 速率限制"):
            self.agent.send_message("你好")
