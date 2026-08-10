"""
行为契约

定义并验证行为不变量，即无论大语言模型返回什么，代理都始终必须做或绝不能做的事情。

核心测试概念：
- 安全契约：即使大语言模型提出请求，也绝不执行被禁止的命令
- 终止保证：代理在 max_iterations 次迭代后停止，防止无限循环
- 历史记录不变量：工具执行后，其结果始终出现在消息历史中
- 健壮性：代理能够妥善处理空响应和格式错误的工具输入
"""

import json
from unittest.mock import MagicMock

from shared.agent import ToolUseAgent
from shared.mock_helpers import create_mock_response, make_text_block, make_tool_use_block


# ---------------------------------------------------------------------------
# 行为契约测试
# ---------------------------------------------------------------------------


class TestSafetyContracts:
    """契约：代理绝不能执行被禁止的命令。"""

    def setup_method(self) -> None:
        """为每个测试创建带有模拟客户端的新代理。"""
        self.mock_client = MagicMock()
        self.agent = ToolUseAgent(client=self.mock_client, max_iterations=5)

    def test_agent_never_executes_blocked_commands(self) -> None:
        """即使大语言模型请求执行 rm -rf /，工具也会返回错误。"""
        # 大语言模型请求执行危险命令
        tool_block = make_tool_use_block("call_1", "run_bash", {"command": "rm -rf /"})
        tool_response = create_mock_response([tool_block], stop_reason="tool_use")

        # 看到错误后，大语言模型返回文本响应
        text_block = make_text_block("我无法执行该命令。")
        text_response = create_mock_response([text_block], stop_reason="end_turn")

        self.mock_client.messages.create.side_effect = [tool_response, text_response]

        self.agent.send_message("删除所有内容")

        # 验证工具结果包含阻止执行的错误
        tool_result_msg = self.agent.messages[2]
        tool_result_data = json.loads(tool_result_msg["content"][0]["content"])
        assert "error" in tool_result_data
        assert "阻止" in tool_result_data["error"]

    def test_blocked_sudo_command(self) -> None:
        """验证 sudo 命令会被阻止。"""
        tool_block = make_tool_use_block("call_1", "run_bash", {"command": "sudo apt install foo"})
        tool_response = create_mock_response([tool_block], stop_reason="tool_use")
        text_block = make_text_block("无法运行 sudo。")
        text_response = create_mock_response([text_block], stop_reason="end_turn")

        self.mock_client.messages.create.side_effect = [tool_response, text_response]

        self.agent.send_message("安装一个软件包")

        tool_result_msg = self.agent.messages[2]
        tool_result_data = json.loads(tool_result_msg["content"][0]["content"])
        assert "error" in tool_result_data
        assert "sudo" in tool_result_data["error"]


class TestTerminationContracts:
    """契约：代理始终必须在 max_iterations 次迭代内终止。"""

    def setup_method(self) -> None:
        """创建迭代次数上限较低的代理，以便测试。"""
        self.mock_client = MagicMock()
        self.agent = ToolUseAgent(client=self.mock_client, max_iterations=3)

    def test_agent_stops_after_max_iterations(self) -> None:
        """如果大语言模型不断请求工具，代理会在达到 max_iterations 时停止。"""
        # 大语言模型始终请求工具，从不提供最终答案
        tool_block = make_tool_use_block(
            "call_n",
            "calculator",
            {
                "operation": "add",
                "a": 1,
                "b": 1,
            },
        )
        infinite_response = create_mock_response([tool_block], stop_reason="tool_use")
        self.mock_client.messages.create.return_value = infinite_response

        result = self.agent.send_message("永远计算下去")

        # 代理必须停止并返回安全提示
        assert "达到最大迭代次数" in result
        # API 调用次数恰好为 max_iterations
        assert self.mock_client.messages.create.call_count == 3


class TestHistoryContracts:
    """契约：工具结果必须始终出现在消息历史中。"""

    def setup_method(self) -> None:
        """创建带有模拟客户端的新代理。"""
        self.mock_client = MagicMock()
        self.agent = ToolUseAgent(client=self.mock_client)

    def test_agent_always_includes_tool_results(self) -> None:
        """工具执行后，结果必须出现在对话历史中。"""
        tool_block = make_tool_use_block(
            "call_1",
            "calculator",
            {
                "operation": "multiply",
                "a": 3,
                "b": 9,
            },
        )
        text_block = make_text_block("27")

        self.mock_client.messages.create.side_effect = [
            create_mock_response([tool_block], stop_reason="tool_use"),
            create_mock_response([text_block], stop_reason="end_turn"),
        ]

        self.agent.send_message("3 * 9?")

        # 查找所有 tool_result 消息
        tool_result_messages = [
            msg
            for msg in self.agent.messages
            if msg["role"] == "user"
            and isinstance(msg["content"], list)
            and any(item.get("type") == "tool_result" for item in msg["content"])
        ]
        assert len(tool_result_messages) == 1

        # 验证结果内容是有效的 JSON
        result_content = json.loads(tool_result_messages[0]["content"][0]["content"])
        assert result_content["result"] == 27

    def test_agent_preserves_conversation_history(self) -> None:
        """消息必须在代理循环中正确累积。"""
        text_block = make_text_block("你好！")
        self.mock_client.messages.create.return_value = create_mock_response(
            [text_block], stop_reason="end_turn"
        )

        self.agent.send_message("你好")

        # 简单交互后：一条用户消息和一条助手响应
        assert len(self.agent.messages) == 2
        assert self.agent.messages[0]["role"] == "user"
        assert self.agent.messages[0]["content"] == "你好"
        assert self.agent.messages[1]["role"] == "assistant"


class TestRobustnessContracts:
    """契约：代理必须妥善处理边界情况。"""

    def setup_method(self) -> None:
        """创建带有模拟客户端的新代理。"""
        self.mock_client = MagicMock()
        self.agent = ToolUseAgent(client=self.mock_client)

    def test_agent_handles_empty_response(self) -> None:
        """大语言模型返回空内容时，应返回空字符串而不是崩溃。"""
        response = create_mock_response([], stop_reason="end_turn")
        self.mock_client.messages.create.return_value = response

        result = self.agent.send_message("什么都不要说")

        assert result == ""

    def test_agent_handles_malformed_tool_input(self) -> None:
        """如果大语言模型发送错误参数，应妥善捕获工具错误。"""
        # 大语言模型为计算器发送错误的键
        tool_block = make_tool_use_block(
            "call_bad",
            "calculator",
            {
                "wrong_key": "not_a_number",
            },
        )
        tool_response = create_mock_response([tool_block], stop_reason="tool_use")

        text_block = make_text_block("抱歉，操作失败了。")
        text_response = create_mock_response([text_block], stop_reason="end_turn")

        self.mock_client.messages.create.side_effect = [tool_response, text_response]

        self.agent.send_message("执行错误操作")

        # 代理不应崩溃，而应在工具结果中捕获错误
        tool_result_msg = self.agent.messages[2]
        tool_result_data = json.loads(tool_result_msg["content"][0]["content"])
        assert "error" in tool_result_data

    def test_tool_results_format_is_consistent(self) -> None:
        """所有工具结果必须具有相同结构：type、tool_use_id、content。"""
        # 连续执行两个工具
        tool_block_1 = make_tool_use_block(
            "call_1",
            "calculator",
            {
                "operation": "add",
                "a": 1,
                "b": 2,
            },
        )
        tool_block_2 = make_tool_use_block(
            "call_2",
            "calculator",
            {
                "operation": "multiply",
                "a": 3,
                "b": 4,
            },
        )
        tool_response = create_mock_response([tool_block_1, tool_block_2], stop_reason="tool_use")

        text_block = make_text_block("完成")
        text_response = create_mock_response([text_block], stop_reason="end_turn")

        self.mock_client.messages.create.side_effect = [tool_response, text_response]

        self.agent.send_message("计算两个表达式")

        # 查找 tool_result 消息
        tool_result_msg = self.agent.messages[2]
        assert tool_result_msg["role"] == "user"

        # 验证每个工具结果都包含必需的键
        for item in tool_result_msg["content"]:
            assert "type" in item
            assert item["type"] == "tool_result"
            assert "tool_use_id" in item
            assert "content" in item
            # content 必须是有效的 JSON
            parsed = json.loads(item["content"])
            assert isinstance(parsed, dict)
