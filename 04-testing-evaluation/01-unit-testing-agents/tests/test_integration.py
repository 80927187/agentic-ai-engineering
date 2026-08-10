"""
使用响应录制文件进行集成测试

使用已记录的 API 响应而不是模拟对象来测试完整的代理循环。
通过重放录制文件中的响应，可以进行确定性测试并覆盖真实的响应解析路径，
无需维护 MagicMock 的数据结构。

核心测试概念：
- 完整循环测试：使用已记录的响应测试代理的多轮对话
- 快照回归：将代理输出与黄金基准比较，发现偏移
- 录制响应耗尽：自动发现代理行为偏离
"""

import json
from pathlib import Path

import pytest

from shared.agent import ToolUseAgent
from tests.conftest import (
    CASSETTE_BLOCKED_COMMAND,
    CASSETTE_CALCULATOR,
    CASSETTE_MULTI_TOOL,
    CASSETTE_TEXT_ONLY,
    CassetteClient,
    CassetteResponse,
    serialize_response,
    write_cassette,
)


# ---------------------------------------------------------------------------
# 测试——通过重放录制响应测试完整代理循环
# ---------------------------------------------------------------------------


class TestCassetteReplay:
    """使用预先录制的 API 响应进行集成测试。"""

    def test_text_only_response(self, cassette_dir: Path) -> None:
        """不需要工具时，代理直接返回文本。"""
        path = write_cassette(cassette_dir, "text_only", CASSETTE_TEXT_ONLY)
        client = CassetteClient(path)
        agent = ToolUseAgent(client=client)

        result = agent.send_message("你好")

        assert result == "你好！我可以帮你进行计算。"
        assert client.calls_remaining == 0

    def test_single_tool_call(self, cassette_dir: Path) -> None:
        """代理执行计算器工具并返回最终答案。"""
        path = write_cassette(cassette_dir, "calculator", CASSETTE_CALCULATOR)
        client = CassetteClient(path)
        agent = ToolUseAgent(client=client)

        result = agent.send_message("12 * 15 等于多少？")

        assert "180" in result
        assert client.calls_remaining == 0
        # 验证工具确实已执行——结果应出现在消息历史中
        tool_result_msg = agent.messages[2]
        tool_result_data = json.loads(tool_result_msg["content"][0]["content"])
        assert tool_result_data["result"] == 180

    def test_multi_turn_tool_use(self, cassette_dir: Path) -> None:
        """代理能够跨轮次处理多个连续的工具调用。"""
        path = write_cassette(cassette_dir, "multi_tool", CASSETTE_MULTI_TOOL)
        client = CassetteClient(path)
        agent = ToolUseAgent(client=client)

        result = agent.send_message("先计算 100 + 200，再乘以 2")

        assert "600" in result
        assert client.calls_remaining == 0
        # 6 条消息：用户、助手（工具）、用户（结果）、助手（工具）、用户（结果）、助手
        assert len(agent.messages) == 6

    def test_blocked_command_integration(self, cassette_dir: Path) -> None:
        """完整集成流程：模型请求危险命令，代理阻止该命令，模型恢复响应。"""
        path = write_cassette(cassette_dir, "blocked", CASSETTE_BLOCKED_COMMAND)
        client = CassetteClient(path)
        agent = ToolUseAgent(client=client)

        result = agent.send_message("删除临时数据")

        assert "阻止" in result or "安全" in result
        # 验证工具结果包含阻止执行的错误
        tool_result_msg = agent.messages[2]
        tool_result_data = json.loads(tool_result_msg["content"][0]["content"])
        assert "error" in tool_result_data
        assert "阻止" in tool_result_data["error"]


class TestCassetteExhaustion:
    """验证响应录制系统能够发现代理行为偏离。"""

    def test_cassette_exhausted_raises_error(self, cassette_dir: Path) -> None:
        """如果代理的 API 调用次数超过记录数量，录制系统将引发错误。"""
        # 使用只有 1 条响应的纯文本录制数据，但让客户端发起 2 次调用
        path = write_cassette(cassette_dir, "short", CASSETTE_TEXT_ONLY)
        client = CassetteClient(path)

        # 第一次调用成功
        response = client.create(model="test", max_tokens=100, tools=[], messages=[])
        assert response.stop_reason == "end_turn"

        # 第二次调用应失败——录制响应已用尽
        with pytest.raises(RuntimeError, match="录制响应已用尽"):
            client.create(model="test", max_tokens=100, tools=[], messages=[])


# ---------------------------------------------------------------------------
# 测试——快照回归测试
# ---------------------------------------------------------------------------


class TestSnapshotRegression:
    """将代理输出与黄金快照比较，以发现回归。"""

    def test_calculator_output_matches_snapshot(self, cassette_dir: Path) -> None:
        """已知输入的代理输出必须与记录的黄金快照一致。"""
        path = write_cassette(cassette_dir, "calculator", CASSETTE_CALCULATOR)

        # 黄金快照——“12 * 15 等于多少？”的预期输出
        golden_snapshot = "12 乘以 15 等于 180。"

        client = CassetteClient(path)
        agent = ToolUseAgent(client=client)
        result = agent.send_message("12 * 15 等于多少？")

        assert result == golden_snapshot, (
            f"输出已偏离快照。\n  预期：{golden_snapshot!r}\n  实际：{result!r}"
        )

    def test_message_history_shape_matches_snapshot(self, cassette_dir: Path) -> None:
        """消息历史的结构必须符合预期模式。"""
        path = write_cassette(cassette_dir, "calculator", CASSETTE_CALCULATOR)
        client = CassetteClient(path)
        agent = ToolUseAgent(client=client)
        agent.send_message("12 * 15 等于多少？")

        # 按顺序记录预期消息角色的快照
        expected_roles = ["user", "assistant", "user", "assistant"]
        actual_roles = [msg["role"] for msg in agent.messages]

        assert actual_roles == expected_roles, (
            f"消息历史结构已发生变化。\n  预期：{expected_roles}\n  实际：{actual_roles}"
        )

    def test_token_usage_within_budget(self, cassette_dir: Path) -> None:
        """令牌总用量必须保持在预期预算内。"""
        path = write_cassette(cassette_dir, "multi_tool", CASSETTE_MULTI_TOOL)
        client = CassetteClient(path)
        agent = ToolUseAgent(client=client)
        agent.send_message("先计算 100 + 200，再乘以 2")

        # 预算快照——令牌用量激增意味着某些行为发生了变化
        max_input_tokens = 1000
        max_output_tokens = 200

        assert agent.token_tracker.total_input_tokens <= max_input_tokens, (
            f"超出输入令牌预算：{agent.token_tracker.total_input_tokens} > {max_input_tokens}"
        )
        assert agent.token_tracker.total_output_tokens <= max_output_tokens, (
            f"超出输出令牌预算：{agent.token_tracker.total_output_tokens} > {max_output_tokens}"
        )


# ---------------------------------------------------------------------------
# 测试——序列化往返
# ---------------------------------------------------------------------------


class TestCassetteSerialization:
    """验证响应的序列化和反序列化不会丢失数据。"""

    def test_text_response_round_trip(self) -> None:
        """纯文本响应经过序列化和反序列化后不会丢失数据。"""
        original_data = CASSETTE_TEXT_ONLY[0]["response"]
        response = CassetteResponse(original_data)
        serialized = serialize_response(response)

        assert serialized["stop_reason"] == "end_turn"
        assert len(serialized["content"]) == 1
        assert serialized["content"][0]["text"] == "你好！我可以帮你进行计算。"
        assert serialized["usage"]["input_tokens"] == 120

    def test_tool_use_response_round_trip(self) -> None:
        """工具调用响应经过序列化和反序列化后不会丢失数据。"""
        original_data = CASSETTE_CALCULATOR[0]["response"]
        response = CassetteResponse(original_data)
        serialized = serialize_response(response)

        assert serialized["stop_reason"] == "tool_use"
        assert serialized["content"][0]["name"] == "calculator"
        assert serialized["content"][0]["input"]["operation"] == "multiply"
        assert serialized["content"][0]["id"] == "toolu_01ABC"
