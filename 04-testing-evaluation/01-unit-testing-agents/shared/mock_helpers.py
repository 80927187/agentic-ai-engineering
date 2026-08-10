"""
用于测试 Anthropic API 交互的模拟响应工厂。

提供用于创建模拟 Anthropic API 响应、文本块和工具调用块的辅助函数，
且不依赖真实的 SDK 类型。
"""

from typing import Any
from unittest.mock import MagicMock, Mock


def create_mock_response(
    content: list[Any],
    stop_reason: str = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> MagicMock:
    """创建模拟的 Anthropic API 响应。"""
    response = MagicMock()
    response.content = content
    response.stop_reason = stop_reason
    # 模拟 usage 对象，使其结构与 Anthropic 保持一致
    response.usage = MagicMock()
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    response.usage.cache_read_input_tokens = None
    response.usage.cache_creation_input_tokens = None
    return response


def make_text_block(text: str) -> Mock:
    """创建模拟的 TextBlock。"""
    block = Mock()
    block.text = text
    # 确保它不会被识别为工具调用块
    block.name = None
    block.input = None
    del block.name
    del block.input
    return block


def make_tool_use_block(tool_id: str, name: str, tool_input: dict[str, Any]) -> Mock:
    """创建模拟的 ToolUseBlock。"""
    block = Mock()
    block.id = tool_id
    block.name = name
    block.input = tool_input
    # 确保它不会被识别为文本块
    block.text = None
    del block.text
    return block
