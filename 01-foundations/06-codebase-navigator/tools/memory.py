"""
记忆工具

用于跨会话保存和回忆持久化记忆的工具。
"""

from typing import Any

from store.memory import MemoryStore

from common.logging_config import setup_logging

logger = setup_logging(__name__)

# Anthropic API 的工具定义
MEMORY_TOOLS = [
    {
        "name": "save_memory",
        "description": (
            "将信息保存到持久化记忆中，供未来会话使用。"
            "使用此工具记住架构见解、用户偏好或重要事实。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["facts", "insights", "preferences"],
                    "description": "类别：facts（事实）、insights（见解）或 preferences（偏好）",
                },
                "content": {
                    "type": "string",
                    "description": "要保存的信息",
                },
            },
            "required": ["category", "content"],
        },
    },
    {
        "name": "recall_memory",
        "description": "检索已保存的记忆，可以选择使用关键词查询进行筛选。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用于筛选记忆的可选关键词",
                },
            },
        },
    },
]


def execute_save_memory(memory: MemoryStore, tool_input: dict[str, Any]) -> str:
    """执行 save_memory 工具。"""
    return memory.save(
        category=tool_input["category"],
        content=tool_input["content"],
    )


def execute_recall_memory(memory: MemoryStore, tool_input: dict[str, Any]) -> str:
    """执行 recall_memory 工具。"""
    query = tool_input.get("query")
    memories = memory.recall(query)
    if not memories:
        return "未找到记忆。" + (f"（筛选条件：“{query}”）" if query else "")

    parts = []
    category_names = {
        "facts": "事实",
        "insights": "见解",
        "preferences": "偏好",
    }
    for category, entries in memories.items():
        parts.append(f"\n## {category_names.get(category, category)}")
        for entry in entries:
            parts.append(f"- {entry['content']} ({entry['created'][:10]})")
    return "\n".join(parts)
