"""
工具模式、网络搜索配置和工具执行器。

使用 Anthropic 的 tool_choice 强制结构化 JSON 响应；服务端 web_search 工具负责实时研究。
"""

import json
from typing import Any

from common import setup_logging

logger = setup_logging(__name__)

# ─── Web Search ──────────────────────────────────────────────────────────────

# Anthropic 服务端网络搜索——由 Claude 决定何时搜索
# max_uses 控制每个阶段的搜索次数，以限制 token 成本
WEB_SEARCH_TOOL: dict = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 100,
}

# ─── Classification ──────────────────────────────────────────────────────────

CLASSIFY_TOOLS: list[dict] = [
    {
        "name": "classify_content",
        "description": "将内容请求分类为类型、主题和关键方面。",
        "input_schema": {
            "type": "object",
            "properties": {
                "content_type": {
                    "type": "string",
                    "enum": ["blog", "tutorial", "concept"],
                    "description": (
                        "blog：观点文章、经验分享、经验教训。"
                        "tutorial：分步指南、操作教程。"
                        "concept：对思想、模式和技术的深入解释。"
                    ),
                },
                "topic": {
                    "type": "string",
                    "description": "用几个词概括核心主题",
                },
                "key_aspects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要涵盖的 2-3 个具体角度",
                },
                "reasoning": {
                    "type": "string",
                    "description": "简要解释分类选择",
                },
            },
            "required": ["content_type", "topic", "key_aspects", "reasoning"],
        },
    }
]

# ─── Planning ────────────────────────────────────────────────────────────────

PLANNING_TOOLS: list[dict] = [
    {
        "name": "create_research_plan",
        "description": "将主题拆分为聚焦的研究子主题。",
        "input_schema": {
            "type": "object",
            "properties": {
                "subtopics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "research_prompt": {
                                "type": "string",
                                "description": "聚焦的研究问题",
                            },
                        },
                        "required": ["title", "research_prompt"],
                    },
                    "description": "并行研究的 2-3 个子主题",
                },
            },
            "required": ["subtopics"],
        },
    }
]

# ─── Evaluation ──────────────────────────────────────────────────────────────

EVALUATION_TOOLS: list[dict] = [
    {
        "name": "evaluate_draft",
        "description": "从多个质量维度评估草稿。",
        "input_schema": {
            "type": "object",
            "properties": {
                "clarity": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                },
                "technical_accuracy": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                },
                "structure": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                },
                "engagement": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                },
                "human_voice": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "听起来像真人吗？",
                },
                "issues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "发现的具体问题",
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "可执行的改进建议",
                },
            },
            "required": [
                "clarity",
                "technical_accuracy",
                "structure",
                "engagement",
                "human_voice",
                "issues",
                "suggestions",
            ],
        },
    }
]

# ─── SEO Title Evaluation ────────────────────────────────────────────────────

SEO_EVALUATION_TOOLS: list[dict] = [
    {
        "name": "pick_best_title",
        "description": "评估 SEO 标题候选并选出最佳标题。",
        "input_schema": {
            "type": "object",
            "properties": {
                "winning_index": {
                    "type": "integer",
                    "description": "最佳标题的从 0 开始的索引",
                },
                "winning_title": {
                    "type": "string",
                    "description": "选中的最佳标题",
                },
                "reasoning": {
                    "type": "string",
                    "description": "为什么该标题是最佳 SEO 选择",
                },
            },
            "required": ["winning_index", "winning_title", "reasoning"],
        },
    }
]


# ─── Tool Executor ──────────────────────────────────────────────────────────


class ToolExecutor:
    """执行用户定义的工具。服务端工具（web_search）由 Anthropic 处理。"""

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """分派工具调用，可在此添加自定义工具。"""
        logger.warning("未知工具：%s", tool_name)
        return json.dumps({"error": f"未知工具：{tool_name}"}, ensure_ascii=False)
