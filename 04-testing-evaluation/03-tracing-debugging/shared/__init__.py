"""追踪与调试教程的共享模块。"""

from shared.agent import TracedResearchAssistant
from shared.knowledge_base import (
    KNOWLEDGE_BASE,
    SYSTEM_PROMPT,
    TOOL_FUNCTIONS,
    TOOLS,
    execute_tool,
    get_document,
    search_knowledge_base,
)
from shared.tracer import Span, TraceCollector, collect_all_spans

__all__ = [
    "KNOWLEDGE_BASE",
    "SYSTEM_PROMPT",
    "Span",
    "TOOL_FUNCTIONS",
    "TOOLS",
    "TraceCollector",
    "TracedResearchAssistant",
    "collect_all_spans",
    "execute_tool",
    "get_document",
    "search_knowledge_base",
]
