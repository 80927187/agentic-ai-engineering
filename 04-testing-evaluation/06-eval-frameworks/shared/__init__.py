"""评测框架教程的共用模块。"""

from shared.knowledge_base import (
    EVAL_TASKS,
    KNOWLEDGE_BASE,
    SIMULATED_RESPONSES,
    get_agent_response,
    search_knowledge_base,
)

__all__ = [
    "EVAL_TASKS",
    "KNOWLEDGE_BASE",
    "SIMULATED_RESPONSES",
    "get_agent_response",
    "search_knowledge_base",
]
