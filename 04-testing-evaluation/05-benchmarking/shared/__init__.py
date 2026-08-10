"""基准测试教程的共享模块。"""

from shared.knowledge_base import (
    BENCHMARK_TASKS,
    KNOWLEDGE_BASE,
    SYSTEM_PROMPT,
    TOOLS_ANTHROPIC,
    TOOLS_OPENAI,
    score_answer,
    search_knowledge_base,
)
from shared.models import (
    BenchmarkConfig,
    BenchmarkResult,
    MODEL_CONFIGS,
    ModelConfig,
)

__all__ = [
    "BENCHMARK_TASKS",
    "BenchmarkConfig",
    "BenchmarkResult",
    "KNOWLEDGE_BASE",
    "MODEL_CONFIGS",
    "ModelConfig",
    "SYSTEM_PROMPT",
    "TOOLS_ANTHROPIC",
    "TOOLS_OPENAI",
    "score_answer",
    "search_knowledge_base",
]
