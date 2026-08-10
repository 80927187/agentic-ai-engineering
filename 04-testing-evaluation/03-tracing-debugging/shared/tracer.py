"""
共享追踪原语：Span 数据类和 TraceCollector。

提供所有追踪教程共用的基于跨度的核心追踪基础设施。
"""

import json
import time
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from common import setup_logging

logger = setup_logging(__name__)


@dataclass
class Span:
    """单个被追踪的操作。"""

    name: str
    span_type: str  # “LLM 调用”“工具调用”“智能体步骤”“搜索”
    start_time: float
    end_time: float | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)
    tokens: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        """以毫秒为单位的持续时间。"""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        """将跨度转换为可序列化的字典。"""
        return {
            "name": self.name,
            "span_type": self.span_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metadata": self.metadata,
            "tokens": self.tokens,
            "error": self.error,
            "children": [child.to_dict() for child in self.children],
        }


class TraceCollector:
    """使用分层跨度收集执行追踪。"""

    def __init__(self) -> None:
        self.trace_id: str = str(uuid.uuid4())[:8]
        self.root_spans: list[Span] = []
        self._span_stack: list[Span] = []

    @contextmanager
    def span(
        self, name: str, span_type: str, inputs: dict[str, Any] | None = None
    ) -> Generator[Span, None, None]:
        """用于创建追踪跨度的上下文管理器。"""
        new_span = Span(
            name=name,
            span_type=span_type,
            start_time=time.time(),
            inputs=inputs or {},
        )

        # 嵌套到当前父跨度下；没有父跨度时添加为根跨度
        if self._span_stack:
            self._span_stack[-1].children.append(new_span)
        else:
            self.root_spans.append(new_span)

        self._span_stack.append(new_span)
        try:
            yield new_span
        except Exception as e:
            new_span.error = str(e)
            raise
        finally:
            new_span.end_time = time.time()
            self._span_stack.pop()

    def traced(self, name: str, span_type: str) -> Callable:
        """用于追踪函数调用的装饰器。"""

        def decorator(func: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.span(name, span_type, inputs={"args": str(args), **kwargs}) as s:
                    result = func(*args, **kwargs)
                    s.outputs = {"result": str(result)[:200]}
                    return result

            return wrapper

        return decorator

    def to_dict(self) -> dict[str, Any]:
        """将追踪导出为可序列化的字典。"""
        return {
            "trace_id": self.trace_id,
            "spans": [span.to_dict() for span in self.root_spans],
        }

    def save(self, path: str) -> None:
        """将追踪保存到 JSON 文件。"""
        from pathlib import Path

        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("追踪已保存到 %s", path)


def collect_all_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按深度优先顺序将跨度树展平为列表。"""
    result: list[dict[str, Any]] = []
    for span in spans:
        result.append(span)
        result.extend(collect_all_spans(span.get("children", [])))
    return result
