"""评测工具使用的轻量级追踪收集器。"""

import logging
import time

from eval_harness.models import TraceSpan

logger = logging.getLogger(__name__)


class SimpleTracer:
    """在评测执行期间记录 span 的轻量级追踪收集器。"""

    def __init__(self) -> None:
        self._spans: list[TraceSpan] = []
        self._active_spans: list[TraceSpan] = []

    def start_span(self, name: str, span_type: str) -> TraceSpan:
        """开始一个新的追踪 span 并将其返回。"""
        span = TraceSpan(
            name=name,
            span_type=span_type,
            start_time=time.time(),
        )

        # 如果存在当前活动 span，则嵌套在该 span 下
        if self._active_spans:
            self._active_spans[-1].children.append(span)
        else:
            self._spans.append(span)

        self._active_spans.append(span)
        logger.debug("span 已开始：%s（%s）", name, span_type)
        return span

    def end_span(self, span: TraceSpan) -> None:
        """记录结束时间，从而结束一个追踪 span。"""
        span.end_time = time.time()

        if self._active_spans and self._active_spans[-1] is span:
            self._active_spans.pop()

        logger.debug("span 已结束：%s（%.1f 毫秒）", span.name, span.duration_ms)

    def get_spans(self) -> list[TraceSpan]:
        """返回收集到的所有根级 span。"""
        return list(self._spans)

    def reset(self) -> None:
        """清除收集到的所有 span。"""
        self._spans.clear()
        self._active_spans.clear()

    def get_total_duration_ms(self) -> float:
        """计算所有根 span 的持续时间总和。"""
        return sum(s.duration_ms for s in self._spans)

    def get_span_count(self) -> int:
        """统计所有 span，包括嵌套的子 span。"""
        count = 0

        def _count(spans: list[TraceSpan]) -> None:
            nonlocal count
            for span in spans:
                count += 1
                _count(span.children)

        _count(self._spans)
        return count
