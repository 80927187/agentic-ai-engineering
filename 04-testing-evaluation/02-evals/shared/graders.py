"""用于确定性智能体评估的代码评分器。

提供关键词匹配、正则表达式匹配、来源引用验证和工具调用检查。
每个评分器都返回包含是否通过、得分（0～1）和易读原因的 GraderResult。
"""

import re
from dataclasses import dataclass
from typing import Any

from common import setup_logging

logger = setup_logging(__name__)


@dataclass
class GraderResult:
    """评分器的评估结果。"""

    passed: bool
    score: float  # 0.0～1.0
    reason: str


class KeywordGrader:
    """根据答案中必须出现的关键词评分。"""

    def grade(self, answer: str, expected_keywords: list[str]) -> GraderResult:
        """检查答案是否包含预期关键词。"""
        answer_lower = answer.lower()
        found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]

        score = len(found) / len(expected_keywords) if expected_keywords else 1.0
        passed = score >= 0.5

        reason = f"找到 {len(found)}/{len(expected_keywords)} 个关键词"
        if missing:
            reason += f"（缺少：{', '.join(missing)}）"

        logger.debug("KeywordGrader: score=%.2f, found=%s", score, found)
        return GraderResult(passed=passed, score=score, reason=reason)


class RegexGrader:
    """根据正则表达式匹配结果评分。"""

    def grade(self, answer: str, pattern: str) -> GraderResult:
        """检查答案是否匹配正则表达式。"""
        match = re.search(pattern, answer, re.IGNORECASE)
        passed = match is not None
        score = 1.0 if passed else 0.0
        reason = f"正则表达式“{pattern}”{'匹配成功' if passed else '未匹配'}"

        logger.debug("RegexGrader: pattern=%s, passed=%s", pattern, passed)
        return GraderResult(passed=passed, score=score, reason=reason)


class SourceCitationGrader:
    """根据答案是否引用来源评分。"""

    def grade(self, answer: str, expected_source_ids: list[str]) -> GraderResult:
        """检查答案是否引用预期的文档 ID。"""
        if not expected_source_ids:
            # 超出范围的任务：检查智能体是否说明没有相关信息
            has_refusal = bool(
                re.search(
                    r"没有相关|未找到|没有信息|无法找到", answer, re.IGNORECASE
                )
            )
            return GraderResult(
                passed=has_refusal,
                score=1.0 if has_refusal else 0.0,
                reason="超出范围：" + ("已正确拒答" if has_refusal else "应当拒答"),
            )

        cited = [sid for sid in expected_source_ids if sid in answer]
        missing = [sid for sid in expected_source_ids if sid not in answer]

        score = len(cited) / len(expected_source_ids)
        passed = score >= 0.5
        reason = f"引用了 {len(cited)}/{len(expected_source_ids)} 个来源"
        if missing:
            reason += f"（缺少：{', '.join(missing)}）"

        logger.debug("SourceCitationGrader: score=%.2f, cited=%s", score, cited)
        return GraderResult(passed=passed, score=score, reason=reason)


class ToolCallGrader:
    """根据智能体是否执行预期的工具调用评分。"""

    def grade(
        self, tool_calls: list[dict[str, Any]], expected_tool: str = "search_knowledge_base"
    ) -> GraderResult:
        """验证智能体是否至少调用过一次预期工具。"""
        tool_names = [tc.get("name", "") for tc in tool_calls]
        called = expected_tool in tool_names
        score = 1.0 if called else 0.0

        reason = (
            f"已调用工具“{expected_tool}”（共调用 {len(tool_calls)} 次）"
            if called
            else f"未调用工具“{expected_tool}”"
        )

        logger.debug("ToolCallGrader: tool=%s, called=%s", expected_tool, called)
        return GraderResult(passed=called, score=score, reason=reason)
