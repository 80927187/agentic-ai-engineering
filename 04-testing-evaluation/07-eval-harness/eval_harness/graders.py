"""用于评估智能体回答的评分器实现。"""

import logging
import re

from eval_harness.models import EvalTask, EvalTrial, GraderScore

logger = logging.getLogger(__name__)


class KeywordGrader:
    """根据回答中是否包含预期关键词进行评分。"""

    def grade(self, answer: str, expected_keywords: list[str]) -> GraderScore:
        """检查回答是否包含预期关键词。"""
        if not expected_keywords:
            return GraderScore(
                grader_name="keyword",
                passed=True,
                score=1.0,
                reason="没有预期关键词",
            )

        answer_lower = answer.lower()
        found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
        score = len(found) / len(expected_keywords)
        passed = score >= 0.5

        reason = f"找到 {len(found)}/{len(expected_keywords)} 个关键词"
        if missing:
            reason += f"（缺少：{', '.join(missing)}）"

        logger.debug("关键词评分器：分数=%.2f，已找到=%s", score, found)
        return GraderScore(grader_name="keyword", passed=passed, score=score, reason=reason)


class SourceCitationGrader:
    """根据回答是否引用预期来源进行评分。"""

    def grade(self, answer: str, expected_source_ids: list[str]) -> GraderScore:
        """检查回答是否引用预期文档 ID。"""
        if not expected_source_ids:
            # 对于范围外任务，智能体应说明没有相关信息
            has_refusal = bool(
                re.search(
                    r"no relevant|not found|no information|cannot find|could not find|"
                    r"没有相关|未找到|没有找到|没有信息|找不到|无法找到",
                    answer,
                    re.IGNORECASE,
                )
            )
            return GraderScore(
                grader_name="citation",
                passed=has_refusal,
                score=1.0 if has_refusal else 0.0,
                reason="范围外任务：" + ("已正确拒绝" if has_refusal else "应当拒绝"),
            )

        cited = [sid for sid in expected_source_ids if sid in answer]
        missing = [sid for sid in expected_source_ids if sid not in answer]
        score = len(cited) / len(expected_source_ids)
        passed = score >= 0.5

        reason = f"引用了 {len(cited)}/{len(expected_source_ids)} 个来源"
        if missing:
            reason += f"（缺少：{', '.join(missing)}）"

        logger.debug("来源引用评分器：分数=%.2f，已引用=%s", score, cited)
        return GraderScore(grader_name="citation", passed=passed, score=score, reason=reason)


class CompositeGrader:
    """按可配置权重组合多个评分器。"""

    def __init__(
        self,
        keyword_weight: float = 0.5,
        citation_weight: float = 0.5,
    ) -> None:
        self.keyword_grader = KeywordGrader()
        self.citation_grader = SourceCitationGrader()
        self.keyword_weight = keyword_weight
        self.citation_weight = citation_weight

    def grade(self, trial: EvalTrial, task: EvalTask) -> list[GraderScore]:
        """运行所有评分器并返回分数列表。"""
        scores: list[GraderScore] = []

        # 关键词评分
        keyword_score = self.keyword_grader.grade(trial.answer, task.expected_keywords)
        scores.append(keyword_score)

        # 来源引用评分
        citation_score = self.citation_grader.grade(trial.answer, task.expected_source_ids)
        scores.append(citation_score)

        # 工具调用评分——验证智能体是否使用了搜索工具
        tool_names = [tc.get("name", "") for tc in trial.tool_calls]
        tool_called = "search_knowledge_base" in tool_names
        tool_score = GraderScore(
            grader_name="tool_call",
            passed=tool_called,
            score=1.0 if tool_called else 0.0,
            reason=(
                f"已调用 search_knowledge_base（共调用 {len(trial.tool_calls)} 次）"
                if tool_called
                else "未调用 search_knowledge_base"
            ),
        )
        scores.append(tool_score)

        # 复合分数——关键词评分与引用评分的加权平均值
        composite_val = (
            keyword_score.score * self.keyword_weight + citation_score.score * self.citation_weight
        )
        composite_passed = composite_val >= 0.5
        scores.append(
            GraderScore(
                grader_name="composite",
                passed=composite_passed,
                score=round(composite_val, 3),
                reason=(
                    f"加权结果：关键词（{self.keyword_weight}）+ "
                    f"引用（{self.citation_weight}）= {composite_val:.3f}"
                ),
            )
        )

        logger.debug(
            "%s 的复合评分器结果：复合分数=%.3f，通过=%s",
            task.id,
            composite_val,
            composite_passed,
        )
        return scores
