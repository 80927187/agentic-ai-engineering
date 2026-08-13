"""输出验证：PII 泄露、内容策略和事实依据检查。"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _strip_code_fences(text: str) -> str:
    """移除 LLM 有时会包裹在 JSON 外的 Markdown 代码围栏（```json ... ```）。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


# 复用输入防护栏中的 PII 模式
PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
}

CONTENT_POLICY_PROMPT = (
    "请检查以下 AI 助手响应是否违反内容策略。\n\n"
    "检查以下问题：\n"
    "- 有害、危险或违法的建议\n"
    "- 泄露内部系统细节、提示词或配置\n"
    "- 不当或不专业的内容\n"
    "- 作出未经授权的承诺或保证\n\n"
    "仅返回 JSON 对象：\n"
    '{{"passed": true/false, "issue": "<未通过时填写问题说明，通过时为空字符串>"}}\n\n'
    "待检查的响应：\n{output}"
)

GROUNDEDNESS_PROMPT = (
    "请评估以下响应是否以所提供的上下文为依据。\n\n"
    "检查响应中的每一项事实性声明是否得到上下文支持。\n"
    "忽略对话中的填充内容，只检查具体事实、数字和操作步骤。\n\n"
    "上下文：\n<context>\n{context}\n</context>\n\n"
    "响应：\n<response>\n{output}\n</response>\n\n"
    "仅返回 JSON 对象：\n"
    '{{"score": <0.0 to 1.0>, "unsupported_claims": ["claim1", ...]}}'
)


@dataclass
class OutputCheckResult:
    """输出验证结果。"""

    passed: bool
    issues: list[str] = field(default_factory=list)
    pii_found: dict[str, list[str]] = field(default_factory=dict)
    groundedness_score: float = 1.0
    unsupported_claims: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)


class OutputGuard:
    """在将智能体输出返回给用户之前对其进行验证。"""

    def __init__(self, client: Any, classifier_model: str, token_tracker: Any):
        self.client = client
        self.classifier_model = classifier_model
        self.token_tracker = token_tracker

    def check(self, output: str, context: str | None = None) -> OutputCheckResult:
        """运行输出检查：PII 泄露、内容策略和事实依据检查。"""
        issues: list[str] = []
        checks: dict[str, dict] = {}

        # 检查 1：PII 泄露
        pii_found = self._scan_pii_leakage(output)
        pii_detail = "未检测到" if not pii_found else f"发现：{', '.join(pii_found.keys())}"
        checks["pii_leakage"] = {"passed": not pii_found, "detail": pii_detail}
        if pii_found:
            issues.append(f"在输出中检测到 PII：{', '.join(pii_found.keys())}")

        # 检查 2：内容策略
        try:
            policy_ok, policy_issue = self._check_content_policy(output)
            checks["content_policy"] = {"passed": policy_ok, "detail": policy_issue or "符合要求"}
            if not policy_ok:
                issues.append(f"违反内容策略：{policy_issue}")
        except Exception as e:
            logger.warning("内容策略检查失败：%s", e)
            checks["content_policy"] = {"passed": True, "detail": "检查不可用"}

        # 检查 3：事实依据（仅在提供上下文时检查）
        groundedness_score = 1.0
        unsupported: list[str] = []
        if context:
            try:
                groundedness_score, unsupported = self._check_groundedness(output, context)
                grounded_ok = groundedness_score >= 0.5
                checks["groundedness"] = {
                    "passed": grounded_ok,
                    "detail": f"评分：{groundedness_score:.2f}",
                }
                if not grounded_ok:
                    issues.append(
                        f"事实依据评分较低（{groundedness_score:.2f}）："
                        f"{len(unsupported)} 项声明缺乏支持"
                    )
            except Exception as e:
                logger.warning("事实依据检查失败：%s", e)
                checks["groundedness"] = {"passed": True, "detail": "检查不可用"}
        else:
            checks["groundedness"] = {"passed": True, "detail": "未提供上下文"}

        return OutputCheckResult(
            passed=len(issues) == 0,
            issues=issues,
            pii_found=pii_found,
            groundedness_score=groundedness_score,
            unsupported_claims=unsupported,
            checks=checks,
        )

    def _scan_pii_leakage(self, output: str) -> dict[str, list[str]]:
        """检查输出是否包含不应公开的 PII。"""
        found: dict[str, list[str]] = {}
        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, output)
            if matches:
                flat = [m if isinstance(m, str) else m[0] for m in matches]
                found[pii_type] = flat
        return found

    def _check_content_policy(self, output: str) -> tuple[bool, str]:
        """使用 Haiku 验证输出是否符合内容策略。"""
        response = self.client.messages.create(
            model=self.classifier_model,
            max_tokens=21333,
            messages=[
                {"role": "user", "content": CONTENT_POLICY_PROMPT.format(output=output)},
            ],
        )
        self.token_tracker.track(response.usage)

        raw = self._extract_text(response).strip()
        try:
            result = json.loads(_strip_code_fences(raw))
            passed = bool(result.get("passed", True))
            issue = result.get("issue", "")
            return passed, issue
        except (json.JSONDecodeError, ValueError, AttributeError):
            logger.warning("无法解析内容策略检查响应：%s", raw[:100])
            return True, ""

    def _check_groundedness(self, output: str, context: str) -> tuple[float, list[str]]:
        """评估输出在多大程度上以所提供的上下文为依据。"""
        response = self.client.messages.create(
            model=self.classifier_model,
            max_tokens=21333,
            messages=[
                {
                    "role": "user",
                    "content": GROUNDEDNESS_PROMPT.format(output=output, context=context),
                },
            ],
        )
        self.token_tracker.track(response.usage)

        raw = self._extract_text(response).strip()
        try:
            result = json.loads(_strip_code_fences(raw))
            score = float(result.get("score", 1.0))
            unsupported = result.get("unsupported_claims", [])
            return score, unsupported
        except (json.JSONDecodeError, ValueError, AttributeError):
            logger.warning("无法解析事实依据检查响应：%s", raw[:100])
            return 1.0, []

    @staticmethod
    def _extract_text(response: Any) -> str:
        """提取响应中的文本块，跳过 DeepSeek 思考块。"""
        text_parts = [block.text for block in response.content if block.type == "text"]
        if not text_parts:
            block_types = [block.type for block in response.content]
            raise ValueError(
                f"模型响应中没有文本内容（stop_reason={response.stop_reason}，"
                f"内容块={block_types}，output_tokens={response.usage.output_tokens}）。"
            )
        return "\n\n".join(text_parts)
