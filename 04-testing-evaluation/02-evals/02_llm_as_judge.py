"""LLM 充当评委的评估示例。

演示如何使用 LLM 按结构化评分标准评估智能体响应。
评委从准确性、完整性和依据充分性三个维度按 1～5 分评分，
并使用 tool_choice 强制输出结构化结果。
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
from common import AnthropicTokenTracker, setup_logging
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared.agent import ResearchAssistant
from shared.knowledge_base import KNOWLEDGE_BASE

load_dotenv(find_dotenv())

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# LLM 充当评委——使用评分标准进行结构化评估
# ---------------------------------------------------------------------------

# 评委使用 tool_choice 强制输出结构化结果
JUDGE_TOOLS = [
    {
        "name": "submit_evaluation",
        "description": "提交对智能体响应的结构化评估分数。",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "对响应质量的思维链推理",
                },
                "accuracy_score": {
                    "type": "integer",
                    "description": "1～5 分的准确性得分",
                    "minimum": 1,
                    "maximum": 5,
                },
                "accuracy_reason": {"type": "string"},
                "completeness_score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                },
                "completeness_reason": {"type": "string"},
                "grounding_score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                },
                "grounding_reason": {"type": "string"},
            },
            "required": [
                "reasoning",
                "accuracy_score",
                "accuracy_reason",
                "completeness_score",
                "completeness_reason",
                "grounding_score",
                "grounding_reason",
            ],
        },
    },
]

# 提供给评委的评分标准，用于保持评估一致性
JUDGE_SYSTEM_PROMPT = """你是一名研究助手智能体评估专家。

请使用以下标准评估智能体的响应：

**准确性（1～5 分）**
1：存在严重事实错误或捏造信息
2：存在多处不准确之处
3：大体准确，但有少量错误
4：准确，只有可忽略不计的问题
5：完全准确，所有事实都与参考文档一致

**完整性（1～5 分）**
1：遗漏绝大多数相关信息
2：覆盖的相关要点不足一半
3：覆盖主要要点，但遗漏部分细节
4：覆盖全面，仅有少量遗漏
5：内容完整，覆盖所有相关方面

**依据充分性（1～5 分）**
1：完全没有引用来源
2：部分说法缺乏支持
3：大多数说法有引用，但仍有缺漏
4：几乎所有说法都得到恰当引用
5：每项说法都有引用来源作为依据

始终使用 submit_evaluation 工具提供结构化评估结果。"""


@dataclass
class JudgeResult:
    """LLM 评委返回的结构化评估结果。"""

    reasoning: str
    accuracy_score: int
    accuracy_reason: str
    completeness_score: int
    completeness_reason: str
    grounding_score: int
    grounding_reason: str

    @property
    def avg_score(self) -> float:
        """计算所有维度的平均分。"""
        return (self.accuracy_score + self.completeness_score + self.grounding_score) / 3.0


class LLMJudge:
    """使用 LLM 按结构化评分标准评估智能体响应。"""

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-5-20250929",
    ) -> None:
        self.client = client
        self.model = model
        self.token_tracker = AnthropicTokenTracker()

    def evaluate(
        self,
        question: str,
        answer: str,
        reference_docs: list[dict[str, Any]],
        expected_answer: str | None = None,
    ) -> JudgeResult:
        """使用思维链评判来评估智能体响应。"""
        # 构建包含评委所需全部上下文的评估提示词
        ref_text = json.dumps(reference_docs, indent=2, ensure_ascii=False)
        prompt = (
            f"## 问题\n{question}\n\n"
            f"## 智能体答案\n{answer}\n\n"
            f"## 参考文档（真实标准）\n{ref_text}"
        )
        if expected_answer:
            prompt += f"\n\n## 预期答案摘要\n{expected_answer}"

        logger.info("LLM 评委正在评估答案（问题：%s...）", question[:50])

        # 通过 submit_evaluation 工具和 tool_choice 强制输出结构化结果
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=JUDGE_SYSTEM_PROMPT,
            tools=JUDGE_TOOLS,
            tool_choice={"type": "tool", "name": "submit_evaluation"},
            messages=[{"role": "user", "content": prompt}],
        )
        self.token_tracker.track(response.usage)

        # 从工具调用中提取结构化评估结果
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_evaluation":
                return JudgeResult(
                    reasoning=block.input["reasoning"],
                    accuracy_score=block.input["accuracy_score"],
                    accuracy_reason=block.input["accuracy_reason"],
                    completeness_score=block.input["completeness_score"],
                    completeness_reason=block.input["completeness_reason"],
                    grounding_score=block.input["grounding_score"],
                    grounding_reason=block.input["grounding_reason"],
                )

        # 未找到工具调用时的后备结果（使用 tool_choice 时不应发生）
        logger.warning("评委未返回结构化评估结果")
        return JudgeResult(
            reasoning="解析失败",
            accuracy_score=1,
            accuracy_reason="解析错误",
            completeness_score=1,
            completeness_reason="解析错误",
            grounding_score=1,
            grounding_reason="解析错误",
        )


# ---------------------------------------------------------------------------
# 演示模式使用的模拟评委结果
# ---------------------------------------------------------------------------

SIMULATED_JUDGE_RESULTS: dict[str, JudgeResult] = {
    "task_001": JudgeResult(
        reasoning=(
            "答案准确指出可扩展性、故障隔离和独立部署是主要优势，与 doc_001 一致。"
        ),
        accuracy_score=5,
        accuracy_reason="列出的所有优势都与参考文档完全一致。",
        completeness_score=4,
        completeness_reason="覆盖了主要优势，但遗漏了技术灵活性这一细节。",
        grounding_score=5,
        grounding_reason="正确引用 doc_001 作为来源。",
    ),
    "task_002": JudgeResult(
        reasoning=(
            "答案涵盖了 doc_002 中的端点名词、HTTP 方法、状态码、版本控制和分页。"
        ),
        accuracy_score=5,
        accuracy_reason="所有事实都与 doc_002 一致。",
        completeness_score=5,
        completeness_reason="覆盖了 REST API 设计的所有关键原则。",
        grounding_score=4,
        grounding_reason="引用了 doc_002，但部分说法没有明确标注来源。",
    ),
    "task_003": JudgeResult(
        reasoning=(
            "提到了 B 树索引、查找结构和 EXPLAIN，均来自 doc_003，但遗漏了复合索引细节。"
        ),
        accuracy_score=5,
        accuracy_reason="根据 doc_003，所述事实均正确。",
        completeness_score=3,
        completeness_reason="遗漏了复合索引和索引过多的权衡。",
        grounding_score=4,
        grounding_reason="引用了 doc_003，但并非所有说法都明确标注了来源。",
    ),
}


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main() -> None:
    """对研究助手的响应运行 LLM 评委评估。"""
    console = Console()
    console.print(
        Panel(
            "[bold cyan]LLM 充当评委的评估[/bold cyan]\n\n"
            "使用 LLM 从三个维度评估智能体响应：\n"
            "准确性、完整性和依据充分性（每项 1～5 分）。\n"
            "通过 tool_choice 强制输出结构化结果。",
            title="评估教程 2",
        )
    )

    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if has_api_key:
        console.print("[green]已找到 API 密钥——正在运行在线评估[/green]\n")
        client = anthropic.Anthropic()
        agent = ResearchAssistant(client, KNOWLEDGE_BASE)
        judge = LLMJudge(client)
    else:
        console.print("[yellow]未找到 API 密钥——演示将使用模拟结果[/yellow]\n")
        agent = None
        judge = None

    # 为本演示加载一部分任务
    dataset_path = Path(__file__).parent / "datasets" / "golden_tasks.json"
    with dataset_path.open(encoding="utf-8") as f:
        data = json.load(f)
    tasks = data["tasks"]

    # 演示仅使用前 3 个任务（LLM 评委成本较高）
    eval_tasks = tasks[:3] if agent is None else tasks[:5]
    console.print(f"正在使用 LLM 评委评估 {len(eval_tasks)} 个任务...\n")

    # 结果表格
    table = Table(title="LLM 评委结果", show_lines=True)
    table.add_column("任务", style="cyan", width=12)
    table.add_column("准确性", width=10, justify="center")
    table.add_column("完整性", width=12, justify="center")
    table.add_column("依据充分性", width=10, justify="center")
    table.add_column("平均分", width=8, justify="center")
    table.add_column("推理", width=50)

    all_results: list[JudgeResult] = []

    for task in eval_tasks:
        task_id = task["id"]
        logger.info("正在使用 LLM 评委评估 %s", task_id)

        if agent is not None and judge is not None:
            try:
                response = agent.answer(task["question"])
                # 收集提供给评委的参考文档
                ref_docs = [
                    doc for doc in KNOWLEDGE_BASE if doc["id"] in task["expected_source_ids"]
                ]
                result = judge.evaluate(
                    question=task["question"],
                    answer=response["answer"],
                    reference_docs=ref_docs if ref_docs else KNOWLEDGE_BASE[:2],
                )
            except Exception as e:
                logger.error("评估 %s 时出错：%s", task_id, e)
                result = JudgeResult(
                    reasoning=f"错误：{e}",
                    accuracy_score=1,
                    accuracy_reason="错误",
                    completeness_score=1,
                    completeness_reason="错误",
                    grounding_score=1,
                    grounding_reason="错误",
                )
        else:
            result = SIMULATED_JUDGE_RESULTS.get(
                task_id,
                JudgeResult(
                    reasoning="没有模拟结果",
                    accuracy_score=3,
                    accuracy_reason="N/A",
                    completeness_score=3,
                    completeness_reason="N/A",
                    grounding_score=3,
                    grounding_reason="N/A",
                ),
            )

        all_results.append(result)

        # 按得分设置颜色
        def score_color(s: int) -> str:
            if s >= 4:
                return f"[green]{s}/5[/green]"
            if s >= 3:
                return f"[yellow]{s}/5[/yellow]"
            return f"[red]{s}/5[/red]"

        # 截断推理文本以便在表格中显示
        short_reasoning = (
            result.reasoning[:80] + "..." if len(result.reasoning) > 80 else result.reasoning
        )

        table.add_row(
            task_id,
            score_color(result.accuracy_score),
            score_color(result.completeness_score),
            score_color(result.grounding_score),
            f"{result.avg_score:.1f}",
            short_reasoning,
        )

    console.print(table)

    # 汇总统计
    if all_results:
        avg_accuracy = sum(r.accuracy_score for r in all_results) / len(all_results)
        avg_completeness = sum(r.completeness_score for r in all_results) / len(all_results)
        avg_grounding = sum(r.grounding_score for r in all_results) / len(all_results)
        overall = sum(r.avg_score for r in all_results) / len(all_results)

        console.print("\n[bold]汇总得分[/bold]")
        console.print(f"  准确性：      {avg_accuracy:.2f}/5")
        console.print(f"  完整性：      {avg_completeness:.2f}/5")
        console.print(f"  依据充分性：  {avg_grounding:.2f}/5")
        console.print(f"  总体：        {overall:.2f}/5")

    # ---------------------------------------------------------------------------
    # 评分器校准：将 LLM 评委得分与人工基线进行比较
    # 最佳实践：使用人类专家的评分仔细校准 LLM 评委
    # ---------------------------------------------------------------------------
    console.print("\n[bold]评分器校准（LLM 评委与人工基线）[/bold]")
    console.print(
        "[dim]以下是前 3 个任务的模拟人工分数——实际应用中应向领域专家收集。[/dim]\n"
    )

    # 模拟的人类专家分数（生产环境中应来自标注环节）
    human_baselines: list[dict[str, int]] = [
        {"accuracy": 5, "completeness": 4, "grounding": 5},
        {"accuracy": 5, "completeness": 5, "grounding": 5},
        {"accuracy": 4, "completeness": 3, "grounding": 4},
    ]

    cal_table = Table(title="校准：LLM 评委与人类专家", show_lines=True)
    cal_table.add_column("任务", style="cyan", width=12)
    cal_table.add_column("维度", width=14)
    cal_table.add_column("人工", width=8, justify="center")
    cal_table.add_column("LLM 评委", width=10, justify="center")
    cal_table.add_column("差值", width=8, justify="center")

    num_calibration = min(3, len(all_results))
    for i in range(num_calibration):
        task_id = eval_tasks[i]["id"] if isinstance(eval_tasks[i], dict) else eval_tasks[i]
        judge_r = all_results[i]
        human = human_baselines[i]

        for dim, human_score, judge_score in [
            ("准确性", human["accuracy"], judge_r.accuracy_score),
            ("完整性", human["completeness"], judge_r.completeness_score),
            ("依据充分性", human["grounding"], judge_r.grounding_score),
        ]:
            delta = judge_score - human_score
            delta_str = f"{delta:+d}"
            delta_color = "green" if delta == 0 else ("yellow" if abs(delta) == 1 else "red")
            cal_table.add_row(
                task_id if dim == "准确性" else "",
                dim,
                str(human_score),
                str(judge_score),
                f"[{delta_color}]{delta_str}[/{delta_color}]",
            )

    console.print(cal_table)

    # token 用量
    if agent is not None:
        console.print("\n[bold]Token 用量[/bold]")
        console.print("[dim]智能体：[/dim]")
        agent.token_tracker.report()
    if judge is not None:
        console.print("[dim]评委：[/dim]")
        judge.token_tracker.report()


if __name__ == "__main__":
    main()
