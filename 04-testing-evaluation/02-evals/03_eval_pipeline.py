"""端到端评估流水线。

演示完整的评估流水线：加载黄金数据集、运行智能体试验、
使用多个评分器打分、汇总结果并检测回归。
报告 pass@k（至少一次成功）和 pass^k（全部成功）指标，
并按评估类型（能力与回归）细分结果。
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
from common import setup_logging
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared.agent import ResearchAssistant
from shared.graders import GraderResult, KeywordGrader, SourceCitationGrader, ToolCallGrader
from shared.knowledge_base import KNOWLEDGE_BASE

load_dotenv(find_dotenv())

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# 流水线数据结构
# ---------------------------------------------------------------------------


@dataclass
class EvalTask:
    """单个评估任务。"""

    id: str
    question: str
    expected_keywords: list[str]
    expected_source_ids: list[str]
    difficulty: str
    category: str
    eval_type: str = "capability"  # capability（新功能）或 regression（不得破坏）


@dataclass
class EvalTrial:
    """智能体针对一个任务的一次运行。"""

    task_id: str
    trial_number: int
    answer: str
    tool_calls: list[dict[str, Any]]
    sources: list[Any]
    latency_ms: float


@dataclass
class EvalResult:
    """一个任务跨多次试验的汇总结果。"""

    task_id: str
    trials: list[EvalTrial]
    grader_results: dict[str, list[GraderResult]]
    pass_rate: float
    avg_score: float
    # pass@k：k 次试验中至少成功一次的概率（乐观指标——衡量能力）
    pass_at_k: float = 0.0
    # pass^k：k 次试验全部成功的概率（严格指标——衡量一致性）
    pass_pow_k: float = 0.0


# ---------------------------------------------------------------------------
# 演示模式使用的模拟响应
# ---------------------------------------------------------------------------

# 每个任务映射到一组试验响应；run_trial 会依次使用它们。
# 试验之间的差异用于展示 pass@k 与 pass^k 的区别。
SIMULATED_RESPONSES: dict[str, dict[str, Any]] = {
    "task_001": {
        "answer": (
            "根据 doc_001，微服务架构的主要优势包括可扩展性、故障隔离和服务独立部署。"
            "每个服务都在自己的进程中运行，并通过 API 通信。"
        ),
        "tool_calls": [
            {
                "name": "search_knowledge_base",
                "input": {"query": "微服务 优势"},
                "results": [KNOWLEDGE_BASE[0]],
            }
        ],
        "sources": [[KNOWLEDGE_BASE[0]]],
    },
    "task_002": {
        "answer": (
            "根据 doc_002，REST API 的最佳实践包括：端点使用名词（例如 /users），操作使用 "
            "HTTP 方法（GET、POST、PUT、DELETE），并使用恰当的状态码。此外，还应采用版本控制，"
            "并对集合进行分页。"
        ),
        "tool_calls": [
            {
                "name": "search_knowledge_base",
                "input": {"query": "REST API 设计"},
                "results": [KNOWLEDGE_BASE[1]],
            }
        ],
        "sources": [[KNOWLEDGE_BASE[1]]],
    },
    "task_003": {
        "answer": (
            "根据 doc_003，数据库索引通过高效的查找结构提升查询性能。B 树索引适用于等值查询和"
            "范围查询。可以使用 EXPLAIN 分析查询计划。"
        ),
        "tool_calls": [
            {
                "name": "search_knowledge_base",
                "input": {"query": "数据库 索引"},
                "results": [KNOWLEDGE_BASE[2]],
            }
        ],
        "sources": [[KNOWLEDGE_BASE[2]]],
    },
    "task_004": {
        "answer": (
            "根据 doc_004，身份认证用于验证身份（你是谁），授权用于控制访问权限（你能做什么）。"
            "JWT 令牌提供无状态身份认证。密码必须始终使用 bcrypt 或 argon2 进行哈希处理。"
        ),
        "tool_calls": [
            {
                "name": "search_knowledge_base",
                "input": {"query": "身份认证 授权"},
                "results": [KNOWLEDGE_BASE[3]],
            }
        ],
        "sources": [[KNOWLEDGE_BASE[3]]],
    },
    "task_005": {
        "answer": (
            "根据 doc_005，CI/CD 的关键实践包括：持续集成在每次提交时自动构建和测试代码，"
            "持续部署将通过测试的构建部署到生产环境。快速反馈循环和基于主干的开发也很重要。"
        ),
        "tool_calls": [
            {
                "name": "search_knowledge_base",
                "input": {"query": "CI/CD 流水线"},
                "results": [KNOWLEDGE_BASE[4]],
            }
        ],
        "sources": [[KNOWLEDGE_BASE[4]]],
    },
    "task_013": {
        "answer": (
            "知识库中未找到有关机器学习编程语言的内容，因此没有相关信息可供回答。"
        ),
        "tool_calls": [
            {
                "name": "search_knowledge_base",
                "input": {"query": "机器学习 编程语言"},
                "results": [],
            }
        ],
        "sources": [[]],
    },
}

# 针对特定试验的覆盖响应，用于模拟 LLM 的非确定性行为。
# 没有覆盖的试验编号会回退到 SIMULATED_RESPONSES 中的默认响应。
SIMULATED_TRIAL_OVERRIDES: dict[str, dict[int, dict[str, Any]]] = {
    "task_001": {
        # 试验 2：较弱的答案缺少预期关键词——用于展示不一致性
        2: {
            "answer": (
                "微服务可以将应用程序拆分为通过网络通信的更小服务。"
            ),
            "tool_calls": [
                {
                    "name": "search_knowledge_base",
                    "input": {"query": "微服务 优势"},
                    "results": [KNOWLEDGE_BASE[0]],
                }
            ],
            "sources": [[KNOWLEDGE_BASE[0]]],
        },
    },
    "task_003": {
        # 试验 3：答案省略来源引用——引用评分器会判定失败
        3: {
            "answer": (
                "数据库索引通过高效的 B 树查找结构提升性能。可以使用 EXPLAIN 分析查询计划。"
            ),
            "tool_calls": [
                {
                    "name": "search_knowledge_base",
                    "input": {"query": "数据库 索引"},
                    "results": [KNOWLEDGE_BASE[2]],
                }
            ],
            "sources": [[]],  # 未引用来源
        },
    },
}


# ---------------------------------------------------------------------------
# 评估流水线
# ---------------------------------------------------------------------------


class EvalPipeline:
    """使用多个评分器打分的端到端评估流水线。"""

    def __init__(self, agent: ResearchAssistant | None = None) -> None:
        self.agent = agent
        self.keyword_grader = KeywordGrader()
        self.citation_grader = SourceCitationGrader()
        self.tool_grader = ToolCallGrader()

    def load_tasks(self, path: str) -> list[EvalTask]:
        """从 JSON 文件加载并解析评估任务。"""
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
        tasks = [
            EvalTask(
                id=t["id"],
                question=t["question"],
                expected_keywords=t["expected_keywords"],
                expected_source_ids=t["expected_source_ids"],
                difficulty=t["difficulty"],
                category=t["category"],
                eval_type=t.get("eval_type", "capability"),
            )
            for t in data["tasks"]
        ]
        logger.info("已从 %s 加载 %d 个评估任务", path, len(tasks))
        return tasks

    def run_trial(self, task: EvalTask, trial_number: int = 1) -> EvalTrial:
        """执行一次试验——运行智能体并测量延迟。"""
        start = time.perf_counter()

        if self.agent is not None:
            try:
                response = self.agent.answer(task.question)
            except Exception as e:
                logger.error("智能体处理 %s 时出错：%s", task.id, e)
                response = {"answer": f"错误：{e}", "tool_calls": [], "sources": []}
        else:
            # 优先查找特定试验的覆盖响应，否则回退到默认响应
            overrides = SIMULATED_TRIAL_OVERRIDES.get(task.id, {})
            response = overrides.get(
                trial_number,
                SIMULATED_RESPONSES.get(
                    task.id,
                    {"answer": "没有模拟响应。", "tool_calls": [], "sources": []},
                ),
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return EvalTrial(
            task_id=task.id,
            trial_number=0,
            answer=response["answer"],
            tool_calls=response.get("tool_calls", []),
            sources=response.get("sources", []),
            latency_ms=elapsed_ms,
        )

    def grade_trial(self, task: EvalTask, trial: EvalTrial) -> dict[str, GraderResult]:
        """使用所有评分器评估单次试验。"""
        return {
            "keywords": self.keyword_grader.grade(trial.answer, task.expected_keywords),
            "citations": self.citation_grader.grade(trial.answer, task.expected_source_ids),
            "tool_calls": self.tool_grader.grade(trial.tool_calls),
        }

    def run_evaluation(self, tasks: list[EvalTask], num_trials: int = 1) -> list[EvalResult]:
        """运行完整评估：每个任务执行多次试验，并逐一评分。"""
        results: list[EvalResult] = []

        for task in tasks:
            logger.info("正在评估 %s（%s，%s）", task.id, task.difficulty, task.category)
            trials: list[EvalTrial] = []
            all_grader_results: dict[str, list[GraderResult]] = {
                "keywords": [],
                "citations": [],
                "tool_calls": [],
            }

            for trial_num in range(num_trials):
                trial = self.run_trial(task, trial_number=trial_num + 1)
                trial.trial_number = trial_num + 1
                trials.append(trial)

                grader_results = self.grade_trial(task, trial)
                for name, result in grader_results.items():
                    all_grader_results[name].append(result)

            # 确定通过的试验（所有评分器都必须判定通过）
            pass_count = 0
            for i in range(num_trials):
                all_passed = all(all_grader_results[g][i].passed for g in all_grader_results)
                if all_passed:
                    pass_count += 1
            pass_rate = pass_count / num_trials

            # pass@k：至少一次试验成功（乐观指标——衡量能力）
            pass_at_k = 1.0 if pass_count > 0 else 0.0
            # pass^k：所有试验都成功（严格指标——衡量一致性/可靠性）
            pass_pow_k = 1.0 if pass_count == num_trials else 0.0

            # 所有评分器和试验的平均分
            all_scores = [
                r.score for grader_list in all_grader_results.values() for r in grader_list
            ]
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

            results.append(
                EvalResult(
                    task_id=task.id,
                    trials=trials,
                    grader_results=all_grader_results,
                    pass_rate=pass_rate,
                    avg_score=avg_score,
                    pass_at_k=pass_at_k,
                    pass_pow_k=pass_pow_k,
                )
            )

        return results

    def detect_regressions(
        self, current: list[EvalResult], baseline: list[EvalResult]
    ) -> list[str]:
        """将当前结果与基线比较，并标记回归。"""
        baseline_map = {r.task_id: r for r in baseline}
        regressions: list[str] = []

        for result in current:
            base = baseline_map.get(result.task_id)
            if base is None:
                continue

            # 通过率下降时标记回归
            if result.pass_rate < base.pass_rate:
                regressions.append(
                    f"{result.task_id}：通过率 {base.pass_rate:.0%} -> {result.pass_rate:.0%}"
                )

            # 平均分显著下降（> 0.1）时标记回归
            if result.avg_score < base.avg_score - 0.1:
                regressions.append(
                    f"{result.task_id}：平均分 {base.avg_score:.2f} -> {result.avg_score:.2f}"
                )

        return regressions


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main() -> None:
    """运行端到端评估流水线。"""
    console = Console()
    console.print(
        Panel(
            "[bold cyan]评估流水线[/bold cyan]\n\n"
            "端到端流程：加载黄金数据集、运行智能体试验、\n"
            "使用多个评分器打分、汇总 pass@k 和 pass^k、\n"
            "按评估类型（能力与回归）细分结果，并检测回归。",
            title="评估教程 3",
        )
    )

    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if has_api_key:
        console.print("[green]已找到 API 密钥——正在运行在线评估[/green]\n")
        client = anthropic.Anthropic()
        agent = ResearchAssistant(client, KNOWLEDGE_BASE)
    else:
        console.print("[yellow]未找到 API 密钥——演示将使用模拟响应[/yellow]\n")
        agent = None

    pipeline = EvalPipeline(agent=agent)

    # 加载任务
    dataset_path = Path(__file__).parent / "datasets" / "golden_tasks.json"
    all_tasks = pipeline.load_tasks(str(dataset_path))

    # 模拟模式仅运行有模拟响应的任务
    if agent is None:
        eval_tasks = [t for t in all_tasks if t.id in SIMULATED_RESPONSES]
        console.print(f"正在运行 {len(eval_tasks)} 个任务（模拟模式）...\n")
    else:
        eval_tasks = all_tasks
        console.print(f"正在运行 {len(eval_tasks)} 个任务...\n")

    # 运行评估——模拟模式使用 3 次试验来展示 pass@k 与 pass^k 的区别
    num_trials = 3 if agent is None else 1
    results = pipeline.run_evaluation(eval_tasks, num_trials=num_trials)

    # 各任务结果表格
    table = Table(title="各任务结果", show_lines=True)
    table.add_column("任务", style="cyan", width=12)
    table.add_column("类型", width=12)
    table.add_column("难度", width=10)
    table.add_column("关键词", width=10, justify="center")
    table.add_column("引用", width=10, justify="center")
    table.add_column("工具", width=10, justify="center")
    table.add_column("pass@k", width=8, justify="center")
    table.add_column("pass^k", width=8, justify="center")
    table.add_column("延迟", width=10, justify="right")

    def grader_cell(grader_name: str, eval_result: "EvalResult") -> str:
        """将评分器得分格式化为带颜色的 Rich 单元格。"""
        scores = eval_result.grader_results[grader_name]
        avg = sum(r.score for r in scores) / len(scores) if scores else 0.0
        color = "green" if avg >= 0.7 else ("yellow" if avg >= 0.4 else "red")
        return f"[{color}]{avg:.0%}[/{color}]"

    for result in results:
        task = next(t for t in eval_tasks if t.id == result.task_id)

        avg_latency = sum(t.latency_ms for t in result.trials) / len(result.trials)
        at_k_color = "green" if result.pass_at_k == 1.0 else "red"
        pow_k_color = "green" if result.pass_pow_k == 1.0 else "red"

        table.add_row(
            result.task_id,
            {"capability": "能力", "regression": "回归"}.get(task.eval_type, task.eval_type),
            task.difficulty,
            grader_cell("keywords", result),
            grader_cell("citations", result),
            grader_cell("tool_calls", result),
            f"[{at_k_color}]{result.pass_at_k:.0%}[/{at_k_color}]",
            f"[{pow_k_color}]{result.pass_pow_k:.0%}[/{pow_k_color}]",
            f"{avg_latency:.0f}ms",
        )

    console.print(table)

    # 汇总指标
    total_at_k = sum(r.pass_at_k for r in results) / len(results) if results else 0.0
    total_pow_k = sum(r.pass_pow_k for r in results) / len(results) if results else 0.0
    total_score = sum(r.avg_score for r in results) / len(results) if results else 0.0

    # 按评估类型细分（能力与回归）
    eval_types: dict[str, list[EvalResult]] = {}
    for result in results:
        task = next(t for t in eval_tasks if t.id == result.task_id)
        eval_types.setdefault(task.eval_type, []).append(result)

    type_table = Table(title="按评估类型细分")
    type_table.add_column("评估类型", style="bold")
    type_table.add_column("任务数", justify="center")
    type_table.add_column("pass@k", justify="center")
    type_table.add_column("pass^k", justify="center")
    type_table.add_column("平均分", justify="center")

    for etype, etype_results in sorted(eval_types.items()):
        e_at_k = sum(r.pass_at_k for r in etype_results) / len(etype_results)
        e_pow_k = sum(r.pass_pow_k for r in etype_results) / len(etype_results)
        e_score = sum(r.avg_score for r in etype_results) / len(etype_results)
        type_table.add_row(
            {"capability": "能力", "regression": "回归"}.get(etype, etype),
            str(len(etype_results)),
            f"{e_at_k:.0%}",
            f"{e_pow_k:.0%}",
            f"{e_score:.2f}",
        )

    console.print(type_table)

    # 按类别细分
    categories: dict[str, list[EvalResult]] = {}
    for result in results:
        task = next(t for t in eval_tasks if t.id == result.task_id)
        categories.setdefault(task.category, []).append(result)

    cat_table = Table(title="按类别细分")
    cat_table.add_column("类别", style="bold")
    cat_table.add_column("任务数", justify="center")
    cat_table.add_column("pass@k", justify="center")
    cat_table.add_column("pass^k", justify="center")
    cat_table.add_column("平均分", justify="center")

    for cat, cat_results in sorted(categories.items()):
        cat_at_k = sum(r.pass_at_k for r in cat_results) / len(cat_results)
        cat_pow_k = sum(r.pass_pow_k for r in cat_results) / len(cat_results)
        cat_score = sum(r.avg_score for r in cat_results) / len(cat_results)
        cat_table.add_row(
            cat, str(len(cat_results)), f"{cat_at_k:.0%}", f"{cat_pow_k:.0%}", f"{cat_score:.2f}"
        )

    console.print(cat_table)

    console.print(f"\n[bold]总体 pass@{num_trials}：[/bold] {total_at_k:.0%}")
    console.print(f"[bold]总体 pass^{num_trials}：[/bold] {total_pow_k:.0%}")
    console.print(f"[bold]总体平均分：[/bold] {total_score:.2f}")

    # 回归检测演示
    # 模拟一个得分略高的“基线”用于演示
    console.print("\n[bold]回归检测[/bold]")
    baseline = [
        EvalResult(
            task_id=r.task_id,
            trials=r.trials,
            grader_results=r.grader_results,
            pass_rate=min(r.pass_rate + 0.1, 1.0),
            avg_score=min(r.avg_score + 0.15, 1.0),
            pass_at_k=min(r.pass_at_k + 0.1, 1.0),
            pass_pow_k=min(r.pass_pow_k + 0.1, 1.0),
        )
        for r in results
    ]

    regressions = pipeline.detect_regressions(results, baseline)
    if regressions:
        console.print(f"[red]发现 {len(regressions)} 项回归：[/red]")
        for reg in regressions:
            console.print(f"  [red]- {reg}[/red]")
    else:
        console.print("[green]未检测到回归[/green]")

    # token 用量
    if agent is not None:
        console.print()
        agent.token_tracker.report()


if __name__ == "__main__":
    main()
