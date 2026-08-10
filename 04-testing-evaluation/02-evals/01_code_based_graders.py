"""用于智能体评估的代码评分器。

演示如何通过关键词匹配、正则表达式、来源引用验证和工具调用检查，
对智能体响应进行确定性评估。让研究助手运行黄金数据集中的任务，并为每个响应评分。
"""

import json
import os
from pathlib import Path
from typing import Any

import anthropic
from common import setup_logging
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared.agent import ResearchAssistant
from shared.graders import (
    GraderResult,
    KeywordGrader,
    RegexGrader,
    SourceCitationGrader,
    ToolCallGrader,
)
from shared.knowledge_base import KNOWLEDGE_BASE

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# 演示模式使用的模拟响应（没有 API 密钥时）
# ---------------------------------------------------------------------------

SIMULATED_RESPONSES: dict[str, dict[str, Any]] = {
    "task_001": {
        "answer": (
            "根据搜索结果（doc_001），微服务架构的主要优势包括可扩展性、故障隔离和技术灵活性。"
            "每个服务都可以独立部署，并在自己的进程中运行。"
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
            "根据 doc_002，REST API 的最佳实践包括：/users 和 /orders 等端点使用名词，"
            "操作使用 HTTP 方法（GET、POST、PUT、DELETE），使用恰当的状态码，实施版本控制，"
            "并对集合使用分页。"
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
            "根据 doc_003，数据库索引通过创建高效的查找结构来提升查询性能。B 树索引适用于"
            "等值查询和范围查询。可以使用 EXPLAIN 分析查询计划。"
        ),
        "tool_calls": [
            {
                "name": "search_knowledge_base",
                "input": {"query": "数据库 索引 性能"},
                "results": [KNOWLEDGE_BASE[2]],
            }
        ],
        "sources": [[KNOWLEDGE_BASE[2]]],
    },
}


# ---------------------------------------------------------------------------
# 评估运行器
# ---------------------------------------------------------------------------


def load_golden_tasks(path: str) -> list[dict[str, Any]]:
    """从 JSON 数据集文件加载评估任务。"""
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    logger.info("已从 %s 加载 %d 个任务（v%s）", path, len(data["tasks"]), data["version"])
    tasks: list[dict[str, Any]] = data["tasks"]
    return tasks


def evaluate_task(
    task: dict[str, Any],
    agent_response: dict[str, Any],
    keyword_grader: KeywordGrader,
    citation_grader: SourceCitationGrader,
    tool_grader: ToolCallGrader,
    regex_grader: RegexGrader,
) -> dict[str, GraderResult]:
    """使用所有评分器评估单个任务的智能体响应。"""
    answer = agent_response["answer"]
    tool_calls = agent_response.get("tool_calls", [])

    results: dict[str, GraderResult] = {}
    results["keywords"] = keyword_grader.grade(answer, task["expected_keywords"])
    results["citations"] = citation_grader.grade(answer, task["expected_source_ids"])
    results["tool_calls"] = tool_grader.grade(tool_calls)

    # 正则检查：答案应包含 doc_XXX 格式的引用（或拒答语句）
    if task["expected_source_ids"]:
        results["regex"] = regex_grader.grade(answer, r"doc_\d{3}")
    else:
        results["regex"] = regex_grader.grade(
            answer, r"(?:没有相关|未找到|没有信息|无法)"
        )

    return results


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main() -> None:
    """使用代码评分器评估黄金数据集。"""
    console = Console()
    console.print(
        Panel(
            "[bold cyan]基于代码的评分器[/bold cyan]\n\n"
            "使用确定性评分器评估研究助手：\n"
            "关键词匹配、正则表达式、来源引用和工具调用验证。",
            title="评估教程 1",
        )
    )

    # 确定运行模式：在线 API 或模拟模式
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if has_api_key:
        console.print("[green]已找到 API 密钥——正在运行在线评估[/green]\n")
        client = anthropic.Anthropic()
        agent = ResearchAssistant(client, KNOWLEDGE_BASE)
    else:
        console.print("[yellow]未找到 API 密钥——演示将使用模拟响应[/yellow]\n")
        agent = None

    # 加载黄金数据集
    dataset_path = Path(__file__).parent / "datasets" / "golden_tasks.json"
    tasks = load_golden_tasks(str(dataset_path))

    # 创建评分器
    keyword_grader = KeywordGrader()
    citation_grader = SourceCitationGrader()
    tool_grader = ToolCallGrader()
    regex_grader = RegexGrader()

    # 结果表格
    table = Table(title="评估结果", show_lines=True)
    table.add_column("任务", style="cyan", width=12)
    table.add_column("难度", width=8)
    table.add_column("关键词", width=18)
    table.add_column("引用", width=18)
    table.add_column("工具调用", width=18)
    table.add_column("正则", width=18)

    total_scores: dict[str, list[float]] = {
        "keywords": [],
        "citations": [],
        "tool_calls": [],
        "regex": [],
    }

    # 模拟模式只运行前几个任务，使演示保持简洁
    eval_tasks = tasks[:3] if agent is None else tasks
    console.print(f"正在运行 {len(eval_tasks)} 个任务...\n")

    for task in eval_tasks:
        logger.info("正在评估任务 %s：%s", task["id"], task["question"][:60])

        # 获取智能体响应（在线或模拟）
        if agent is not None:
            try:
                response = agent.answer(task["question"])
            except Exception as e:
                logger.error("智能体处理 %s 时出错：%s", task["id"], e)
                response = {"answer": f"错误：{e}", "tool_calls": [], "sources": []}
        else:
            response = SIMULATED_RESPONSES.get(
                task["id"],
                {"answer": "没有可用的模拟响应。", "tool_calls": [], "sources": []},
            )

        # 为响应评分
        grader_results = evaluate_task(
            task, response, keyword_grader, citation_grader, tool_grader, regex_grader
        )

        # 格式化表格结果
        def fmt(result: GraderResult) -> str:
            icon = "[green]通过[/green]" if result.passed else "[red]失败[/red]"
            return f"{icon} ({result.score:.0%})"

        table.add_row(
            task["id"],
            task["difficulty"],
            fmt(grader_results["keywords"]),
            fmt(grader_results["citations"]),
            fmt(grader_results["tool_calls"]),
            fmt(grader_results["regex"]),
        )

        for grader_name, result in grader_results.items():
            total_scores[grader_name].append(result.score)

    console.print(table)

    # 汇总
    console.print("\n[bold]汇总得分[/bold]")
    for grader_name, scores in total_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            console.print(f"  {grader_name:12s}：平均 {avg:.0%}（{len(scores)} 个任务）")

    # token 用量报告（仅在线模式）
    if agent is not None:
        console.print()
        agent.token_tracker.report()


if __name__ == "__main__":
    main()
