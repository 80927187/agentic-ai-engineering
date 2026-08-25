"""
Braintrust AutoEvals——用于智能体评测的预置评分器。

演示如何使用 Braintrust `autoevals` 库评测智能体响应。AutoEvals 提供开箱即用的
字符串相似度评分器（本地运行，无需 API 密钥）、事实性等基于 LLM 的评分器
（需要 OpenAI 密钥）以及自定义 LLM 分类器。

本脚本将：
1. 展示完全在本地运行的字符串评分器（Levenshtein、ExactMatch）
2. 演示需要 OPENAI_API_KEY 的 LLM 评分器（Factuality、ClosedQA）
3. 构建用于特定领域评分的自定义 LLM 分类器
4. 使用所有评分器评测研究助手的响应

安装：pip install autoevals
"""

import os

from common import setup_logging
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared.knowledge_base import EVAL_TASKS, get_agent_response

load_dotenv(find_dotenv())

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# 演示模式下的模拟评分结果（未安装 autoevals 时使用）
# ---------------------------------------------------------------------------

SIMULATED_SCORES: dict[str, dict[str, float]] = {
    "task_001": {"levenshtein": 0.42, "factuality": 0.9, "closedqa": 0.85},
    "task_002": {"levenshtein": 0.38, "factuality": 0.95, "closedqa": 0.90},
    "task_003": {"levenshtein": 0.45, "factuality": 0.85, "closedqa": 0.80},
    "task_004": {"levenshtein": 0.51, "factuality": 0.90, "closedqa": 0.85},
    "task_005": {"levenshtein": 0.30, "factuality": 0.80, "closedqa": 0.70},
}


def run_string_scorers(output: str, expected: str, task_id: str) -> dict[str, float]:
    """运行字符串评分器（无需 API 密钥）。"""
    try:
        from autoevals import Levenshtein

        lev = Levenshtein()
        lev_result = lev.eval(output=output, expected=expected)
        return {"levenshtein": lev_result.score or 0.0}
    except ImportError:
        logger.info("未安装 autoevals，使用模拟评分")
        return {"levenshtein": SIMULATED_SCORES.get(task_id, {}).get("levenshtein", 0.0)}


def run_llm_scorers(question: str, output: str, expected: str, task_id: str) -> dict[str, float]:
    """运行基于 LLM 的评分器（需要 OPENAI_API_KEY）。"""
    has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))

    if not has_openai_key:
        logger.info("未设置 OPENAI_API_KEY，使用模拟的 LLM 评分结果")
        sim = SIMULATED_SCORES.get(task_id, {})
        return {
            "factuality": sim.get("factuality", 0.0),
            "closedqa": sim.get("closedqa", 0.0),
        }

    try:
        from autoevals import ClosedQA, Factuality

        scores: dict[str, float] = {}

        # Factuality：检查输出在事实层面是否与预期答案一致
        factuality = Factuality("gpt-5.6-terra")
        fact_result = factuality.eval(
            input=question,
            output=output,
            expected=expected,
        )
        scores["factuality"] = fact_result.score or 0.0

        # ClosedQA：根据问题评估答案质量
        closedqa = ClosedQA("gpt-5.6-terra")
        cqa_result = closedqa.eval(
            input=question,
            output=output,
            expected=expected,
            criteria=f"回答应正确回答问题，并符合参考答案：{expected}",
        )
        scores["closedqa"] = cqa_result.score or 0.0

        return scores
    except ImportError:
        logger.info("未安装 autoevals，使用模拟评分")
        sim = SIMULATED_SCORES.get(task_id, {})
        return {
            "factuality": sim.get("factuality", 0.0),
            "closedqa": sim.get("closedqa", 0.0),
        }
    except Exception as e:
        logger.error("LLM 评分器出错：%s", e)
        return {"factuality": 0.0, "closedqa": 0.0}


def run_custom_classifier(output: str, task_id: str) -> dict[str, float]:
    """运行用于检查来源依据的自定义 LLM 分类器。"""
    has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))

    if not has_openai_key:
        # 模拟模式：检查输出是否包含 doc_XXX 格式的文本
        has_source = "doc_" in output
        return {"grounding": 1.0 if has_source else 0.0}

    try:
        from autoevals import LLMClassifier

        grounding_classifier = LLMClassifier(
            name="来源依据",
            model="gpt-5.6-terra",
            prompt_template=(
                "以下回答是否引用了具体来源（如 doc_001）来支持其陈述？"
                "\n\n回答：{{output}}\n\n"
                "若引用了来源，请回答‘是’，否则回答‘否’。"
            ),
            choice_scores={"是": 1.0, "否": 0.0},
        )
        result = grounding_classifier.eval(output=output)
        return {"grounding": result.score or 0.0}
    except ImportError:
        has_source = "doc_" in output
        return {"grounding": 1.0 if has_source else 0.0}
    except Exception as e:
        logger.error("自定义分类器出错：%s", e)
        return {"grounding": 0.0}


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main() -> None:
    """使用 Braintrust AutoEvals 评分器评测研究助手的响应。"""
    console = Console()
    console.print(
        Panel(
            "[bold cyan]Braintrust AutoEvals——预置评分器[/bold cyan]\n\n"
            "使用以下评分器评测智能体响应：\n"
            "  - 字符串评分器：Levenshtein 相似度（本地运行，无需 API 密钥）\n"
            "  - LLM 评分器：Factuality、ClosedQA（需要 OPENAI_API_KEY）\n"
            "  - 自定义分类器：来源依据检查\n\n"
            "安装：pip install autoevals",
            title="02 - Braintrust AutoEvals",
        )
    )

    # 检查可用组件
    has_autoevals = False
    try:
        import autoevals  # noqa: F401

        has_autoevals = True
    except ImportError:
        pass

    has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))

    if has_autoevals and has_openai_key:
        console.print("[green]已安装 autoevals 并设置 OpenAI 密钥——实时运行所有评分器[/green]")
    elif has_autoevals:
        console.print(
            "[yellow]已安装 autoevals，但未设置 OpenAI 密钥——"
            "实时运行字符串评分器，模拟 LLM 评分器[/yellow]"
        )
    else:
        console.print("[yellow]未安装 autoevals——使用模拟评分进行演示[/yellow]")
    console.print()

    # 结果表格
    table = Table(title="AutoEvals 评分结果", show_lines=True)
    table.add_column("任务", style="cyan", width=12)
    table.add_column("Levenshtein", width=12, justify="center")
    table.add_column("事实性", width=12, justify="center")
    table.add_column("ClosedQA", width=12, justify="center")
    table.add_column("来源依据", width=12, justify="center")
    table.add_column("平均分", width=10, justify="center")

    all_scores: list[dict[str, float]] = []

    for task in EVAL_TASKS:
        response = get_agent_response(task["id"])
        output = response["answer"]
        expected = task["reference_answer"]

        # 运行所有类别的评分器
        scores: dict[str, float] = {}
        scores.update(run_string_scorers(output, expected, task["id"]))
        scores.update(run_llm_scorers(task["question"], output, expected, task["id"]))
        scores.update(run_custom_classifier(output, task["id"]))

        all_scores.append(scores)

        # 设置表格行格式
        def fmt(s: float) -> str:
            color = "green" if s >= 0.7 else ("yellow" if s >= 0.4 else "red")
            return f"[{color}]{s:.2f}[/{color}]"

        avg = sum(scores.values()) / len(scores) if scores else 0.0
        avg_color = "green" if avg >= 0.7 else ("yellow" if avg >= 0.4 else "red")

        table.add_row(
            task["id"],
            fmt(scores.get("levenshtein", 0.0)),
            fmt(scores.get("factuality", 0.0)),
            fmt(scores.get("closedqa", 0.0)),
            fmt(scores.get("grounding", 0.0)),
            f"[{avg_color}]{avg:.2f}[/{avg_color}]",
        )

    console.print(table)

    # 汇总结果
    if all_scores:
        console.print("\n[bold]汇总评分[/bold]")
        for scorer_name in ["levenshtein", "factuality", "closedqa", "grounding"]:
            values = [s.get(scorer_name, 0.0) for s in all_scores]
            avg = sum(values) / len(values)
            console.print(f"  {scorer_name:12s}: {avg:.2f}")

    # 展示代码示例
    console.print("\n[bold]用法示例（独立使用 AutoEvals）：[/bold]\n")
    code = (
        "from autoevals import Factuality, Levenshtein\n\n"
        "# 字符串评分器——完全本地运行，无需 API 密钥\n"
        "lev = Levenshtein()\n"
        'result = lev.eval(output="你好世届", expected="你好世界")\n'
        "print(result.score)  # ~0.91\n\n"
        "# LLM 评分器——需要 OPENAI_API_KEY\n"
        "fact = Factuality()\n"
        "result = fact.eval(\n"
        '    input="法国的首都是哪里？",\n'
        '    output="法国的首都是巴黎。",\n'
        '    expected="巴黎是法国的首都。",\n'
        ")\n"
        "print(result.score)  # 1.0\n"
    )
    from rich.syntax import Syntax

    console.print(Syntax(code, "python", theme="monokai", line_numbers=True))

    # 展示 Braintrust Eval() 模式
    console.print("\n[bold]完整评测流水线（使用 Braintrust 平台）：[/bold]\n")
    eval_code = (
        "from braintrust import Eval\n"
        "from autoevals import Factuality, Levenshtein\n\n"
        "Eval(\n"
        '    "研究助手评测套件",\n'
        "    data=lambda: [\n"
        '        {"input": "什么是微服务？", "expected": "..."},\n'
        "    ],\n"
        "    task=lambda input: my_agent.answer(input),\n"
        "    scores=[Factuality, Levenshtein],\n"
        ")\n"
    )
    console.print(Syntax(eval_code, "python", theme="monokai", line_numbers=True))


if __name__ == "__main__":
    main()
