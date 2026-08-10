"""评测工具的 Rich 终端报告生成器。"""

import logging
from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from eval_harness.models import BenchmarkEntry, EvalReport, EvalResult, SafetyResult

logger = logging.getLogger(__name__)


class EvalReporter:
    """根据评测结果生成 Rich 终端报告。"""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def print_report(self, report: EvalReport) -> None:
        """输出完整的评测报告。"""
        self.print_summary_panel(report)
        self.console.print()
        self.print_quality_table(report.eval_results)
        self.console.print()
        self.print_safety_table(report.safety_results)
        self.console.print()
        self.print_benchmark_table(report.benchmark_entries)

    def print_quality_table(self, results: list[EvalResult]) -> None:
        """以表格形式输出详细的质量评测结果。"""
        table = Table(title="质量评测结果", show_lines=True)
        table.add_column("任务", style="cyan", width=10)
        table.add_column("通过率", width=10)
        table.add_column("平均分", width=10)
        table.add_column("关键词", width=20)
        table.add_column("引用", width=20)
        table.add_column("工具调用", width=20)

        for result in results:
            # 提取各评分器的分数
            grader_map = {gs.grader_name: gs for gs in result.grader_scores}

            def fmt_grader(name: str, gm: dict = grader_map) -> str:
                gs = gm.get(name)
                if not gs:
                    return "[dim]不适用[/dim]"
                icon = "[green]通过[/green]" if gs.passed else "[red]失败[/red]"
                return f"{icon} ({gs.score:.0%})"

            pass_style = "green" if result.pass_rate >= 0.5 else "red"
            table.add_row(
                result.task_id,
                f"[{pass_style}]{result.pass_rate:.0%}[/{pass_style}]",
                f"{result.avg_score:.2f}",
                fmt_grader("keyword"),
                fmt_grader("citation"),
                fmt_grader("tool_call"),
            )

        self.console.print(table)

    def print_safety_table(self, results: list[SafetyResult]) -> None:
        """以表格形式输出安全测试结果。"""
        if not results:
            return

        table = Table(title="安全测试结果", show_lines=True)
        table.add_column("ID", style="dim", width=8)
        table.add_column("攻击", width=28)
        table.add_column("类别", width=18)
        table.add_column("严重程度", width=10)
        table.add_column("结果", width=10)

        for result in results:
            severity_style = {
                "high": "red",
                "critical": "red bold",
                "medium": "yellow",
                "low": "green",
            }.get(result.severity, "white")

            result_style = "green" if result.blocked else "red bold"
            result_text = "已阻止" if result.blocked else "已绕过"
            category_text = {
                "injection": "提示词注入",
                "hallucination": "幻觉",
                "scope": "范围",
                "indirect_injection": "间接注入",
                "exfiltration": "数据窃取",
                "jailbreak": "越狱",
                "encoding": "编码绕过",
            }.get(result.category, result.category)
            severity_text = {
                "critical": "严重",
                "high": "高",
                "medium": "中",
                "low": "低",
            }.get(result.severity, result.severity)

            table.add_row(
                result.attack_id,
                result.attack_name,
                category_text,
                f"[{severity_style}]{severity_text}[/{severity_style}]",
                f"[{result_style}]{result_text}[/{result_style}]",
            )

        self.console.print(table)

    def print_benchmark_table(self, entries: list[BenchmarkEntry]) -> None:
        """按模型配置汇总并输出基准测试结果。"""
        if not entries:
            return

        # 按配置汇总
        config_data: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"accuracy": [], "latency": [], "cost": [], "tokens": []}
        )
        for entry in entries:
            config_data[entry.config_name]["accuracy"].append(entry.accuracy)
            config_data[entry.config_name]["latency"].append(entry.latency_ms)
            config_data[entry.config_name]["cost"].append(entry.cost_usd)
            config_data[entry.config_name]["tokens"].append(float(entry.tokens))

        table = Table(title="基准测试结果（按模型配置）", show_lines=True)
        table.add_column("配置", style="cyan", width=16)
        table.add_column("平均准确率", width=14)
        table.add_column("平均延迟", width=14)
        table.add_column("总成本", width=14)
        table.add_column("平均 token 数", width=14)

        for config_name, data in config_data.items():
            avg_acc = sum(data["accuracy"]) / len(data["accuracy"])
            avg_lat = sum(data["latency"]) / len(data["latency"])
            total_cost = sum(data["cost"])
            avg_tok = sum(data["tokens"]) / len(data["tokens"])

            acc_style = "green" if avg_acc >= 0.8 else ("yellow" if avg_acc >= 0.6 else "red")
            table.add_row(
                config_name,
                f"[{acc_style}]{avg_acc:.1%}[/{acc_style}]",
                f"{avg_lat:.0f}ms",
                f"${total_cost:.4f}",
                f"{avg_tok:.0f}",
            )

        self.console.print(table)

    def print_summary_panel(self, report: EvalReport) -> None:
        """输出最终汇总面板。"""
        # 质量统计
        total_tasks = len(report.eval_results)
        passed_tasks = sum(1 for r in report.eval_results if r.pass_rate >= 0.5)
        quality_pct = (passed_tasks / total_tasks * 100) if total_tasks else 0

        # 安全统计
        total_attacks = len(report.safety_results)
        blocked_attacks = sum(1 for r in report.safety_results if r.blocked)
        safety_pct = (blocked_attacks / total_attacks * 100) if total_attacks else 0

        # 延迟
        avg_latency = report.total_latency_ms / total_tasks if total_tasks else 0

        # 构建汇总文本
        quality_style = "green" if quality_pct >= 80 else ("yellow" if quality_pct >= 60 else "red")
        safety_style = "green" if safety_pct >= 80 else ("yellow" if safety_pct >= 60 else "red")

        summary = (
            f"  质量评测             "
            f"[{quality_style}]{passed_tasks}/{total_tasks} 项任务通过 "
            f"({quality_pct:.1f}%)[/{quality_style}]\n"
            f"  安全分数             "
            f"[{safety_style}]阻止了 {blocked_attacks}/{total_attacks} 次攻击 "
            f"({safety_pct:.1f}%)[/{safety_style}]\n"
            f"  平均延迟             每项任务 {avg_latency / 1000:.1f} 秒\n"
            f"  总成本               ${report.total_cost_usd:.4f}"
        )

        self.console.print(
            Panel(
                summary,
                title=f"评测报告：{report.agent_name}",
                border_style="bold cyan",
            )
        )
