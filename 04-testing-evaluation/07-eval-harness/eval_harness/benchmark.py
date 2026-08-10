"""评测工具的模型基准测试与帕累托分析。"""

import logging
import random
from typing import Any

from eval_harness.models import BenchmarkEntry, EvalTask

logger = logging.getLogger(__name__)

# 具有现实性能特征的模拟模型配置
DEFAULT_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "claude-haiku",
        "model": "claude-3-5-haiku-20241022",
        "avg_latency_ms": 600.0,
        "cost_per_1k_input": 0.001,
        "cost_per_1k_output": 0.005,
        "accuracy_modifier": 0.75,
    },
    {
        "name": "claude-sonnet",
        "model": "claude-sonnet-4-5-20250929",
        "avg_latency_ms": 1500.0,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "accuracy_modifier": 0.90,
    },
    {
        "name": "claude-opus",
        "model": "claude-opus-4-0-20250514",
        "avg_latency_ms": 3000.0,
        "cost_per_1k_input": 0.015,
        "cost_per_1k_output": 0.075,
        "accuracy_modifier": 0.95,
    },
]


class BenchmarkRunner:
    """针对多种模型配置运行基准测试。"""

    def __init__(self, configs: list[dict[str, Any]] | None = None) -> None:
        self.configs = configs or DEFAULT_CONFIGS

    def run_benchmark(
        self, tasks: list[EvalTask], configs: list[dict[str, Any]] | None = None
    ) -> list[BenchmarkEntry]:
        """针对多种模型配置和任务运行模拟基准测试。"""
        configs = configs or self.configs
        entries: list[BenchmarkEntry] = []

        for config in configs:
            logger.info("正在对配置进行基准测试：%s", config["name"])
            for task in tasks:
                entry = self._simulate_benchmark(task, config)
                entries.append(entry)

        logger.info(
            "基准测试完成：%d 条结果，涉及 %d 项配置",
            len(entries),
            len(configs),
        )
        return entries

    def _simulate_benchmark(self, task: EvalTask, config: dict[str, Any]) -> BenchmarkEntry:
        """模拟一个任务与配置组合的基准测试运行。"""
        # 模拟带波动的延迟
        base_latency = config["avg_latency_ms"]
        latency = base_latency + random.uniform(-base_latency * 0.2, base_latency * 0.2)

        # 根据任务难度和模型能力模拟准确率
        difficulty_modifier = {"easy": 1.0, "medium": 0.85, "hard": 0.7}.get(task.difficulty, 0.85)
        accuracy = min(1.0, config["accuracy_modifier"] * difficulty_modifier)

        # 模拟 token 用量
        input_tokens = random.randint(200, 400)
        output_tokens = random.randint(80, 200)
        total_tokens = input_tokens + output_tokens

        # 计算成本
        cost = (
            input_tokens / 1000 * config["cost_per_1k_input"]
            + output_tokens / 1000 * config["cost_per_1k_output"]
        )

        return BenchmarkEntry(
            config_name=config["name"],
            task_id=task.id,
            accuracy=round(accuracy, 3),
            latency_ms=round(latency, 1),
            cost_usd=round(cost, 6),
            tokens=total_tokens,
        )

    def find_pareto_optimal(self, entries: list[BenchmarkEntry]) -> list[str]:
        """找出在准确率、延迟和成本之间取得平衡的帕累托最优配置。"""
        # 按配置汇总指标
        config_metrics: dict[str, dict[str, float]] = {}
        for entry in entries:
            if entry.config_name not in config_metrics:
                config_metrics[entry.config_name] = {
                    "accuracy_sum": 0.0,
                    "latency_sum": 0.0,
                    "cost_sum": 0.0,
                    "count": 0,
                }
            metrics = config_metrics[entry.config_name]
            metrics["accuracy_sum"] += entry.accuracy
            metrics["latency_sum"] += entry.latency_ms
            metrics["cost_sum"] += entry.cost_usd
            metrics["count"] += 1

        # 计算平均值
        averages: dict[str, dict[str, float]] = {}
        for name, metrics in config_metrics.items():
            count = metrics["count"]
            averages[name] = {
                "accuracy": metrics["accuracy_sum"] / count,
                "latency": metrics["latency_sum"] / count,
                "cost": metrics["cost_sum"] / count,
            }

        # 查找帕累托最优解：如果另一配置在所有维度上都更好，则当前配置被支配
        pareto: list[str] = []
        config_names = list(averages.keys())

        for name in config_names:
            dominated = False
            for other_name in config_names:
                if name == other_name:
                    continue
                other = averages[other_name]
                current = averages[name]
                # 如果另一配置准确率更高、延迟更低且成本更低，则它支配当前配置
                if (
                    other["accuracy"] >= current["accuracy"]
                    and other["latency"] <= current["latency"]
                    and other["cost"] <= current["cost"]
                    and (
                        other["accuracy"] > current["accuracy"]
                        or other["latency"] < current["latency"]
                        or other["cost"] < current["cost"]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(name)

        logger.info("帕累托最优配置：%s", pareto)
        return pareto
