"""
基准测试脚本共享的数据类和模型配置。

定义 ModelConfig、BenchmarkResult、BenchmarkConfig 和默认模型配置。
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """待基准测试模型的配置。"""

    name: str
    provider: str  # "anthropic" 或 "openai"
    model_id: str
    cost_per_input_token: float  # 每 100 万输入 Token 的美元价格
    cost_per_output_token: float  # 每 100 万输出 Token 的美元价格


@dataclass
class BenchmarkResult:
    """单个基准测试任务的运行结果。"""

    task_id: str
    config_name: str
    answer: str
    keyword_score: float  # 根据命中的预期关键词计算，范围为 0.0～1.0
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tool_calls: int


@dataclass
class BenchmarkConfig:
    """单个基准测试配置（模型与提示词的组合）。"""

    name: str
    model: ModelConfig
    prompt_strategy: str
    system_prompt: str


# DeepSeek 模型常量（保留两个层级名称，便于课程示例对照）。
MODEL = "deepseek-v4-flash"
PRO_MODEL = "deepseek-v4-pro"

# 默认模型配置
MODEL_CONFIGS = [
    ModelConfig("DeepSeek Flash", "anthropic", MODEL, 0.14, 0.28),
    ModelConfig("DeepSeek Pro", "anthropic", PRO_MODEL, 0.55, 2.19),
    ModelConfig("GPT-4.1 mini", "openai", "gpt-4.1-mini", 0.40, 1.60),
]
