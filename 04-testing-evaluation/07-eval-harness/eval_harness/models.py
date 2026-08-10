"""评测工具流水线使用的 Pydantic 数据模型。"""

from pydantic import BaseModel, Field


class EvalTask(BaseModel):
    """黄金数据集中的单项评测任务。"""

    id: str
    question: str
    expected_keywords: list[str] = Field(default_factory=list)
    expected_source_ids: list[str] = Field(default_factory=list)
    difficulty: str = "medium"
    category: str = "general"


class TraceSpan(BaseModel):
    """执行追踪中的单个 span。"""

    name: str
    span_type: str
    start_time: float
    end_time: float = 0.0
    tokens: dict[str, int] = Field(default_factory=dict)
    children: list["TraceSpan"] = Field(default_factory=list)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        """以毫秒为单位的持续时间。"""
        return (self.end_time - self.start_time) * 1000 if self.end_time else 0.0


class GraderScore(BaseModel):
    """单个评分器给出的分数。"""

    grader_name: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class EvalTrial(BaseModel):
    """智能体针对一项任务的一次执行。"""

    task_id: str
    trial_number: int = 1
    answer: str = ""
    tool_calls: list[dict] = Field(default_factory=list)
    trace: list[TraceSpan] = Field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class EvalResult(BaseModel):
    """一项任务的汇总结果。"""

    task_id: str
    trials: list[EvalTrial] = Field(default_factory=list)
    grader_scores: list[GraderScore] = Field(default_factory=list)
    pass_rate: float = 0.0
    avg_score: float = 0.0


class SafetyResult(BaseModel):
    """安全/红队测试的结果。"""

    attack_id: str
    attack_name: str
    category: str
    blocked: bool
    severity: str = "medium"
    details: str = ""


class BenchmarkEntry(BaseModel):
    """一条基准测试测量结果。"""

    config_name: str
    task_id: str
    accuracy: float = 0.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tokens: int = 0


class EvalReport(BaseModel):
    """完整的评测报告。"""

    agent_name: str = "研究助手"
    eval_results: list[EvalResult] = Field(default_factory=list)
    safety_results: list[SafetyResult] = Field(default_factory=list)
    benchmark_entries: list[BenchmarkEntry] = Field(default_factory=list)
    overall_pass_rate: float = 0.0
    overall_safety_score: float = 0.0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
