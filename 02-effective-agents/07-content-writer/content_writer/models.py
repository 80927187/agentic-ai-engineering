"""
流水线数据和类型化事件系统的 Pydantic 模型。

数据模型用于校验 LLM 的结构化输出。事件模型通过异步生成器实现类型安全的进度跟踪，
入口使用模式匹配，以合适的 Rich 界面渲染每种事件。
"""

from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, Field, computed_field


# ─── 数据模型 ────────────────────────────────────────────────────────────────


class ContentType(str, Enum):
    """流水线可以生成的内容类型。"""

    BLOG = "blog"
    TUTORIAL = "tutorial"
    CONCEPT = "concept"


class ClassificationResult(BaseModel):
    """路由输出：内容类型与主题分析。"""

    content_type: ContentType
    topic: str
    key_aspects: list[str]
    reasoning: str


class Source(BaseModel):
    """研究或修订过程中发现的网络来源。"""

    title: str
    url: str


class Subtopic(BaseModel):
    """研究计划条目：要调查的一个子主题。"""

    title: str
    research_prompt: str


class ResearchSection(BaseModel):
    """研究输出：综合发现的一个章节。"""

    title: str
    content: str
    sources: list[Source] = []


class EvaluationResult(BaseModel):
    """对草稿进行五维质量评估。"""

    clarity: int = Field(ge=1, le=10)
    technical_accuracy: int = Field(ge=1, le=10)
    structure: int = Field(ge=1, le=10)
    engagement: int = Field(ge=1, le=10)
    human_voice: int = Field(ge=1, le=10)
    issues: list[str] = []
    suggestions: list[str] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def avg_score(self) -> float:
        """五个维度的平均分。"""
        return (
            self.clarity
            + self.technical_accuracy
            + self.structure
            + self.engagement
            + self.human_voice
        ) / 5


class SocialContent(BaseModel):
    """扇出输出：三个平台的社交媒体内容。"""

    linkedin: str = ""
    twitter: str = ""
    newsletter: str = ""


class SeoResult(BaseModel):
    """投票输出：候选标题中的最佳 SEO 标题。"""

    winning_title: str
    reasoning: str
    candidates: list[str] = []


class WritingResult(BaseModel):
    """流水线最终输出：文章及可选推广包。"""

    content_type: ContentType
    title: str
    content: str
    final_score: float
    iterations: int
    sources: list[Source] = []
    social: SocialContent | None = None
    seo: SeoResult | None = None


# ─── 类型化事件 ──────────────────────────────────────────────────────────────
# 每个事件都是带有 Literal 阶段字段的 Pydantic 模型。
# 异步生成器产生这些事件，入口使用 match/case 进行渲染。


class ClassifyStartEvent(BaseModel):
    """分类开始时产生。"""

    stage: Literal["classify_start"] = "classify_start"


class ClassifyDoneEvent(BaseModel):
    """分类完成时产生。"""

    stage: Literal["classify_done"] = "classify_done"
    classification: ClassificationResult


class HumanCheckpointEvent(BaseModel):
    """代理需要人工输入时产生。"""

    stage: Literal["human_checkpoint"] = "human_checkpoint"
    checkpoint_id: str
    title: str
    content: str
    question: str


class PlanStartEvent(BaseModel):
    """研究计划开始时产生。"""

    stage: Literal["plan_start"] = "plan_start"


class PlanDoneEvent(BaseModel):
    """研究计划完成时产生。"""

    stage: Literal["plan_done"] = "plan_done"
    subtopics: list[Subtopic]


class ResearchStartEvent(BaseModel):
    """并行研究开始时产生。"""

    stage: Literal["research_start"] = "research_start"
    count: int


class ResearchSectionDoneEvent(BaseModel):
    """一个研究工作器完成时产生。"""

    stage: Literal["research_section_done"] = "research_section_done"
    title: str
    sources: list[Source] = []


class ResearchDoneEvent(BaseModel):
    """所有研究工作器完成时产生。"""

    stage: Literal["research_done"] = "research_done"
    sections: list[ResearchSection]


class WriteStartEvent(BaseModel):
    """写作开始时产生。"""

    stage: Literal["write_start"] = "write_start"
    iteration: int


class WriteDoneEvent(BaseModel):
    """写作完成时产生。"""

    stage: Literal["write_done"] = "write_done"
    iteration: int
    content_length: int
    content: str
    sources: list[Source] = []


class EvaluateStartEvent(BaseModel):
    """评估开始时产生。"""

    stage: Literal["evaluate_start"] = "evaluate_start"
    iteration: int


class EvaluateDoneEvent(BaseModel):
    """评估完成时产生。"""

    stage: Literal["evaluate_done"] = "evaluate_done"
    iteration: int
    evaluation: EvaluationResult


class RefineStartEvent(BaseModel):
    """修订重写开始时产生。"""

    stage: Literal["refine_start"] = "refine_start"
    iteration: int


class SocialStartEvent(BaseModel):
    """社交媒体扇出开始时产生。"""

    stage: Literal["social_start"] = "social_start"


class SocialWriterDoneEvent(BaseModel):
    """一个社交媒体写作者完成时产生。"""

    stage: Literal["social_writer_done"] = "social_writer_done"
    name: str


class SocialDoneEvent(BaseModel):
    """所有社交媒体写作者完成时产生。"""

    stage: Literal["social_done"] = "social_done"
    social: SocialContent


class SeoStartEvent(BaseModel):
    """SEO 标题投票开始时产生。"""

    stage: Literal["seo_start"] = "seo_start"


class SeoCandidateEvent(BaseModel):
    """生成一个 SEO 标题候选时产生。"""

    stage: Literal["seo_candidate"] = "seo_candidate"
    title: str


class SeoDoneEvent(BaseModel):
    """SEO 标题投票完成时产生。"""

    stage: Literal["seo_done"] = "seo_done"
    seo: SeoResult


class CompleteEvent(BaseModel):
    """完整流水线完成时产生。"""

    stage: Literal["complete"] = "complete"
    result: WritingResult


AgentEvent = Union[
    ClassifyStartEvent,
    ClassifyDoneEvent,
    HumanCheckpointEvent,
    PlanStartEvent,
    PlanDoneEvent,
    ResearchStartEvent,
    ResearchSectionDoneEvent,
    ResearchDoneEvent,
    WriteStartEvent,
    WriteDoneEvent,
    EvaluateStartEvent,
    EvaluateDoneEvent,
    RefineStartEvent,
    SocialStartEvent,
    SocialWriterDoneEvent,
    SocialDoneEvent,
    SeoStartEvent,
    SeoCandidateEvent,
    SeoDoneEvent,
    CompleteEvent,
]
