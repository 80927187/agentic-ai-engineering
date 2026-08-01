"""
评估器—优化器：“编辑工作台”

演示由一个大语言模型生成内容，另一个大语言模型在循环中进行评估和改进，
直到达到质量阈值。生成器与评估器使用目标不同的提示词。

流水线：调研（网页搜索）→ 写作（不使用工具）→ 评估 → 改进循环
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import anthropic
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from common import AnthropicTokenTracker, interactive_menu, setup_logging

load_dotenv(find_dotenv())
logger = setup_logging(__name__)

OUTPUT_DIR = Path("output")
MODEL = "claude-sonnet-4-6"
LIGHT_MODEL = "claude-haiku-4-5-20251001"

# Anthropic 服务端网页搜索工具——由 Claude 决定何时搜索
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 1}

SUGGESTED_TOPICS = [
    "构建事件驱动的微服务",
    "用于实时 AI 的边缘计算",
    "现代 CSS 布局技术",
    "数据库分片策略",
]

# --- 提示词 ---

RESEARCH_SYSTEM_PROMPT = (
    "你是一名技术研究员。使用网页搜索查找有关该主题的最新准确信息。"
    "用 2～3 个短段落综合最相关的发现。重点介绍实用细节、取舍和真实场景中的模式。"
    "不要写开场白。"
)

WRITER_SYSTEM_PROMPT = (
    "你是一名技术博客作者。请根据调研笔记撰写一篇简洁的中文博客文章。"
    "文章应包含引言、3～5 个带有明确标题的章节、适当的代码示例和结语。"
    "语气应专业且平易近人。全文尽量控制在 1500 个汉字以内，不要堆砌无用内容。"
)

REFINER_SYSTEM_PROMPT = (
    "你是一名技术博客作者。请根据提供的反馈修改中文草稿。处理每一个问题和建议。"
    "保留整体结构，同时提高文章质量。返回完整的修订后文章。"
)

EVALUATOR_SYSTEM_PROMPT = """\
你是一名要求严格的技术编辑。请从以下方面为内容打 1～10 分：

1. 清晰度：工程师能否无需反复阅读就理解内容？\
（9～10 分：极其清晰；7～8 分：少数地方不够顺畅；5～6 分：理解起来比较费力）
2. 技术准确性：信息是否正确且符合现状？\
（9～10 分：可用于生产实践；7～8 分：存在少量不够精确之处）
3. 结构：逻辑是否流畅，内容是否易于浏览？\
（9～10 分：层层递进，便于快速阅读）
4. 吸引力：工程师是否愿意阅读？\
（9～10 分：引人入胜，令人印象深刻）
5. 自然表达：读起来是否像真人所写？\
（9～10 分：自然且节奏富于变化；5～6 分：机械或千篇一律）

反馈必须具体，例如“引言过于宽泛——请从具体问题切入”，而不能只说“写得更吸引人”。"""

# 五维结构化评估输出
EVALUATION_TOOLS = [
    {
        "name": "evaluate_draft",
        "description": "从多个质量维度评估博客文章草稿。",
        "input_schema": {
            "type": "object",
            "properties": {
                "clarity": {"type": "integer", "minimum": 1, "maximum": 10},
                "technical_accuracy": {"type": "integer", "minimum": 1, "maximum": 10},
                "structure": {"type": "integer", "minimum": 1, "maximum": 10},
                "engagement": {"type": "integer", "minimum": 1, "maximum": 10},
                "human_voice": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "读起来是否像真人所写？",
                },
                "issues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "发现的具体问题",
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "可执行的改进建议",
                },
            },
            "required": [
                "clarity",
                "technical_accuracy",
                "structure",
                "engagement",
                "human_voice",
                "issues",
                "suggestions",
            ],
        },
    }
]

SCORE_THRESHOLD = 7.0
MAX_REFINEMENTS = 2

# 回调类型：智能体发出（事件名称、事件数据），由调用方决定如何显示
EvaluatorCallback = Callable[[str, dict[str, Any]], None]

SCORE_DIMENSIONS = ["清晰度", "技术准确性", "结构", "吸引力", "自然表达"]
SCORE_KEYS = ["clarity", "technical_accuracy", "structure", "engagement", "human_voice"]


def _extract_scores(evaluation: dict[str, Any]) -> tuple[dict[str, int], float]:
    """提取各维度分数并计算平均分。"""
    scores = dict(zip(SCORE_DIMENSIONS, (evaluation[k] for k in SCORE_KEYS)))
    return scores, sum(scores.values()) / len(scores)


class EvaluatorOptimizer:
    """执行调研 → 写作 → 评估 → 改进循环，直到达到质量阈值。"""

    def __init__(self, model: str, light_model: str, token_tracker: AnthropicTokenTracker):
        self.client = anthropic.Anthropic()
        self.model = model
        self.light_model = light_model
        self.token_tracker = token_tracker
        self._notify: EvaluatorCallback = lambda _e, _d: None

    def _call_llm(
        self,
        system: str,
        messages: list[dict[str, Any]],
        *,
        use_light: bool = False,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, str] | None = None,
    ) -> anthropic.types.Message:
        """调用一次大语言模型并跟踪令牌用量。"""
        model = self.light_model if use_light else self.model
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        tool_names = [t.get("name", t.get("type", "unknown")) for t in tools or []]
        logger.info("Calling %s, tools=%s", model, tool_names)

        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            **kwargs,
        )
        self.token_tracker.track(response.usage)
        return response

    def _call_llm_text(self, system: str, user_message: str, **kwargs: Any) -> str:
        """调用大语言模型并返回文本内容。"""
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        return cast(str, self._call_llm(system, messages, **kwargs).content[0].text)

    def _research(self, topic: str) -> str:
        """调研阶段：通过网页搜索收集该主题的最新资料。"""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"调研以下主题：{topic}"}
        ]
        response = self._call_llm(
            RESEARCH_SYSTEM_PROMPT,
            messages,
            use_light=True,
            max_tokens=1024,
            tools=[WEB_SEARCH_TOOL],
        )
        text_parts = [block.text for block in response.content if block.type == "text"]
        return "\n\n".join(text_parts)

    def _write(self, topic: str, research: str) -> str:
        """写作阶段：整合调研资料，不使用工具或网页搜索。"""
        user_msg = f"调研资料：\n{research}\n\n围绕以下主题撰写一篇博客文章：{topic}"
        return self._call_llm_text(WRITER_SYSTEM_PROMPT, user_msg, use_light=True)

    def _refine(self, topic: str, draft: str, research: str, evaluation: dict[str, Any]) -> str:
        """改进阶段：根据反馈重写，不使用工具。"""
        feedback = (
            f"问题：{json.dumps(evaluation['issues'], ensure_ascii=False)}\n"
            f"建议：{json.dumps(evaluation['suggestions'], ensure_ascii=False)}"
        )
        user_msg = (
            f"主题：{topic}\n\n"
            f"调研资料：\n{research}\n\n"
            f"需要处理的反馈：\n{feedback}\n\n"
            f"上一版草稿：\n{draft}\n\n"
            "修改草稿，处理所有反馈。"
        )
        return self._call_llm_text(REFINER_SYSTEM_PROMPT, user_msg)

    def _evaluate(self, draft: str, topic: str) -> dict[str, Any]:
        """评估器：从五个维度为草稿评分并提供反馈。"""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"主题：{topic}\n\n待评估的草稿：\n\n{draft}"}
        ]
        response = self._call_llm(
            EVALUATOR_SYSTEM_PROMPT,
            messages,
            use_light=True,
            max_tokens=1024,
            tools=EVALUATION_TOOLS,
            tool_choice={"type": "tool", "name": "evaluate_draft"},
        )

        for block in response.content:
            if block.type == "tool_use":
                return cast(dict[str, Any], block.input)

        raise ValueError("评估器未返回结构化评估结果")

    def run(self, topic: str, on_event: EvaluatorCallback | None = None) -> str:
        """运行完整流水线：调研 → 写作 → 评估 → 改进循环。"""
        self._notify = on_event or (lambda _e, _d: None)

        # 第 1 步：调研（通过网页搜索收集资料）
        self._notify("research_start", {})
        research = self._research(topic)
        self.token_tracker.report()
        self._notify("research_complete", {"chars": len(research)})

        # 第 2 步：写作（根据调研资料写作，不使用工具）
        self._notify("write_start", {})
        draft = self._write(topic, research)
        self._notify("draft_complete", {"chars": len(draft)})

        # 第 3 步：评估 → 改进循环
        for iteration in range(1, MAX_REFINEMENTS + 1):
            self._notify("evaluate_start", {"iteration": iteration})
            evaluation = self._evaluate(draft, topic)
            scores, avg_score = _extract_scores(evaluation)
            self.token_tracker.report()

            self._notify(
                "evaluation_complete",
                {
                    "iteration": iteration,
                    "scores": scores,
                    "avg": avg_score,
                    "issues": evaluation.get("issues", []),
                    "suggestions": evaluation.get("suggestions", []),
                },
            )

            if avg_score >= SCORE_THRESHOLD:
                self._notify("threshold_met", {"avg": avg_score})
                break

            if iteration < MAX_REFINEMENTS:
                self._notify("refining", {"avg": avg_score})
                draft = self._refine(topic, draft, research, evaluation)
                self._notify("draft_complete", {"chars": len(draft)})
            else:
                self._notify("max_iterations", {"avg": avg_score})

        return draft


def main() -> None:
    """运行评估器—优化器演示。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()

    def on_event(event: str, data: dict[str, Any]) -> None:
        """处理流水线事件并在终端中显示。"""
        if event == "research_start":
            console.print("\n[bold yellow]正在调研：[/bold yellow] 正在收集最新资料……")
        elif event == "research_complete":
            console.print(f"  [green]✓[/green] 调研完成：{data['chars']} 个字符")
        elif event == "write_start":
            console.print("\n[bold yellow]正在写作：[/bold yellow] 正在生成初稿……")
        elif event == "draft_complete":
            console.print(f"  [green]✓[/green] 草稿：{data['chars']} 个字符")
        elif event == "evaluate_start":
            console.print(
                f"\n[bold yellow]正在评估：[/bold yellow] 第 {data['iteration']} 轮"
                f"/{MAX_REFINEMENTS}..."
            )
        elif event == "evaluation_complete":
            scores = data["scores"]
            avg = data["avg"]
            table = Table(title=f"评估结果（平均分：{avg:.1f}/10）")
            table.add_column("维度", style="cyan")
            table.add_column("分数", justify="center")
            for dim, score in scores.items():
                color = "green" if score >= 8 else "yellow" if score >= 6 else "red"
                table.add_row(dim, f"[{color}]{score}/10[/{color}]")
            console.print(table)
            if data["issues"]:
                console.print("[bold red]问题：[/bold red]")
                for issue in data["issues"]:
                    console.print(f"  [red]•[/red] {issue}")
            if data.get("suggestions"):
                console.print("[bold yellow]建议：[/bold yellow]")
                for suggestion in data["suggestions"]:
                    console.print(f"  [yellow]•[/yellow] {suggestion}")
        elif event == "threshold_met":
            console.print(f"\n[green]得分 {data['avg']:.1f} >= {SCORE_THRESHOLD}——完成！[/green]")
        elif event == "refining":
            console.print(
                f"[yellow]得分 {data['avg']:.1f} < {SCORE_THRESHOLD}——正在改进……[/yellow]"
            )
        elif event == "max_iterations":
            console.print(f"\n[yellow]已达到最大迭代次数（得分：{data['avg']:.1f}）[/yellow]")

    header = Panel(
        "[bold cyan]评估器—优化器：编辑工作台[/bold cyan]\n\n"
        "主题 → [研究员] → 资料\n"
        "     → [作者] → 草稿（根据调研资料写作，不进行网页搜索）\n"
        "     → [评估器] → 五维评分 + 反馈\n"
        f"     → 得分 >= {SCORE_THRESHOLD}？→ 完成\n"
        "     → 未达到阈值 → [改进器] → 循环\n\n"
        f"最多改进 {MAX_REFINEMENTS} 次。"
        "评分维度：清晰度、准确性、结构、吸引力、自然表达（1～10 分）",
        title="评估器—优化器",
    )

    try:
        while True:
            topic = interactive_menu(
                console,
                SUGGESTED_TOPICS,
                title="选择主题",
                header=header,
                allow_custom=True,
                custom_prompt="输入你的主题",
            )
            if not topic:
                break

            console.print(f"\n[bold green]主题：[/bold green] {topic}")
            eo = EvaluatorOptimizer(MODEL, LIGHT_MODEL, token_tracker)

            try:
                result = eo.run(topic, on_event=on_event)

                OUTPUT_DIR.mkdir(exist_ok=True)
                slug = topic.lower().replace(" ", "_")[:50]
                path = OUTPUT_DIR / f"{slug}.md"
                path.write_text(result, encoding="utf-8")

                console.print("\n[bold blue]最终文章：[/bold blue]")
                console.print(Markdown(result))
                abs_path = path.resolve()
                console.print(f"\n[dim]已保存至 [link=file://{abs_path}]{path}[/link][/dim]")

                console.print("\n[dim]按 Enter 键继续……[/dim]")
                input()
            except Exception as e:
                logger.error("评估器—优化器运行失败：%s", e)
                console.print(f"\n[red]错误：{e}[/red]")
            finally:
                token_tracker.reset()

    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")


if __name__ == "__main__":
    main()
