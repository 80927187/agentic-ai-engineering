"""
完整代理——“内容写作者”

将本模块的全部模式组合为生产级内容创作流水线：
- 路由（03）：内容类型分类 → 类型专属提示词
- 提示词链（02）：研究 → 以类型专属语气写作
- 编排器-工作器（05）：动态研究规划 → 并行研究
- 并行化（04）：社交媒体扇出 + SEO 标题投票
- 评估器-优化器（06）：写作-评估-改进循环与质量门槛
- 人在回路（07）：在关键决策点设置战略检查点
"""

import asyncio
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from common import AnthropicTokenTracker, interactive_menu, setup_logging
from content_writer import (
    ClassifyDoneEvent,
    ClassifyStartEvent,
    CompleteEvent,
    ContentWriterAgent,
    EvaluateDoneEvent,
    EvaluateStartEvent,
    EvaluationResult,
    HumanCheckpointEvent,
    PlanDoneEvent,
    PlanStartEvent,
    RefineStartEvent,
    ResearchDoneEvent,
    ResearchSectionDoneEvent,
    ResearchStartEvent,
    SeoCandidateEvent,
    SeoDoneEvent,
    SeoResult,
    SeoStartEvent,
    SocialContent,
    SocialDoneEvent,
    SocialStartEvent,
    SocialWriterDoneEvent,
    Source,
    WriteDoneEvent,
    WriteStartEvent,
    WritingResult,
)

load_dotenv(find_dotenv())
logger = setup_logging(__name__)

MODEL = "deepseek-v4-flash"
RESEARCH_MODEL = "deepseek-v4-flash" # 低级一些的模型，因为websearch的成本比较高
OUTPUT_DIR = Path("output")
SCORE_THRESHOLD = 7.0
MAX_REFINEMENTS = 2

SUGGESTED_TOPICS = [
    "为什么每个后端团队都应该尝试功能开关",
    "如何使用 Python 和 Click 构建 CLI 工具",
    "什么是向量数据库，它们为什么重要",
    "结构化并发改变了我对异步的看法",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _topic_dir(topic: str) -> Path:
    """创建并返回每个主题专属的输出目录。"""
    slug = topic.lower().replace(" ", "_")[:50]
    path = OUTPUT_DIR / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_artifact(topic: str, filename: str, content: str) -> Path:
    """将单个产物文件保存到主题目录。"""
    path = _topic_dir(topic) / filename
    path.write_text(content, encoding="utf-8")
    logger.info("已保存：%s（%d 个字符）", path, len(content))
    return path


def _save_social(topic: str, social: SocialContent) -> list[Path]:
    """将每个社交媒体产物分别保存为文件。"""
    paths: list[Path] = []
    for name, content in [
        ("linkedin.md", social.linkedin),
        ("twitter.md", social.twitter),
        ("newsletter.md", social.newsletter),
    ]:
        if content and not content.startswith("Error:"):
            paths.append(_save_artifact(topic, name, content))
    return paths


def _save_seo(topic: str, seo: SeoResult) -> Path:
    """保存 SEO 投票结果。"""
    parts = [f"# SEO 标题\n\n{seo.winning_title}\n\n"]
    if seo.candidates:
        parts.append("## 候选标题\n\n")
        for i, c in enumerate(seo.candidates, 1):
            parts.append(f"{i}. {c}\n")
        parts.append(f"\n## 选择理由\n\n{seo.reasoning}\n")
    return _save_artifact(topic, "seo.md", "".join(parts))


def _display_evaluation(console: Console, evaluation: EvaluationResult, iteration: int) -> None:
    """在 Rich 表格中显示五维评估分数。"""
    dimensions = {
        "清晰度": evaluation.clarity,
        "技术准确性": evaluation.technical_accuracy,
        "结构": evaluation.structure,
        "吸引力": evaluation.engagement,
        "人类语气": evaluation.human_voice,
    }

    table = Table(title=f"第 {iteration} 轮——平均分：{evaluation.avg_score:.1f}/10")
    table.add_column("维度", style="cyan")
    table.add_column("分数", justify="center")
    for dim, score in dimensions.items():
        color = "green" if score >= 8 else "yellow" if score >= 6 else "red"
        table.add_row(dim, f"[{color}]{score}/10[/{color}]")
    console.print(table)

    if evaluation.issues:
        console.print("[bold red]问题：[/bold red]")
        for issue in evaluation.issues:
            console.print(f"  [red]•[/red] {issue}")

    if evaluation.suggestions:
        console.print("[bold yellow]建议：[/bold yellow]")
        for suggestion in evaluation.suggestions:
            console.print(f"  [yellow]•[/yellow] {suggestion}")


def _print_path(console: Console, label: str, path: Path) -> None:
    """打印可点击的文件链接。"""
    console.print(f"  [dim]{label}: [link=file://{path.resolve()}]{path}[/link][/dim]")


def _show_sources(console: Console, sources: list[Source]) -> None:
    """在面板中显示可点击的网络搜索来源。"""
    if not sources:
        return
    lines = [f"  [dim]•[/dim] [link={s.url}]{s.title}[/link]" for s in sources]
    console.print(Panel("\n".join(lines), title="来源", border_style="dim"))


def _human_checkpoint(console: Console, event: HumanCheckpointEvent) -> tuple[bool, str]:
    """在关键决策点暂停，等待人工审核。"""
    console.print(
        Panel(event.content, title=f"检查点：{event.title}", border_style="bright_magenta")
    )
    console.print(f"\n[bold magenta]{event.question}[/bold magenta]")
    console.print("[dim](y)是 / (n)否并提供反馈[/dim]")
    console.print("[bold magenta]> [/bold magenta]", end="")

    response = input().strip().lower()
    if response in ["y", "yes", ""]:
        return True, ""

    console.print("[dim]反馈：[/dim] ", end="")
    feedback = input().strip() if response == "n" else response
    return False, feedback


# ─── Event Consumer ──────────────────────────────────────────────────────────


async def _run_with_events(
    agent: ContentWriterAgent,
    topic: str,
    console: Console,
    tracker: AnthropicTokenTracker,
) -> WritingResult | None:
    """消费代理产生的带类型事件，并使用 Rich 渲染。"""
    state: dict[str, Path | None] = {"last_draft_path": None}

    def on_checkpoint(event: HumanCheckpointEvent) -> tuple[bool, str]:
        draft_path = state["last_draft_path"]
        if draft_path and event.checkpoint_id == "final_review":
            _print_path(console, "Latest draft", draft_path)
        return _human_checkpoint(console, event)

    result: WritingResult | None = None

    async for event in agent.run_stream(
        topic,
        score_threshold=SCORE_THRESHOLD,
        max_refinements=MAX_REFINEMENTS,
        on_human_checkpoint=on_checkpoint,
    ):
        match event:
            # 阶段 1：分类
            case ClassifyStartEvent():
                console.print("\n[bold yellow]阶段 1：[/bold yellow]正在分类内容类型……")

            case ClassifyDoneEvent(classification=c):
                console.print(f"  [green]✓[/green] {c.content_type.value}: {c.topic}")
                tracker.report()

            # 阶段 2：研究规划
            case PlanStartEvent():
                console.print("\n[bold yellow]阶段 2：[/bold yellow]正在规划研究……")

            case PlanDoneEvent(subtopics=subs):
                for i, s in enumerate(subs, 1):
                    console.print(f"  {i}. [bold]{s.title}[/bold]")
                tracker.report()

            # 阶段 3：并行研究
            case ResearchStartEvent(count=n):
                console.print(
                    f"\n[bold yellow]阶段 3：[/bold yellow]"
                    f"正在并行研究 {n} 个子主题……"
                )

            case ResearchSectionDoneEvent(title=t, sources=srcs):
                console.print(f"  [green]✓[/green] {t}")
                _show_sources(console, srcs)

            case ResearchDoneEvent():
                tracker.report()

            # 阶段 4：写作
            case WriteStartEvent(iteration=i):
                label = "写作" if i == 1 else f"重写（第 {i - 1} 轮）"
                console.print(f"\n[bold yellow]阶段 4：[/bold yellow]{label}……")

            case WriteDoneEvent(iteration=i, content_length=length, content=draft, sources=srcs):
                path = _save_artifact(topic, f"draft_v{i}.md", draft)
                state["last_draft_path"] = path
                console.print(
                    f"  [green]✓[/green] v{i}: {length:,} chars — "
                    f"[dim][link=file://{path.resolve()}]{path}[/link][/dim]"
                )
                _show_sources(console, srcs)

            # 阶段 5：评估与改进
            case EvaluateStartEvent(iteration=i):
                console.print(f"\n[bold yellow]阶段 5：[/bold yellow]正在评估（第 {i} 轮）……")

            case EvaluateDoneEvent(iteration=i, evaluation=e):
                _display_evaluation(console, e, i)
                tracker.report()

                if e.avg_score >= SCORE_THRESHOLD:
                    console.print(
                        f"\n[green]得分 {e.avg_score:.1f} >= {SCORE_THRESHOLD}"
                        f"——质量达标！[/green]"
                    )
                else:
                    console.print(f"\n[yellow]得分 {e.avg_score:.1f} < {SCORE_THRESHOLD}[/yellow]")

            case RefineStartEvent(iteration=i):
                console.print(f"\n[yellow]正在改进（第 {i - 1}/{MAX_REFINEMENTS} 轮）……[/yellow]")

            # 阶段 6：社交媒体
            case SocialStartEvent():
                console.print(
                    "\n[bold yellow]阶段 6：[/bold yellow]社交媒体分发（扇出）……"
                )

            case SocialWriterDoneEvent(name=n):
                console.print(f"  [green]✓[/green] {n}")

            case SocialDoneEvent(social=s):
                tracker.report()
                paths = _save_social(topic, s)
                for p in paths:
                    _print_path(console, p.stem, p)
                for key, content in [
                    ("LINKEDIN", s.linkedin),
                    ("TWITTER", s.twitter),
                    ("NEWSLETTER", s.newsletter),
                ]:
                    if content and not content.startswith("Error:"):
                        console.print(Panel(Markdown(content), title=key, border_style="cyan"))

            # 阶段 7：SEO 标题投票
            case SeoStartEvent():
                console.print("\n[bold yellow]阶段 7：[/bold yellow]SEO 标题投票……")

            case SeoCandidateEvent(title=t):
                console.print(f"  [dim]• {t}[/dim]")

            case SeoDoneEvent(seo=s):
                console.print(f"  [green]✓[/green] 获胜标题：{s.winning_title}")
                console.print(f"  [dim]{s.reasoning}[/dim]")
                tracker.report()
                path = _save_seo(topic, s)
                _print_path(console, "seo", path)

            # 流水线完成
            case CompleteEvent(result=r):
                result = r

    return result


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    """运行完整的内容写作代理。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()
    agent = ContentWriterAgent(MODEL, RESEARCH_MODEL, token_tracker)

    header = Panel(
        "[bold cyan]完整代理——内容写作者[/bold cyan]\n\n"
        "将本模块的全部模式组合为一个流水线：\n"
        "  [分类] → [规划] → [研究] → [写作] → [评估] → [改进]\n"
        "  → [人工审核] → [社交媒体分发] → [SEO 标题投票]\n\n"
        "模式：\n"
        "  路由（03）| 提示词链（02）| 并行化（04）\n"
        "  编排器-工作器（05）| 评估器-优化器（06）| 人在回路（07）",
        title="内容写作者",
    )

    async def async_main() -> None:
        while True:
            topic = interactive_menu(
                console,
                SUGGESTED_TOPICS,
                title="选择主题",
                header=header,
                allow_custom=True,
                custom_prompt="请输入主题",
            )
            if not topic:
                break

            console.print(f"\n[bold green]主题：[/bold green] {topic}")

            try:
                result = await _run_with_events(agent, topic, console, token_tracker)

                if result:
                    # Save final article
                    article_path = _save_artifact(topic, "article.md", result.content)
                    _print_path(console, "最终文章", article_path)

                    # Show final article
                    console.print("\n[bold blue]最终文章：[/bold blue]")
                    console.print(Markdown(result.content))

                    # Show output directory
                    topic_dir = _topic_dir(topic)
                    console.print(
                        f"\n[dim]全部产物："
                        f"[link=file://{topic_dir.resolve()}]{topic_dir}/[/link][/dim]"
                    )

                console.print("\n[dim]按 Enter 继续……[/dim]")
                input()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error("流水线失败：%s", e)
                console.print(f"\n[red]错误：{e}[/red]")
            finally:
                token_tracker.report()
                token_tracker.reset()

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")
        os._exit(130)


if __name__ == "__main__":
    main()
