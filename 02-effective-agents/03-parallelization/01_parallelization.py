"""
并行化——“社交媒体推广组合”

演示如何通过扇出处理独立工作，再通过扇入合并结果。
输入一篇博客文章，并行生成社交媒体内容，同时使用投票模式选择 SEO 标题。
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import anthropic
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from common import AnthropicTokenTracker, interactive_menu, setup_logging

load_dotenv(find_dotenv())
logger = setup_logging(__name__)

MODEL = "claude-sonnet-4-6"
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")

ARTICLE_TITLES = {
    "building_agentic_ai_systems": "构建智能体式 AI 系统",
    "practical_async_programming_in_python": "Python 异步编程实践",
    "webassembly_beyond_the_browser": "超越浏览器的 WebAssembly",
}

CONTENT_LABELS = {
    "linkedin": "LINKEDIN 帖子",
    "twitter": "TWITTER/X 帖子串",
    "newsletter": "邮件简报",
    "seo_vote": "最佳 SEO 标题",
}

# --- 提示词 ---

LINKEDIN_SYSTEM_PROMPT = (
    "你是一名 LinkedIn 内容专家。请根据给定的博客文章撰写一篇适合 LinkedIn 发布的专业摘要。"
    "加入相关话题标签，全文不超过 600 个汉字。"
)

TWITTER_SYSTEM_PROMPT = (
    "你是一名 Twitter/X 内容专家。请根据给定的博客文章创建一个恰好包含 5 条推文的帖子串。"
    "每条推文不超过 280 个字符，按 1/5 至 5/5 编号，并让第一条推文具有吸引力。"
)

NEWSLETTER_SYSTEM_PROMPT = (
    "你是一名电子邮件营销专家。请根据给定的博客文章撰写：1）一个引人注目的邮件主题；"
    "2）一段由 2 至 3 句话组成的预览或引言，吸引读者点击阅读全文。"
    "输出格式为‘主题：……’，后接引言。"
)

SEO_TITLE_SYSTEM_PROMPT = (
    "你是一名 SEO 专家。请为这篇博客文章生成且仅生成一个引人注目的 SEO 标题。"
    "标题应包含相关关键词，长度控制在 20 至 30 个汉字，并具有点击吸引力。"
    "只输出标题，不要输出其他内容。"
)

SEO_EVALUATOR_SYSTEM_PROMPT = (
    "你是一名 SEO 评估专家。请根据候选标题和博客摘要选出最佳标题。"
    "评估时考虑关键词相关性、点击吸引力、长度和清晰度。只输出编号和胜出的标题。"
)

# 回调类型：生成器发出（事件名称、事件数据），由调用方决定如何显示
GeneratorCallback = Callable[[str, dict[str, Any]], None]


class ParallelContentGenerator:
    """在多个相互独立的 LLM 调用之间扇出内容生成任务。"""

    def __init__(self, model: str, token_tracker: AnthropicTokenTracker):
        self.client = anthropic.Anthropic()
        self.model = model
        self.token_tracker = token_tracker
        self._notify: GeneratorCallback = lambda _e, _d: None

    def _call_llm(self, system: str, user_message: str, temperature: float = 1.0) -> str:
        """执行一次 LLM 调用。"""
        logger.info("正在调用 %s（温度=%.1f）", self.model, temperature)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        self.token_tracker.track(response.usage)
        return cast(str, response.content[0].text)

    def _write_linkedin(self, blog_post: str) -> str:
        """生成 LinkedIn 专业摘要。"""
        return self._call_llm(
            LINKEDIN_SYSTEM_PROMPT,
            f"根据以下博客文章创建一篇 LinkedIn 帖子：\n\n{blog_post}",
        )

    def _write_twitter(self, blog_post: str) -> str:
        """生成包含 5 条推文的 Twitter/X 帖子串。"""
        return self._call_llm(
            TWITTER_SYSTEM_PROMPT,
            f"根据以下博客文章创建一个推文串：\n\n{blog_post}",
        )

    def _write_newsletter(self, blog_post: str) -> str:
        """生成邮件简报的主题和引言段落。"""
        return self._call_llm(
            NEWSLETTER_SYSTEM_PROMPT,
            f"根据以下博客文章创建邮件简报引言：\n\n{blog_post}",
        )

    def _generate_seo_title(self, blog_post: str, temperature: float) -> str:
        """以给定温度生成一个 SEO 标题候选项。"""
        return self._call_llm(
            SEO_TITLE_SYSTEM_PROMPT,
            f"为以下内容生成一个 SEO 标题：\n\n{blog_post[:500]}",
            temperature=temperature,
        )

    def _vote_best_title(self, titles: list[str], blog_post: str) -> str:
        """使用评估器从候选项中选出最佳 SEO 标题。"""
        titles_text = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
        return self._call_llm(
            SEO_EVALUATOR_SYSTEM_PROMPT,
            f"博客摘要：{blog_post[:300]}\n\n候选标题：\n{titles_text}\n\n哪个 SEO 标题最好？",
        )

    def run(self, blog_post: str, on_event: GeneratorCallback | None = None) -> dict[str, str]:
        """执行完整的并行化流水线。"""
        self._notify = on_event or (lambda _e, _d: None)
        results: dict[str, str] = {}

        # 扇出：并发运行所有写作智能体
        self._notify("fanout_start", {})
        writers = {
            "linkedin": self._write_linkedin,
            "twitter": self._write_twitter,
            "newsletter": self._write_newsletter,
        }

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(fn, blog_post): name for name, fn in writers.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                    self._notify("writer_complete", {"name": name})
                except Exception as e:
                    logger.error("写作智能体 %s 失败：%s", name, e)
                    results[name] = f"错误：{e}"
        self.token_tracker.report()

        # 投票模式：以不同温度生成 3 个 SEO 标题
        self._notify("voting_start", {})
        temperatures = [0.3, 0.7, 1.0]
        titles: list[str] = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures_list = [
                executor.submit(self._generate_seo_title, blog_post, temp) for temp in temperatures
            ]
            for future in as_completed(futures_list):
                try:
                    title = future.result().strip()
                    titles.append(title)
                    self._notify("title_candidate", {"title": title})
                except Exception as e:
                    logger.error("标题生成失败：%s", e)
        self.token_tracker.report()

        # 评估并选出最佳标题
        if titles:
            self._notify("evaluating_start", {})
            results["seo_vote"] = self._vote_best_title(titles, blog_post)
            self.token_tracker.report()

        self._notify("pipeline_complete", {})
        return results


def _load_input_files() -> dict[str, Path]:
    """发现输入目录中的博客文章，并以显示名称作为键。"""
    if not INPUT_DIR.exists():
        return {}
    posts: dict[str, Path] = {}
    for path in sorted(INPUT_DIR.glob("*.md")):
        article_title = ARTICLE_TITLES.get(path.stem, path.stem.replace("_", " ").title())
        label = f"{article_title}  [grey50]({path})[/grey50]"
        posts[label] = path
    return posts


def _clean_label(label: str) -> str:
    """从菜单标签中移除 Rich 标记，获得纯文本文章名称。"""
    return label.split("  [grey50]")[0]


def main() -> None:
    """运行并行化演示。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()
    generator = ParallelContentGenerator(MODEL, token_tracker)

    def on_event(event: str, data: dict[str, Any]) -> None:
        """在控制台中输出流水线进度。"""
        if event == "fanout_start":
            console.print("\n[bold yellow]扇出：[/bold yellow]正在并行生成社交媒体内容……")
        elif event == "writer_complete":
            writer_name = CONTENT_LABELS.get(data["name"], data["name"])
            console.print(f"  [green]✓[/green] {writer_name}已完成")
        elif event == "voting_start":
            console.print("\n[bold yellow]投票：[/bold yellow]正在生成 SEO 标题候选项……")
        elif event == "title_candidate":
            console.print(f"  [dim]• {data['title']}[/dim]")
        elif event == "evaluating_start":
            console.print("\n[bold yellow]评估：[/bold yellow]正在选择最佳 SEO 标题……")

    # 从 input/ 加载预置博客文章
    input_files = _load_input_files()
    labels = list(input_files.keys())

    header = Panel(
        "[bold cyan]并行化——社交媒体推广组合[/bold cyan]\n\n"
        "博客文章 → [LinkedIn 写作智能体] + [Twitter 写作智能体] + [邮件简报写作智能体]\n"
        "         → [聚合器] → 推广组合\n\n"
        "此外：投票模式——以不同温度生成 3 个 SEO 标题 → 评估器选出最佳标题",
        title="并行化",
    )

    try:
        while True:
            choice = interactive_menu(
                console,
                labels,
                title="选择一篇博客文章",
                header=header,
                allow_custom=True,
                custom_label="✏️  粘贴自己的内容……",
                custom_prompt="输入一个简短的博客主题（或粘贴文本）",
            )
            if not choice:
                break

            # 解析博客文章内容
            name = _clean_label(choice) if choice in input_files else choice
            if choice in input_files:
                blog_post = input_files[choice].read_text(encoding="utf-8")
                console.print(f"\n[bold green]博客文章：[/bold green]{name}")
            elif len(choice) < 200:
                # 简短的自定义输入——直接作为主题或短文使用
                blog_post = choice
                console.print(f"\n[bold green]主题：[/bold green]{name}")
            else:
                blog_post = choice
                console.print(f"\n[bold green]自定义文章：[/bold green]（{len(choice)} 个字符）")

            try:
                results = generator.run(blog_post, on_event=on_event)

                # 将推广组合保存到输出目录
                OUTPUT_DIR.mkdir(exist_ok=True)
                slug = name.lower().replace(" ", "_")[:50]
                path = OUTPUT_DIR / f"{slug}_promo.md"
                output_parts = [f"# 推广组合：{name}\n"]
                for key, value in results.items():
                    output_parts.append(f"## {CONTENT_LABELS.get(key, key.upper())}\n\n{value}\n")
                path.write_text("\n".join(output_parts), encoding="utf-8")

                console.print("\n[bold blue]推广组合：[/bold blue]")
                for key, value in results.items():
                    console.print(
                        Panel(
                            Markdown(value),
                            title=CONTENT_LABELS.get(key, key.upper()),
                            border_style="cyan",
                        )
                    )

                abs_path = path.resolve()
                console.print(f"\n[dim]已保存至 [link=file://{abs_path}]{path}[/link][/dim]")

                console.print("\n[dim]按 Enter 键继续……[/dim]")
                input()
            except Exception as e:
                logger.error("并行化失败：%s", e)
                console.print(f"\n[red]错误：{e}[/red]")
            finally:
                token_tracker.report()
                token_tracker.reset()

    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")


if __name__ == "__main__":
    main()
