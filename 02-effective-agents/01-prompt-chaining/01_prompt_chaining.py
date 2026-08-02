"""
提示词链——“科技博客流水线”

演示如何将任务拆分为一系列固定步骤，其中每次 LLM 调用都会处理上一步的输出。
一个主题将依次经过大纲规划器 → 作者 → 编辑。
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from common import AnthropicTokenTracker, interactive_menu, setup_logging

load_dotenv(find_dotenv())
logger = setup_logging(__name__)

OUTPUT_DIR = Path("output")
MODEL = "deepseek-v4-flash"
LIGHT_MODEL = "deepseek-v4-flash"

# Anthropic 服务端网络搜索工具——Claude 自行决定如何搜索
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 10}

SUGGESTED_TOPICS = [
    "Python 异步编程实践",
    "生产环境中的 AI 智能体",
    "浏览器之外的 WebAssembly",
    "面向初创公司的零信任安全",
]


# --- 提示词 ---

OUTLINER_SYSTEM_PROMPT = (
    "你是一名研究规划师。给定一个主题后，找出 3～5 个宽泛的研究方向，"
    "涵盖该主题的不同维度，例如市场格局、技术深度、采用模式和性能分析等。"
    "每个方向都应当可以独立研究。第一行输出主题标题，然后以项目符号列出研究方向。"
    "方向名称应简短且具有概括性，不要添加额外说明。"
)
OUTLINER_USER_PROMPT = "为以下主题创建博客大纲：{topic}"

WRITER_SYSTEM_PROMPT = (
    "你是一名技术博客作者。根据给定的大纲（标题和项目符号列表），撰写一篇简洁的博客文章。"
    "使用标题作为 H1 标题，并将每个项目符号作为一个 H2 小节。"
    "每节写 1～2 个短段落，不要填充无关内容或空话。语气应专业但平易近人，"
    "全文尽量控制在 1000 字以内。始终使用网络搜索，以当前且准确的信息为写作提供事实依据。"
)
WRITER_USER_PROMPT = "根据以下大纲撰写一篇完整的博客文章：\n\n{outline}"

EDITOR_SYSTEM_PROMPT = (
    "你是一名专业编辑。润色给定博客文章的语法、清晰度和行文流畅度。"
    "在文章末尾添加“## 核心要点”一节，用 3～5 个项目符号概括主要观点。"
    "返回经过编辑的完整文章。"
)
EDITOR_USER_PROMPT = "编辑并润色以下博客文章：\n\n{draft}"

# 回调类型：智能体发出 (event_name, event_data)，由调用方决定如何显示
ChainCallback = Callable[[str, dict[str, Any]], None]


class PromptChain:
    """顺序执行的 LLM 调用链，每一步的输出都会传给下一步。"""

    def __init__(self, model: str, light_model: str, token_tracker: AnthropicTokenTracker):
        self.client = anthropic.Anthropic()
        self.model = model
        self.light_model = light_model
        self.token_tracker = token_tracker
        self._notify: ChainCallback = lambda _e, _d: None

    def _call_llm(
        self,
        system: str,
        messages: list[dict[str, Any]],
        *,
        use_light: bool = False,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> anthropic.types.Message:
        """执行一次 LLM 调用并追踪 token。"""
        model = self.light_model if use_light else self.model
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
        tool_names = [t.get("name", t.get("type", "unknown")) for t in tools or []]
        logger.info("正在调用 %s，工具=%s", model, tool_names)

        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            **kwargs,
        )
        self.token_tracker.track(response.usage)
        return response

    def _call_llm_text(self, system: str, user_message: str) -> str:
        """调用 LLM 并返回文本内容。"""
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        response = self._call_llm(system, messages)
        text_parts = [block.text for block in response.content if block.type == "text"]
        if not text_parts:
            raise ValueError("模型响应中没有文本内容。")
        return "\n\n".join(text_parts)

    def _run_agentic_loop(
        self,
        system: str,
        user_message: str,
        *,
        use_light: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, str]]]:
        """运行可使用工具的 LLM，并在需要时持续进行多轮对话。"""
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        searches: list[dict[str, str]] = []

        response = self._call_llm(system, messages, use_light=use_light, tools=tools)

        for _ in range(4):
            # 收集本轮的搜索结果
            for block in response.content:
                if block.type == "web_search_tool_result" and isinstance(block.content, list):
                    for result in block.content:
                        searches.append({"title": result.title, "url": result.url})

            if response.stop_reason == "end_turn":
                break

            # 携带工具结果继续对话
            messages.append({"role": "assistant", "content": response.content})
            tool_results = [
                {"type": "tool_result", "tool_use_id": b.id, "content": "搜索已完成。"}
                for b in response.content
                if b.type == "tool_use"
            ]
            if not tool_results:
                break
            messages.append({"role": "user", "content": tool_results})

            response = self._call_llm(system, messages, use_light=use_light, tools=tools)

        text_parts = [b.text for b in response.content if b.type == "text"]
        return "\n\n".join(text_parts), searches

    def _step_outline(self, topic: str) -> str:
        """第 1 步：生成包含标题和项目符号列表的结构化大纲。"""
        return self._call_llm_text(OUTLINER_SYSTEM_PROMPT, OUTLINER_USER_PROMPT.format(topic=topic))

    def _step_write(self, outline: str) -> tuple[str, list[dict[str, str]]]:
        """第 2 步：将大纲扩展为完整博客文章，并可使用网络搜索。"""
        return self._run_agentic_loop(
            WRITER_SYSTEM_PROMPT,
            WRITER_USER_PROMPT.format(outline=outline),
            use_light=True,
            tools=[WEB_SEARCH_TOOL],
        )

    def _step_edit(self, draft: str) -> str:
        """第 3 步：润色初稿并添加“核心要点”一节。"""
        return self._call_llm_text(EDITOR_SYSTEM_PROMPT, EDITOR_USER_PROMPT.format(draft=draft))

    def run(self, topic: str, on_event: ChainCallback | None = None) -> str:
        """执行完整的提示词链：规划大纲 → 写作 → 编辑。"""
        self._notify = on_event or (lambda _e, _d: None)

        # 第 1 步：规划大纲
        self._notify("step_start", {"name": "规划大纲"})
        outline = self._step_outline(topic)
        if not outline.strip():
            raise ValueError("大纲规划器生成了空内容，正在中止提示词链。")
        self.token_tracker.report()
        self._notify("step_complete", {"name": "规划大纲", "result": outline})

        # 第 2 步：写作
        self._notify("step_start", {"name": "写作"})
        logger.info("[写作] 正在调用 %s", self.light_model)
        draft, searches = self._step_write(outline)
        self.token_tracker.report()
        self._notify(
            "step_complete",
            {"name": "写作", "result": draft, "searches": searches},
        )

        # 第 3 步：编辑
        self._notify("step_start", {"name": "编辑"})
        final = self._step_edit(draft)
        self.token_tracker.report()
        self._notify("step_complete", {"name": "编辑"})

        self._notify("chain_complete", {})
        return final


def main() -> None:
    """运行提示词链演示。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()

    def on_chain_event(event: str, data: dict[str, Any]) -> None:
        """在控制台中显示步骤进度。"""
        if event == "step_start":
            console.print(f"  [cyan]{data['name']}...[/cyan]")
        elif event == "step_complete":
            console.print("  [green]✓[/green] 完成")
            if data["name"] == "规划大纲" and data.get("result"):
                console.print(Panel(data["result"], title="大纲", border_style="dim"))
            if data["name"] == "写作" and data.get("result"):
                console.print(
                    Panel(Markdown(data["result"]), title="写作初稿", border_style="dim")
                )
            if data["name"] == "写作" and data.get("searches"):
                lines = [
                    f"  [dim]•[/dim] [link={s['url']}]{s['title']}[/link]" for s in data["searches"]
                ]
                console.print(Panel("\n".join(lines), title="信息来源", border_style="dim"))

    header = Panel(
        "[bold cyan]提示词链——科技博客流水线[/bold cyan]\n\n"
        "主题 → [大纲规划器] → [作者] → [编辑] → 最终文章\n\n"
        "每一步都会将输出传给下一步。",
        title="提示词链",
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

            console.print(f"\n[bold green]主题：[/bold green]{topic}")
            chain = PromptChain(MODEL, LIGHT_MODEL, token_tracker)

            try:
                result = chain.run(topic, on_event=on_chain_event)

                # 将文章保存到输出目录
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
                logger.error("提示词链运行失败：%s", e)
                console.print(f"\n[red]错误：{e}[/red]")
            finally:
                token_tracker.reset()

    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")


if __name__ == "__main__":
    main()
