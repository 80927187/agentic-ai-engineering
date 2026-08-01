"""
路由——“内容策略师”

演示如何根据内容分类将请求路由到专门的处理器。
大语言模型分类器先确定内容类型，再将其分派给相应的专用链
（教程、新闻或概念讲解）。
"""

from collections.abc import Callable
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

OUTPUT_DIR = Path("output")
MODEL = "claude-sonnet-4-6"
LIGHT_MODEL = "claude-haiku-4-5-20251001"

CONTENT_TYPE_LABELS = {
    "tutorial": "教程",
    "news": "新闻",
    "concept": "概念讲解",
}

SUGGESTED_TOPICS = [
    "如何在 AWS Lambda 上部署 FastAPI 应用",
    "Python 3.13 移除了 GIL",
    "什么是检索增强生成（RAG）",
    "如何使用 GitHub Actions 配置 CI/CD",
]

# --- 用于结构化输出的分类模式 ---

CLASSIFY_TOOLS = [
    {
        "name": "classify_content",
        "description": "对给定主题的内容类型进行分类。",
        "input_schema": {
            "type": "object",
            "properties": {
                "content_type": {
                    "type": "string",
                    "enum": ["tutorial", "news", "concept"],
                    "description": (
                        "tutorial：操作指南（例如“如何安装 Docker”）。"
                        "news：公告或变更（例如“Docker 更改了许可协议”）。"
                        "concept：解释某个概念（例如“什么是容器化”）。"
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": "简要说明选择该分类的原因。",
                },
            },
            "required": ["content_type", "reasoning"],
        },
    }
]

# --- 提示词 ---

CLASSIFY_SYSTEM_PROMPT = "将以下主题归入 tutorial、news 或 concept 其中一类。"

# 路由链步骤

# 教程路由
TUTORIAL_PREREQS_PROMPT = (
    "你是一名技术写作者。请列出开始本教程前需要满足的前置条件，"
    "明确说明版本和工具，并以项目符号列表输出。"
)
TUTORIAL_STEPS_PROMPT = (
    "你是一名技术写作者。请根据这些前置条件和主题撰写清晰的分步指南。"
    "为每个步骤编号，并在适当的位置加入代码示例。"
)
TUTORIAL_TROUBLESHOOTING_PROMPT = (
    "你是一名技术支持写作者。请为给定教程添加故障排除章节，"
    "列出 3～5 个常见问题及其解决方案，格式为“### 问题”/“**解决方案**”。"
)

# 新闻路由
NEWS_SUMMARY_PROMPT = (
    "你是一名科技记者。请概括主要变更或新闻，做到实事求是、简明扼要。"
    "每项独立变更使用一个项目符号。"
)
NEWS_IMPACT_PROMPT = (
    "你是一名技术分析师。请根据这份变更摘要分析其对开发者和团队的影响，"
    "包括：受影响的对象、发生的变化以及迁移注意事项。"
)
NEWS_CTA_PROMPT = (
    "你是一名科技编辑。请根据新闻和影响分析撰写简短的行动建议章节，"
    "告诉读者下一步应该做什么。建议要具体且可执行。"
)

# 概念路由
CONCEPT_ANALOGY_PROMPT = (
    "你是一名技术教育工作者。请使用清晰、贴近生活的类比来解释给定概念。"
    "先介绍类比，再过渡到技术概念。"
)
CONCEPT_ARCHITECTURE_PROMPT = (
    "你是一名软件架构师。请根据概念介绍详细说明其技术架构，"
    "包括组件之间的交互方式和常见实现。"
)
CONCEPT_PROS_CONS_PROMPT = (
    "你是一名务实的工程师。请根据概念和架构列出优点与缺点，"
    "坦诚说明其中的权衡，并以两个项目符号列表呈现。"
)

# 回调类型：路由器发出（事件名称、事件数据），由调用方决定如何显示
RouterCallback = Callable[[str, dict[str, Any]], None]


class ContentRouter:
    """根据分类结果，将主题路由到专门的内容生成链。"""

    def __init__(self, model: str, light_model: str, token_tracker: AnthropicTokenTracker):
        self.client = anthropic.Anthropic()
        self.model = model
        self.light_model = light_model
        self.token_tracker = token_tracker
        self._notify: RouterCallback = lambda _e, _d: None

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
        """执行一次大语言模型调用并跟踪令牌用量。"""
        model = self.light_model if use_light else self.model
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
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

    def _call_llm_text(self, system: str, user_message: str, *, use_light: bool = False) -> str:
        """调用大语言模型并返回文本内容。"""
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        return cast(str, self._call_llm(system, messages, use_light=use_light).content[0].text)

    def _classify(self, topic: str) -> dict[str, str]:
        """使用基于工具的结构化输出（Haiku）对主题进行分类。"""
        messages: list[dict[str, Any]] = [{"role": "user", "content": topic}]
        response = self._call_llm(
            CLASSIFY_SYSTEM_PROMPT,
            messages,
            use_light=True,
            max_tokens=256,
            tools=CLASSIFY_TOOLS,
            tool_choice={"type": "tool", "name": "classify_content"},
        )

        for block in response.content:
            if block.type == "tool_use":
                return cast(dict[str, str], block.input)

        raise ValueError("分类器未返回工具调用")

    def _chain_tutorial(self, topic: str) -> str:
        """教程链：前置条件 → 分步说明 → 故障排除。"""
        self._notify("step_start", {"name": "前置条件"})
        prerequisites = self._call_llm_text(
            TUTORIAL_PREREQS_PROMPT,
            f"以下主题需要满足哪些前置条件：{topic}",
        )
        self._notify("step_complete", {"name": "前置条件"})

        self._notify("step_start", {"name": "操作步骤"})
        steps = self._call_llm_text(
            TUTORIAL_STEPS_PROMPT,
            f"主题：{topic}\n\n前置条件：\n{prerequisites}\n\n请撰写分步指南。",
            use_light=True,
        )
        self._notify("step_complete", {"name": "操作步骤"})

        self._notify("step_start", {"name": "故障排除"})
        troubleshooting = self._call_llm_text(
            TUTORIAL_TROUBLESHOOTING_PROMPT,
            f"请为本教程添加故障排除内容：\n\n{steps}",
        )
        self._notify("step_complete", {"name": "故障排除"})
        return (
            f"# {topic}\n\n"
            f"## 前置条件\n\n{prerequisites}\n\n"
            f"## 分步指南\n\n{steps}\n\n"
            f"## 故障排除\n\n{troubleshooting}"
        )

    def _chain_news(self, topic: str) -> str:
        """新闻链：变更摘要 → 影响分析 → 行动建议。"""
        self._notify("step_start", {"name": "摘要"})
        summary = self._call_llm_text(
            NEWS_SUMMARY_PROMPT,
            f"请概括以下变更：{topic}",
        )
        self._notify("step_complete", {"name": "摘要"})

        # 中间步骤：直接扩展结构化上下文
        self._notify("step_start", {"name": "影响"})
        impact = self._call_llm_text(
            NEWS_IMPACT_PROMPT,
            f"请分析以下变更的影响：\n\n{summary}",
            use_light=True,
        )
        self._notify("step_complete", {"name": "影响"})

        self._notify("step_start", {"name": "行动建议"})
        cta = self._call_llm_text(
            NEWS_CTA_PROMPT,
            f"新闻：{summary}\n\n影响：{impact}\n\n请撰写行动建议。",
        )
        self._notify("step_complete", {"name": "行动建议"})

        return (
            f"# {topic}\n\n"
            f"## 发生了哪些变化\n\n{summary}\n\n"
            f"## 影响分析\n\n{impact}\n\n"
            f"## 你应该采取的行动\n\n{cta}"
        )

    def _chain_concept(self, topic: str) -> str:
        """概念链：类比 → 架构说明 → 优缺点。"""
        self._notify("step_start", {"name": "类比"})
        analogy = self._call_llm_text(
            CONCEPT_ANALOGY_PROMPT,
            f"请使用类比解释：{topic}",
        )
        self._notify("step_complete", {"name": "类比"})

        # 中间步骤：直接扩展结构化上下文
        self._notify("step_start", {"name": "架构"})
        architecture = self._call_llm_text(
            CONCEPT_ARCHITECTURE_PROMPT,
            f"概念介绍：{analogy}\n\n现在请说明以下主题的架构：{topic}",
            use_light=True,
        )
        self._notify("step_complete", {"name": "架构"})

        self._notify("step_start", {"name": "优缺点"})
        pros_cons = self._call_llm_text(
            CONCEPT_PROS_CONS_PROMPT,
            f"架构：{architecture}\n\n请列出以下主题的优缺点：{topic}",
        )
        self._notify("step_complete", {"name": "优缺点"})

        return (
            f"# {topic}\n\n"
            f"## 理解概念\n\n{analogy}\n\n"
            f"## 架构\n\n{architecture}\n\n"
            f"## 优缺点\n\n{pros_cons}"
        )

    def run(self, topic: str, on_event: RouterCallback | None = None) -> str:
        """对主题进行分类，将其路由到适当的链并返回结果。"""
        self._notify = on_event or (lambda _e, _d: None)

        # 第 1 步：分类
        self._notify("classify_start", {})
        classification = self._classify(topic)
        content_type = classification["content_type"]
        reasoning = classification["reasoning"]
        self.token_tracker.report()
        self._notify("classify_complete", {"content_type": content_type, "reasoning": reasoning})

        # 第 2 步：路由到专用链
        routes: dict[str, Callable[[str], str]] = {
            "tutorial": self._chain_tutorial,
            "news": self._chain_news,
            "concept": self._chain_concept,
        }

        chain_fn = routes.get(content_type)
        if not chain_fn:
            raise ValueError(f"未知的内容类型：{content_type}")

        self._notify("chain_start", {"content_type": content_type})
        result = chain_fn(topic)
        self.token_tracker.report()
        self._notify("chain_complete", {"content_type": content_type})

        return result


def main() -> None:
    """运行路由演示。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()

    def on_router_event(event: str, data: dict[str, Any]) -> None:
        """在控制台中呈现路由器事件。"""
        if event == "classify_start":
            console.print("\n[bold yellow]第 1 步：[/bold yellow]正在对主题进行分类……")
        elif event == "classify_complete":
            content_type_label = CONTENT_TYPE_LABELS.get(
                data["content_type"], data["content_type"]
            )
            console.print(
                Panel(
                    f"[bold]{content_type_label}[/bold]\n{data['reasoning']}",
                    title="分类结果",
                    border_style="cyan",
                )
            )
        elif event == "chain_start":
            content_type_label = CONTENT_TYPE_LABELS.get(
                data["content_type"], data["content_type"]
            )
            console.print(
                f"\n[bold yellow]第 2 步：[/bold yellow]正在运行{content_type_label}链……"
            )
        elif event == "step_start":
            console.print(f"  [cyan]{data['name']}...[/cyan]")
        elif event == "step_complete":
            console.print("  [green]\u2713[/green] 完成")

    header = Panel(
        "[bold cyan]路由——内容策略师[/bold cyan]\n\n"
        "主题 \u2192 [分类器] \u2192 路由 [bold]A[/bold]、[bold]B[/bold] 或 [bold]C[/bold] \u2192 [专用链] \u2192 文章\n\n"
        "[bold]A.[/bold] 教程（操作指南）：前置条件 \u2192 操作步骤 \u2192 故障排除\n"
        "[bold]B.[/bold] 新闻/公告：变更 \u2192 影响 \u2192 行动建议\n"
        "[bold]C.[/bold] 概念讲解：类比 \u2192 架构 \u2192 优缺点",
        title="路由演示",
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
            router = ContentRouter(MODEL, LIGHT_MODEL, token_tracker)

            try:
                result = router.run(topic, on_event=on_router_event)

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
                logger.error("路由失败：%s", e)
                console.print(f"\n[red]错误：{e}[/red]")
            finally:
                token_tracker.reset()

    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")


if __name__ == "__main__":
    main()
