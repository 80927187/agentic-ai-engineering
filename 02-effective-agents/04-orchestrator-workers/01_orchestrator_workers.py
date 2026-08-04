"""
编排器—工作器：“深度研究员”

演示中央大语言模型如何动态拆解任务、将子任务委派给工作器大语言模型，
并将它们的结果综合成最终文章。
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

OUTPUT_DIR = Path("output")
MODEL = "deepseek-v4-flash"
LIGHT_MODEL = "deepseek-v4-flash"

# Anthropic 兼容接口的服务端网络搜索工具——由模型决定何时搜索
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

SUGGESTED_TOPICS = [
    "比较 Bun 与 Node.js 在后端开发中的表现",
    "Python 3.13 的新功能与性能",
    "WebAssembly 生产应用现状",
    "2025 年 AI 代码审查工具全景",
]

# --- 提示词 ---

ORCHESTRATOR_SYSTEM_PROMPT = (
    "你是一名研究编排器。请将给定主题拆解为 2～4 个可独立调查的具体研究子主题。"
    "每个子主题应涵盖该主题的不同维度。请像记者一样思考，在动笔前分别研究每个角度。"
    "必须调用 create_research_plan 工具提交研究计划。"
)

WORKER_SYSTEM_PROMPT = (
    "你是一名严谨的技术研究员。请深入研究给定主题，并尽可能提供具体细节、示例、"
    "比较和数据。撰写 3～4 段内容充实的分析。如果最新信息有助于该主题，请使用网络搜索。"
)

SYNTHESIZER_SYSTEM_PROMPT = (
    "你是一名资深技术作者。请将来自多个来源、针对不同子主题的研究，"
    "综合成一篇连贯且结构清晰的文章。"
)

# 供编排器拆解任务的工具
PLANNING_TOOLS = [
    {
        "name": "create_research_plan",
        "description": "将主题拆解为供工作器研究的具体子主题。",
        "input_schema": {
            "type": "object",
            "properties": {
                "subtopics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "子主题标题",
                            },
                            "research_prompt": {
                                "type": "string",
                                "description": "交给工作器的具体研究问题",
                            },
                        },
                        "required": ["title", "research_prompt"],
                    },
                    "description": "需要并行研究的子主题列表",
                },
                "synthesis_instructions": {
                    "type": "string",
                    "description": "说明如何将研究内容整合成最终文章",
                },
            },
            "required": ["subtopics", "synthesis_instructions"],
        },
    }
]

# 回调类型：智能体发出（事件名称、事件数据），由调用方决定如何展示
OrchestratorCallback = Callable[[str, dict[str, Any]], None]


class OrchestratorWorkers:
    """编排器动态拆解任务，工作器并行执行。"""

    def __init__(self, model: str, light_model: str, token_tracker: AnthropicTokenTracker):
        self.client = anthropic.Anthropic()
        self.model = model
        self.light_model = light_model
        self.token_tracker = token_tracker
        self._notify: OrchestratorCallback = lambda _e, _d: None

    def _call_llm(
        self,
        system: str,
        messages: list[dict[str, Any]],
        *,
        use_light: bool = False,
        max_tokens: int = 21333,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, str] | None = None,
    ) -> anthropic.types.Message:
        """执行一次大语言模型调用并跟踪 token 用量。"""
        model = self.light_model if use_light else self.model
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
        # DeepSeek 思考模式不支持 tool_choice；保留形参和调用点用于对照学习。
        # if tool_choice:
        #     kwargs["tool_choice"] = tool_choice
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
        response = self._call_llm(system, messages, **kwargs)
        text_parts = [block.text for block in response.content if block.type == "text"]
        if not text_parts:
            block_types = [block.type for block in response.content]
            raise ValueError(
                f"模型响应中没有文本内容（stop_reason={response.stop_reason}，"
                f"内容块={block_types}，output_tokens={response.usage.output_tokens}）。"
            )
        return "\n\n".join(text_parts)

    def _plan(self, topic: str) -> dict[str, Any]:
        """编排器：将主题动态拆解为多个子主题。"""
        logger.info("编排器正在规划：%s", topic)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"请为以下主题制定研究计划：{topic}"}
        ]
        response = self._call_llm(
            ORCHESTRATOR_SYSTEM_PROMPT,
            messages,
            max_tokens=21333,
            tools=PLANNING_TOOLS,
            tool_choice={"type": "tool", "name": "create_research_plan"},
        )

        for block in response.content:
            if block.type == "tool_use":
                return cast(dict[str, Any], block.input)

        block_types = [block.type for block in response.content]
        raise ValueError(
            f"编排器未生成研究计划（stop_reason={response.stop_reason}，"
            f"内容块={block_types}，output_tokens={response.usage.output_tokens}）。"
        )

    def _research_subtopic(
        self, subtopic: dict[str, str], max_turns: int = 100
    ) -> dict[str, str]:
        """工作器：深入研究一个子主题，并可按需使用网络搜索。"""
        title = subtopic["title"]
        logger.info("工作器正在研究：%s", title)

        messages: list[dict[str, Any]] = [{"role": "user", "content": subtopic["research_prompt"]}]
        response = self._call_llm(
            WORKER_SYSTEM_PROMPT, messages, use_light=True, tools=[WEB_SEARCH_TOOL]
        )

        for attempt in range(max_turns):
            search_uses = sum(block.type == "server_tool_use" for block in response.content)
            if search_uses:
                logger.info("工作器 %s 本轮发起 %d 次网络搜索", title, search_uses)

            for block in response.content:
                if block.type == "web_search_tool_result" and isinstance(block.content, list):
                    for result in block.content:
                        if result.type == "web_search_tool_result_error":
                            logger.warning("网络搜索失败：%s", result.error_code)

            if response.stop_reason == "end_turn":
                break
            if response.stop_reason != "pause_turn":
                logger.warning("工作器 %s 以未处理的原因停止：%s", title, response.stop_reason)
                break
            if attempt == max_turns - 1:
                raise RuntimeError("服务端工具连续暂停，超过最大续传次数。")

            messages.append({"role": "assistant", "content": response.content})
            response = self._call_llm(
                WORKER_SYSTEM_PROMPT,
                messages,
                use_light=True,
                tools=[WEB_SEARCH_TOOL],
            )

        text_parts = [block.text for block in response.content if block.type == "text"]
        if not text_parts:
            block_types = [block.type for block in response.content]
            raise ValueError(
                f"模型响应中没有文本内容（stop_reason={response.stop_reason}，"
                f"内容块={block_types}，output_tokens={response.usage.output_tokens}）。"
            )
        return {"title": title, "content": "\n\n".join(text_parts)}

    def _synthesize(self, topic: str, research: list[dict[str, str]], instructions: str) -> str:
        """综合器：将所有工作器的研究整合成连贯的最终文章。"""
        logger.info("正在综合 %d 个研究章节", len(research))
        sections = "\n\n---\n\n".join(f"## {r['title']}\n\n{r['content']}" for r in research)
        system = f"{SYNTHESIZER_SYSTEM_PROMPT} 综合要求：{instructions}"
        user_msg = (
            f"# {topic}\n\n研究章节：\n\n{sections}\n\n"
            "请综合成一篇完整、连贯的文章。"
        )
        return self._call_llm_text(system, user_msg)

    def run(self, topic: str, on_event: OrchestratorCallback | None = None) -> str:
        """执行完整的编排器—工作器流程。"""
        self._notify = on_event or (lambda _e, _d: None)

        # 第 1 步：编排器规划
        self._notify("plan_start", {})
        plan = self._plan(topic)
        subtopics = plan["subtopics"]
        instructions = plan["synthesis_instructions"]
        self.token_tracker.report()
        self._notify("plan_complete", {"subtopics": subtopics})

        # 第 2 步：工作器并行研究
        self._notify("workers_start", {"count": len(subtopics)})
        research_results: list[dict[str, str]] = []

        with ThreadPoolExecutor(max_workers=len(subtopics)) as executor:
            futures = {
                executor.submit(self._research_subtopic, sub): sub["title"] for sub in subtopics
            }
            for future in as_completed(futures):
                title = futures[future]
                try:
                    result = future.result()
                    research_results.append(result)
                    self._notify("worker_complete", {"title": title})
                except Exception as e:
                    logger.error("工作器研究 %s 时失败：%s", title, e)

        self.token_tracker.report()

        # 第 3 步：综合
        self._notify("synthesize_start", {})
        final = self._synthesize(topic, research_results, instructions)
        self.token_tracker.report()
        self._notify("synthesize_complete", {})

        return final


def main() -> None:
    """运行编排器—工作器演示。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()

    def on_event(event: str, data: dict[str, Any]) -> None:
        """处理流程事件并在控制台展示。"""
        if event == "plan_start":
            console.print("\n[bold yellow]编排器：[/bold yellow]正在规划研究……")
        elif event == "plan_complete":
            subtopics = data["subtopics"]
            console.print(
                Panel(
                    "\n".join(f"• {s['title']}" for s in subtopics),
                    title=f"研究计划（{len(subtopics)} 个子主题）",
                    border_style="cyan",
                )
            )
        elif event == "workers_start":
            console.print(
                f"\n[bold yellow]工作器：[/bold yellow]"
                f"正在并行研究 {data['count']} 个子主题……"
            )
        elif event == "worker_complete":
            console.print(f"  [green]✓[/green] {data['title']}")
        elif event == "synthesize_start":
            console.print("\n[bold yellow]综合器：[/bold yellow]正在整合研究……")
        elif event == "synthesize_complete":
            console.print("  [green]✓[/green] 完成")

    header = Panel(
        "[bold cyan]编排器—工作器：深度研究员[/bold cyan]\n\n"
        "主题 → [编排器] → 动态子主题列表\n"
        "     → [工作器 1] + [工作器 2] + [工作器 N]（并行）\n"
        "     → [综合器] → 最终文章\n\n"
        "由大语言模型决定研究内容——你定义的是工作器能力，而不是任务。",
        title="编排器—工作器",
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
            orch = OrchestratorWorkers(MODEL, LIGHT_MODEL, token_tracker)

            try:
                result = orch.run(topic, on_event=on_event)

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
                logger.error("编排失败：%s", e)
                console.print(f"\n[red]错误：{e}[/red]")
            finally:
                token_tracker.reset()

    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")


if __name__ == "__main__":
    main()
