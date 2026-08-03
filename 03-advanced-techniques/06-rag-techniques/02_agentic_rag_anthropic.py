"""智能体 RAG（Anthropic）。

演示如何在智能体循环中将 RAG 作为工具使用。智能体会决定何时搜索、使用什么
查询，以及结果是否足够。如果首次检索结果不充分，智能体会改写查询并再次搜索。

与脚本 01（流水线 RAG）中每个问题都会触发检索不同，这里的智能体会自行判断：
有些问题可根据对话上下文回答，搜索查询也由智能体自己拟定。

需要设置 ANTHROPIC_API_KEY 环境变量。
"""

from pathlib import Path

import anthropic
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from common import AnthropicTokenTracker, setup_logging
from rag import HybridRetriever, LocalEmbedder, Reranker, VectorStore, recursive_split

# 从根目录的 .env 文件加载环境变量
load_dotenv(find_dotenv())

# 配置日志
logger = setup_logging(__name__)

# 模型配置
MODEL = "claude-sonnet-4-6"
SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"
CHROMA_PERSIST_DIR = str(Path(__file__).parent / ".chroma_db")

SYSTEM_PROMPT = (
    "你是 TechFlow Solutions 的技术支持智能体，可以通过搜索工具访问公司文档。\n\n"
    "准则：\n"
    "- 需要具体技术细节时搜索文档\n"
    "- 使用明确、有针对性的搜索查询，避免过于宽泛\n"
    "- 如果首次结果不充分，改写查询后再次搜索\n"
    "- 并非每个问题都需要搜索，请自行判断\n"
    "- 始终注明信息来自哪份文档（例如 [api_reference.md]）\n"
    "- 如果文档未涵盖某个主题，请明确说明"
)

TOOLS = [
    {
        "name": "search_docs",
        "description": (
            "在 TechFlow 文档中搜索信息。使用明确、有针对性的查询可获得最佳结果。"
            "可以使用不同查询多次调用此工具，以查找更多信息。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询，应明确具体并使用技术术语",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的结果数量（默认 5，最多 10）",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    }
]


class AgenticRAG:
    """在推理循环中将检索作为工具使用的智能体。"""

    def __init__(
        self,
        model: str,
        retriever: HybridRetriever,
        token_tracker: AnthropicTokenTracker,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.retriever = retriever
        self.token_tracker = token_tracker
        self.messages: list[dict] = []

    def chat(self, user_input: str, console: Console) -> str:
        """智能体循环：发送 → 检测工具调用 → 执行搜索 → 继续。"""
        self.messages.append({"role": "user", "content": user_input})

        # 智能体循环：持续运行，直到模型生成文本响应
        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self.messages,
            )

            self.token_tracker.track(response.usage)

            # 检查模型是否要使用工具
            if response.stop_reason == "tool_use":
                # 处理本次响应中的所有工具调用
                self.messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        query = block.input.get("query", "")
                        top_k = min(block.input.get("top_k", 5), 10)

                        console.print(f"  [dim]正在搜索：[/dim] [italic]{query}[/italic]")

                        result = self._execute_search(query, top_k)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )

                self.messages.append({"role": "user", "content": tool_results})
                continue

            # 模型已经生成最终文本响应
            assistant_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    assistant_text += block.text

            self.messages.append({"role": "assistant", "content": assistant_text})
            return assistant_text

    def _execute_search(self, query: str, top_k: int) -> str:
        """执行检索，并为智能体格式化结果。"""
        chunks = self.retriever.retrieve(query, top_k=top_k)

        if not chunks:
            return "未找到与该查询相关的文档。"

        results = []
        for i, chunk in enumerate(chunks, 1):
            results.append(f"[{i}] 来源：{chunk.source}\n{chunk.content}")

        return "\n\n---\n\n".join(results)


def _build_retriever() -> HybridRetriever:
    """构建检索技术栈并摄取文档。"""
    embedder = LocalEmbedder()
    store = VectorStore(embedder, persist_dir=CHROMA_PERSIST_DIR)
    reranker = Reranker()
    retriever = HybridRetriever(store, reranker)

    # 摄取示例文档
    all_chunks = []
    for doc_path in sorted(SAMPLE_DOCS_DIR.glob("*.md")):
        text = doc_path.read_text(encoding="utf-8")
        chunks = recursive_split(text, source=doc_path.name)
        all_chunks.extend(chunks)

    store.add_chunks(all_chunks)
    return retriever


def main() -> None:
    """智能体 RAG 演示的主编排函数。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()

    console.print(
        Panel(
            "[bold cyan]智能体 RAG 演示[/bold cyan]\n\n"
            "与流水线 RAG（脚本 01）不同，此智能体会决定[bold]何时[/bold]搜索、\n"
            "使用[bold]什么查询[/bold]，以及[bold]结果是否充分[/bold]。\n\n"
            "智能体可以调用 [cyan]search_docs[/cyan] 工具，也可以选择不调用。\n"
            "请留意：\n"
            "  - 智能体自行选择搜索查询（可能与用户问题不同）\n"
            "  - 智能体为复杂问题执行多次搜索\n"
            "  - 智能体不搜索，直接根据对话上下文回答\n\n"
            "[bold]可以尝试以下问题：[/bold]\n"
            "  1. 如何通过 API 进行身份验证？\n"
            "  2. 部署失败时会怎样？（追问：数据库回滚呢？）\n"
            "  3. API 请求为什么可能很慢？\n"
            "  4. 比较不同的套餐等级。\n\n"
            "输入 [bold]'quit'[/bold] 或 [bold]'exit'[/bold] 结束。",
            title="智能体 RAG",
        )
    )

    # 构建检索技术栈（首次运行会下载约 80MB 的嵌入模型）
    console.print("\n[bold]正在加载模型并摄取文档...[/bold]")
    try:
        with console.status("[bold]正在加载嵌入模型（首次运行会下载约 80MB）...[/bold]"):
            retriever = _build_retriever()
        doc_count = len(list(SAMPLE_DOCS_DIR.glob("*.md")))
        console.print(f"[green]已索引来自 {doc_count} 个文件的文档[/green]\n")
    except Exception as e:
        logger.error("文档摄取失败：%s", e)
        console.print(f"[red]文档摄取失败：{e}[/red]")
        return

    agent = AgenticRAG(MODEL, retriever, token_tracker)

    while True:
        console.print("[bold green]你：[/bold green] ", end="")
        user_input = input().strip()

        if user_input.lower() in ["quit", "exit", ""]:
            console.print("\n[yellow]正在结束会话...[/yellow]")
            break

        try:
            response = agent.chat(user_input, console)

            console.print("\n[bold blue]智能体：[/bold blue]")
            console.print(Markdown(response))
            console.print()

        except Exception as e:
            logger.error("对话期间出错：%s", e)
            console.print(f"\n[red]错误：{e}[/red]")
            break

    # 最终 Token 用量报告
    console.print()
    token_tracker.report()


if __name__ == "__main__":
    main()
