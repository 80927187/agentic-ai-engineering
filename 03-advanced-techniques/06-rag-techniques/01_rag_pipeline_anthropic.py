"""RAG 流水线（Anthropic）。

演示完整的检索增强生成流水线：摄取文档、分块、使用本地 sentence-transformer
模型生成嵌入向量、写入 ChromaDB 和 BM25 索引、通过混合搜索与重排序检索，
最后使用 Claude 生成答案。

需要设置 ANTHROPIC_API_KEY 环境变量。
"""

from pathlib import Path

import anthropic
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from common import AnthropicTokenTracker, setup_logging
from common.menu import interactive_menu
from rag import HybridRetriever, LocalEmbedder, Reranker, VectorStore, recursive_split

# 从根目录的 .env 文件加载环境变量
load_dotenv(find_dotenv())

# 配置日志
logger = setup_logging(__name__)

# 模型配置
MODEL = "deepseek-v4-flash"
SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"
CHROMA_PERSIST_DIR = str(Path(__file__).parent / ".chroma_db")

SYSTEM_PROMPT = (
    "你是 TechFlow Solutions 的技术支持助手。"
    "只能根据提供的上下文回答问题。"
    "每项事实都要引用来源文档（例如 [api_reference.md]）。"
    "如果上下文中没有答案，请明确说明，不要编造信息。"
)

# 预设的演示问题，覆盖不同文档和检索模式
DEMO_QUESTIONS = [
    "如何通过 TechFlow API 进行身份验证？",
    "TechFlow 使用什么数据库进行缓存？",
    "如何回滚失败的部署？",
    "为什么我的 Webhook 没有触发？",
    "专业版套餐的速率限制是多少？",
    "请说明 TechFlow 架构中的服务如何相互通信。",
]


class RAGPipeline:
    """完整的 RAG 流水线：摄取 → 检索 → 生成。"""

    def __init__(self, model: str, token_tracker: AnthropicTokenTracker):
        self.client = anthropic.Anthropic()
        self.model = model
        self.token_tracker = token_tracker

        # 构建检索技术栈
        self.embedder = LocalEmbedder()
        self.store = VectorStore(self.embedder, persist_dir=CHROMA_PERSIST_DIR)
        self.reranker = Reranker()
        self.retriever = HybridRetriever(self.store, self.reranker)

    def ingest(self, docs_dir: Path) -> int:
        """加载 Markdown 文件，执行分块、嵌入和索引，并返回文本块数量。"""
        all_chunks = []

        for doc_path in sorted(docs_dir.glob("*.md")):
            text = doc_path.read_text(encoding="utf-8")
            chunks = recursive_split(text, source=doc_path.name)
            all_chunks.extend(chunks)
            logger.info("已将 %s 切分为 %d 个文本块", doc_path.name, len(chunks))

        self.store.add_chunks(all_chunks)
        return len(all_chunks)

    def query(self, question: str, top_k: int = 5) -> tuple[str, list]:
        """检索相关文本块，并生成带有引用的答案。"""
        chunks = self.retriever.retrieve(question, top_k=top_k)
        context = self._build_context(chunks)
        answer = self._generate(question, context)
        return answer, chunks

    def _build_context(self, chunks: list) -> str:
        """将检索到的文本块格式化为带编号的上下文块。"""
        if not chunks:
            return "未找到相关上下文。"

        blocks = []
        for i, chunk in enumerate(chunks, 1):
            blocks.append(f"[{i}] 来源：{chunk.source}\n{chunk.content}")
        return "\n\n---\n\n".join(blocks)

    def _generate(self, question: str, context: str) -> str:
        """将问题和上下文发送给 Claude，并返回答案。"""
        user_message = f"上下文：\n{context}\n\n问题：{question}"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=21333,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        self.token_tracker.track(response.usage)
        text_parts = [block.text for block in response.content if block.type == "text"]
        if not text_parts:
            block_types = [block.type for block in response.content]
            raise ValueError(
                f"模型响应中没有文本内容（stop_reason={response.stop_reason}，"
                f"内容块={block_types}，output_tokens={response.usage.output_tokens}）。"
            )
        return "\n\n".join(text_parts)


def _render_chunks(console: Console, chunks: list) -> None:
    """显示检索到的文本块、来源和内容预览。"""
    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("#", style="dim", width=3)
    table.add_column("来源", style="cyan", min_width=20)
    table.add_column("预览", ratio=1)

    for i, chunk in enumerate(chunks, 1):
        preview = chunk.content[:120].replace("\n", " ") + "..."
        table.add_row(str(i), chunk.source, f"[dim]{preview}[/dim]")

    console.print(Panel(table, title="检索到的文本块", border_style="dim", padding=(0, 1)))


def _run_demo(console: Console, pipeline: RAGPipeline) -> None:
    """逐个运行预设的演示问题，每次等待用户输入。"""
    console.print(f"\n[bold]即将运行 {len(DEMO_QUESTIONS)} 个演示问题。[/bold]")
    console.print("[dim]按 Enter 运行下一个问题，输入 'q' 停止。[/dim]\n")

    for i, question in enumerate(DEMO_QUESTIONS, 1):
        console.print(f"[bold green]问题 {i}/{len(DEMO_QUESTIONS)}：[/bold green] {question}")
        console.print("[dim]按 Enter 运行...[/dim] ", end="")
        try:
            if input().strip().lower() == "q":
                break
        except EOFError:
            break

        try:
            answer, chunks = pipeline.query(question)

            _render_chunks(console, chunks)

            console.print("\n[bold blue]回答：[/bold blue]")
            console.print(Markdown(answer))
            console.print("\n" + "─" * 60 + "\n")

        except Exception as e:
            logger.error("处理第 %d 个问题时出错：%s", i, e)
            console.print(f"[red]错误：{e}[/red]\n")


def _run_interactive(console: Console, pipeline: RAGPipeline) -> None:
    """交互模式：由用户提问。"""
    console.print(
        "\n[bold]交互模式[/bold]：可以询问有关 TechFlow 的问题。\n"
        "输入 [bold]'quit'[/bold] 或 [bold]'exit'[/bold] 结束。\n"
    )

    while True:
        console.print("[bold green]问题：[/bold green] ", end="")
        user_input = input().strip()

        if user_input.lower() in ["quit", "exit", ""]:
            break

        try:
            answer, chunks = pipeline.query(user_input)

            _render_chunks(console, chunks)

            console.print("\n[bold blue]回答：[/bold blue]")
            console.print(Markdown(answer))
            console.print()

        except Exception as e:
            logger.error("处理问题时出错：%s", e)
            console.print(f"\n[red]错误：{e}[/red]")


def main() -> None:
    """RAG 流水线演示的主编排函数。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()

    with console.status("[bold]正在加载嵌入模型（首次运行会下载约 80MB）...[/bold]"):
        pipeline = RAGPipeline(MODEL, token_tracker)

    header = Panel(
        "[bold cyan]RAG 流水线演示[/bold cyan]\n\n"
        "本演示会摄取 TechFlow 文档、构建混合索引（向量 + BM25），\n"
        "并在回答问题时引用信息来源。\n\n"
        "[bold]流水线：[/bold]分块 → 嵌入（本地）→ 索引（ChromaDB + BM25）\n"
        "         → 混合检索 → 重排序（FlashRank）→ 生成（Claude）\n\n"
        "[bold]可以尝试以下问题：[/bold]\n"
        "  1. 如何通过 TechFlow API 进行身份验证？\n"
        "  2. TechFlow 使用什么数据库进行缓存？\n"
        "  3. 如何回滚失败的部署？\n"
        "  4. 为什么我的 Webhook 没有触发？\n"
        "  5. 专业版套餐的速率限制是多少？",
        title="RAG 流水线",
    )
    console.print(header)

    # 摄取文档
    console.print("\n[bold]正在摄取文档...[/bold]")
    try:
        chunk_count = pipeline.ingest(SAMPLE_DOCS_DIR)
        console.print(
            f"[green]已从 {len(list(SAMPLE_DOCS_DIR.glob('*.md')))} 份文档中"
            f"索引 {chunk_count} 个文本块[/green]\n"
        )
    except Exception as e:
        logger.error("文档摄取失败：%s", e)
        console.print(f"[red]文档摄取失败：{e}[/red]")
        return

    mode = interactive_menu(
        console,
        items=[
            "演示——使用完整流水线运行示例问题",
            "交互——提出自己的问题",
        ],
        title="选择模式",
    )

    if mode is None:
        return

    if mode.startswith("演示"):
        _run_demo(console, pipeline)
    else:
        _run_interactive(console, pipeline)

    # 最终 Token 用量报告
    console.print()
    token_tracker.report()


if __name__ == "__main__":
    main()
