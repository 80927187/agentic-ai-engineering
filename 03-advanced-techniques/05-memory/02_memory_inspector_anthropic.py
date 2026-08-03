"""记忆检查器——无需调用 LLM 即可浏览、搜索和管理持久记忆。

用于检查记忆智能体创建的情景（JSON）和语义（ChromaDB）记忆库的实用工具。
可用于调试、审计以及了解智能体记住了什么。
"""

from datetime import timezone

import readchar
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from common import setup_logging
from common.menu import interactive_menu
from memory import EpisodicMemory, SemanticMemory

logger = setup_logging(__name__)

MENU_OPTIONS = [
    "浏览情景记忆",
    "搜索语义记忆",
    "记忆统计",
    "清除记忆",
]


def browse_episodic(console: Console, episodic: EpisodicMemory) -> None:
    """在 Rich 表格中显示所有情景记忆。"""
    entries = episodic.list_all()
    if not entries:
        console.print("[dim]未找到情景记忆。[/dim]")
        return

    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("ID", style="dim", width=14)
    table.add_column("日期", style="cyan", width=19)
    table.add_column("内容", ratio=1)
    table.add_column("重要性", style="yellow", width=7, justify="right")

    for entry in entries:
        date_str = entry.timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
        content_preview = entry.content[:100].replace("\n", " ")
        if len(entry.content) > 100:
            content_preview += "..."
        table.add_row(entry.id, date_str, content_preview, f"{entry.importance:.1f}")

    console.print(Panel(table, title=f"情景记忆（{len(entries)} 条）", border_style="cyan"))


def search_semantic(console: Console, semantic: SemanticMemory) -> None:
    """搜索语义记忆，并显示带相似度分数的结果。"""
    if semantic.collection.count() == 0:
        console.print("[dim]未找到语义记忆。[/dim]")
        return

    console.print("[bold]请输入搜索查询：[/bold] ", end="")
    try:
        query = input().strip()
    except EOFError:
        return

    if not query:
        return

    results = semantic.search(query, limit=10)
    if not results:
        console.print("[dim]未找到结果。[/dim]")
        return

    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("#", style="dim", width=3)
    table.add_column("相似度", style="green", width=10, justify="right")
    table.add_column("内容", ratio=1)
    table.add_column("重要性", style="yellow", width=7, justify="right")

    for i, (entry, similarity) in enumerate(results, 1):
        content_preview = entry.content[:100].replace("\n", " ")
        if len(entry.content) > 100:
            content_preview += "..."
        table.add_row(str(i), f"{similarity:.3f}", content_preview, f"{entry.importance:.1f}")

    console.print(
        Panel(
            table,
            title=f'语义搜索：“{query}”（{len(results)} 条结果）',
            border_style="green",
        )
    )


def show_statistics(console: Console, episodic: EpisodicMemory, semantic: SemanticMemory) -> None:
    """显示各层记忆的统计信息。"""
    ep_stats = episodic.stats()
    sem_stats = semantic.stats()

    lines = [
        "[bold cyan]情景记忆[/bold cyan]",
        f"  条目数：{ep_stats['count']}",
        f"  文件：{ep_stats['file']}",
    ]
    if ep_stats["oldest"]:
        lines.append(f"  最早：{ep_stats['oldest']}")
        lines.append(f"  最新：{ep_stats['newest']}")

    lines.extend(
        [
            "",
            "[bold green]语义记忆[/bold green]",
            f"  条目数：{sem_stats['count']}",
            f"  集合：{sem_stats['collection']}",
        ]
    )

    total = ep_stats["count"] + sem_stats["count"]
    lines.extend(["", f"[bold]持久记忆总数：{total}[/bold]"])

    console.print(Panel("\n".join(lines), title="记忆统计", border_style="blue"))


def clear_memories(console: Console, episodic: EpisodicMemory, semantic: SemanticMemory) -> None:
    """选择记忆层并确认后清除记忆。"""
    clear_options = ["情景记忆", "语义记忆", "所有记忆"]
    choice = interactive_menu(console, clear_options, title="选择要清除的记忆")
    if not choice:
        return

    console.print(
        f"[yellow]确定要清除{choice}吗？(y/N)[/yellow] ", end=""
    )
    try:
        confirm = input().strip().lower()
    except EOFError:
        return

    if confirm != "y":
        console.print("[dim]已取消。[/dim]")
        return

    if choice in ("情景记忆", "所有记忆"):
        episodic.clear()
        console.print("[green]已清除情景记忆。[/green]")
    if choice in ("语义记忆", "所有记忆"):
        semantic.clear()
        console.print("[green]已清除语义记忆。[/green]")


def main() -> None:
    """运行记忆检查器。"""
    console = Console()

    episodic = EpisodicMemory()
    semantic = SemanticMemory()

    ep_count = episodic.stats()["count"]
    sem_count = semantic.stats()["count"]

    header = Panel(
        "[bold cyan]记忆检查器[/bold cyan]\n\n"
        "浏览和管理持久记忆——无需调用 LLM。\n\n"
        f"  情景记忆：[cyan]{ep_count}[/cyan] 条\n"
        f"  语义记忆：[green]{sem_count}[/green] 条",
        title="教程 05——记忆检查器",
    )

    while True:
        choice = interactive_menu(console, MENU_OPTIONS, title="记忆检查器", header=header)
        if not choice:
            break

        console.print()

        if choice == MENU_OPTIONS[0]:
            browse_episodic(console, episodic)
        elif choice == MENU_OPTIONS[1]:
            search_semantic(console, semantic)
        elif choice == MENU_OPTIONS[2]:
            show_statistics(console, episodic, semantic)
        elif choice == MENU_OPTIONS[3]:
            clear_memories(console, episodic, semantic)

        console.print("\n[dim]按任意键继续……[/dim]")
        readchar.readkey()


if __name__ == "__main__":
    main()
