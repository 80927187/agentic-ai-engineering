"""
代理单元测试——测试运行器

运行单元测试教程的完整测试套件，演示代理测试的四个层级：模拟大语言模型响应、
隔离测试工具、行为契约，以及使用响应录制文件的集成测试。

测试位于 tests/，也可以直接通过 pytest 运行：
  pytest tests/ -v
"""

from pathlib import Path

import pytest
from common import setup_logging
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

# 测试模块及其说明
TEST_SUITES = [
    ("tests/test_mock_llm.py", "模拟大语言模型测试", "模拟 API 响应，测试代理循环逻辑"),
    ("tests/test_tools.py", "工具测试", "隔离测试工具函数及其边界情况"),
    (
        "tests/test_behavioral_contracts.py",
        "行为契约",
        "验证代理不变量（安全、终止和历史记录）",
    ),
    (
        "tests/test_integration.py",
        "集成测试",
        "记录/重放 API 响应，执行快照回归测试",
    ),
]


def main() -> None:
    """显示测试套件概览，并在用户确认后运行测试。"""
    console = Console()

    console.print(
        Panel(
            "[bold cyan]代理单元测试[/bold cyan]\n\n"
            "使用四种互补策略测试工具调用代理循环：\n"
            "  1. 模拟大语言模型响应——确定性的代理循环测试\n"
            "  2. 隔离工具——涵盖边界情况的纯函数测试\n"
            "  3. 行为契约——安全、终止和历史记录不变量\n"
            "  4. 集成测试——使用响应录制文件测试完整循环\n\n"
            "无需 API 密钥——所有内容均为模拟。",
            title="01——代理单元测试",
        )
    )

    # 显示可用的测试套件
    table = Table(title="测试套件", show_lines=True)
    table.add_column("#", width=3, justify="center")
    table.add_column("套件", style="cyan", width=24)
    table.add_column("说明", width=50)

    for i, (_, name, desc) in enumerate(TEST_SUITES, 1):
        table.add_row(str(i), name, desc)

    console.print(table)

    # 执行前请求用户确认
    console.print("\n[bold]即将运行上面列出的全部测试套件。[/bold]")
    try:
        answer = console.input("[dim]按 Enter 运行，或输入“q”退出：[/dim]")
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]已取消。[/yellow]")
        return

    if answer.strip().lower() in ("q", "quit", "exit"):
        console.print("[yellow]已取消。[/yellow]")
        return

    # 运行测试
    console.print("\n[bold]正在运行测试……[/bold]\n")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "test_results.xml"

    test_files = [str(Path(__file__).parent / path) for path, _, _ in TEST_SUITES]
    exit_code = pytest.main(
        [
            *test_files,
            "-v",
            "--tb=short",
            "--no-header",
            f"--junitxml={report_path}",
        ]
    )

    if exit_code == 0:
        console.print("\n[bold green]全部测试通过！[/bold green]")
    else:
        console.print(f"\n[bold red]部分测试失败（退出码：{exit_code}）[/bold red]")

    console.print(f"[dim]测试报告已保存至 {report_path}[/dim]")


if __name__ == "__main__":
    main()
