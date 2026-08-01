"""
文件工具

用于读取文件和探索已索引仓库目录结构的工具。
"""

from pathlib import Path
from typing import Any

from common.logging_config import setup_logging

logger = setup_logging(__name__)

# 克隆仓库的存储位置
REPOS_DIR = Path(__file__).parent.parent / "repos"

FILE_TOOLS = [
    {
        "name": "read_file",
        "description": "读取已索引仓库中文件的完整内容，并显示行号。",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "仓库名称（例如“pallets-flask”）",
                },
                "filepath": {
                    "type": "string",
                    "description": "文件在仓库内的路径（例如“src/flask/app.py”）",
                },
            },
            "required": ["repo", "filepath"],
        },
    },
    {
        "name": "list_directory",
        "description": "列出已索引仓库中指定路径下的文件和目录。",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "仓库名称（例如“pallets-flask”）",
                },
                "path": {
                    "type": "string",
                    "description": "仓库内的路径（默认为根目录）",
                    "default": "",
                },
            },
            "required": ["repo"],
        },
    },
]


def _find_repo_path(repo: str) -> Path | None:
    """查找仓库的本地路径。"""
    # 检查仓库目录
    repo_path = REPOS_DIR / repo
    if repo_path.is_dir():
        return repo_path

    # 检查是否为带有 local- 前缀的集合
    if not repo.startswith("local-"):
        repo_path = REPOS_DIR / f"local-{repo}"
        if repo_path.is_dir():
            return repo_path

    return None


def execute_read_file(_vector_store: Any, tool_input: dict[str, Any]) -> str:
    """读取已索引仓库中的文件并显示行号。"""
    repo = tool_input["repo"]
    filepath = tool_input["filepath"]

    repo_path = _find_repo_path(repo)
    if not repo_path:
        return f"在本地找不到仓库“{repo}”。"

    file_path = repo_path / filepath
    if not file_path.is_file():
        return f"在 {repo} 中找不到文件：{filepath}"

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return f"读取文件时出错：{e}"

    # 添加行号，并截断大文件以避免上下文急剧膨胀
    lines = content.split("\n")
    max_lines = 300
    truncated = len(lines) > max_lines
    display_lines = lines[:max_lines]
    numbered = [f"{i + 1:4d} | {line}" for i, line in enumerate(display_lines)]
    result = f"文件：{filepath}（{len(lines)} 行）\n\n" + "\n".join(numbered)
    if truncated:
        result += f"\n\n……内容已截断（剩余 {len(lines) - max_lines} 行）"
    return result


def execute_list_directory(_vector_store: Any, tool_input: dict[str, Any]) -> str:
    """列出已索引仓库中的目录内容。"""
    repo = tool_input["repo"]
    subpath = tool_input.get("path", "")

    repo_path = _find_repo_path(repo)
    if not repo_path:
        return f"在本地找不到仓库“{repo}”。"

    target = repo_path / subpath
    if not target.is_dir():
        return f"在 {repo} 中找不到目录：{subpath or '/'}"

    entries = sorted(target.iterdir())
    lines = [f"目录：{repo} 中的 {subpath or '/'}\n"]

    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            lines.append(f"  📁 {entry.name}/")
        else:
            size = entry.stat().st_size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            lines.append(f"  📄 {entry.name} ({size_str})")

    return "\n".join(lines)
