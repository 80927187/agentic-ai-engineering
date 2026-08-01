"""
代码分块器

将源代码文件拆分为适合生成嵌入和语义搜索的代码块。
使用简单的启发式规则：Python 文件按类/函数定义拆分，
其他文件按固定行数拆分，并在相邻块之间保留重叠内容。
"""

from pathlib import Path
from typing import Any

from common.logging_config import setup_logging

logger = setup_logging(__name__)

# 要建立索引的文件扩展名
INDEXABLE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".go",
    ".rs",
    ".java",
    ".rb",
    ".php",
    ".c",
    ".cpp",
    ".h",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
}

# 要跳过的目录
SKIP_DIRS = {
    "node_modules",
    "venv",
    ".venv",
    ".git",
    "__pycache__",
    "dist",
    "build",
    ".next",
    "vendor",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    "egg-info",
}

# 非 Python 文件每个代码块的最大行数
CHUNK_SIZE = 50
OVERLAP = 10


def collect_files(repo_path: Path) -> list[Path]:
    """收集仓库中所有可建立索引的文件。"""
    files = []
    for path in repo_path.rglob("*"):
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        if path.is_file() and path.suffix in INDEXABLE_EXTENSIONS:
            files.append(path)
    return sorted(files)


def chunk_python(content: str, filepath: str, repo: str) -> list[dict[str, Any]]:
    """按顶层类/函数定义拆分 Python 文件。"""
    lines = content.split("\n")
    chunks: list[dict[str, Any]] = []
    current_chunk_start = 0

    for i, line in enumerate(lines):
        # 在顶层定义处拆分（行首没有空白字符）
        if i > 0 and (line.startswith("class ") or line.startswith("def ")):
            chunk_content = "\n".join(lines[current_chunk_start:i]).strip()
            if chunk_content:
                chunks.append(
                    {
                        "content": chunk_content,
                        "filepath": filepath,
                        "start_line": current_chunk_start + 1,
                        "end_line": i,
                        "repo": repo,
                    }
                )
            current_chunk_start = i

    # 不要遗漏最后一个代码块
    chunk_content = "\n".join(lines[current_chunk_start:]).strip()
    if chunk_content:
        chunks.append(
            {
                "content": chunk_content,
                "filepath": filepath,
                "start_line": current_chunk_start + 1,
                "end_line": len(lines),
                "repo": repo,
            }
        )

    return chunks


def chunk_generic(content: str, filepath: str, repo: str) -> list[dict[str, Any]]:
    """按固定行数拆分非 Python 文件，并保留重叠内容。"""
    lines = content.split("\n")
    chunks: list[dict[str, Any]] = []

    i = 0
    while i < len(lines):
        end = min(i + CHUNK_SIZE, len(lines))
        chunk_content = "\n".join(lines[i:end]).strip()
        if chunk_content:
            chunks.append(
                {
                    "content": chunk_content,
                    "filepath": filepath,
                    "start_line": i + 1,
                    "end_line": end,
                    "repo": repo,
                }
            )
        i += CHUNK_SIZE - OVERLAP

    return chunks


def chunk_file(path: Path, repo_path: Path, repo_name: str) -> list[dict[str, Any]]:
    """使用适当的策略拆分单个文件。"""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning("无法读取 %s：%s", path, e)
        return []

    if not content.strip():
        return []

    # 限制超大文件的大小
    if len(content) > 100_000:
        content = content[:100_000]

    filepath = str(path.relative_to(repo_path))

    if path.suffix == ".py":
        return chunk_python(content, filepath, repo_name)
    return chunk_generic(content, filepath, repo_name)


def chunk_repository(repo_path: Path, repo_name: str) -> list[dict[str, Any]]:
    """拆分仓库中的所有文件。"""
    files = collect_files(repo_path)
    logger.info("在 %s 中找到 %d 个可建立索引的文件", repo_path, len(files))

    all_chunks: list[dict[str, Any]] = []
    for path in files:
        all_chunks.extend(chunk_file(path, repo_path, repo_name))

    logger.info("从 %d 个文件创建了 %d 个代码块", len(files), len(all_chunks))
    return all_chunks
