"""
搜索工具

在已索引代码库中进行语义搜索和正则表达式 grep 搜索。
"""

import re
from pathlib import Path
from typing import Any

from indexer.chunker import INDEXABLE_EXTENSIONS
from indexer.embedder import Embedder
from store.vector import VectorStore

from common.logging_config import setup_logging

logger = setup_logging(__name__)

# 克隆仓库的存储位置
REPOS_DIR = Path(__file__).parent.parent / "repos"

SEARCH_TOOLS = [
    {
        "name": "search_code",
        "description": (
            "在已索引代码库中进行语义搜索。适用于概念性问题，"
            "例如“路由是如何工作的？”或“身份验证在哪里处理？”"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "自然语言搜索查询",
                },
                "repo": {
                    "type": "string",
                    "description": "将搜索范围限定到指定仓库（可选）",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "grep",
        "description": (
            "使用正则表达式在仓库文件中进行精确模式搜索。"
            "适用于查找指定标识符、TODO 或精确字符串。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "要搜索的正则表达式模式",
                },
                "repo": {
                    "type": "string",
                    "description": "将搜索范围限定到指定仓库（可选）",
                },
            },
            "required": ["pattern"],
        },
    },
]


def execute_search_code(
    vector_store: VectorStore, embedder: Embedder, tool_input: dict[str, Any]
) -> str:
    """在已索引代码库中进行语义搜索。"""
    query = tool_input["query"]
    repo = tool_input.get("repo")

    query_embedding = embedder.embed_query(query)
    results = vector_store.search(
        query_embedding=query_embedding,
        collection_name=repo,
        n_results=5,
    )

    if not results:
        return f"未找到与以下查询相关的结果：{query}"

    parts = [f"“{query}”的搜索结果：\n"]
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        score = 1 - r["distance"]  # 将距离转换为相似度
        parts.append(
            f"### 结果 {i}（相关度：{score:.2f}）\n"
            f"**{meta['filepath']}** 第 {meta['start_line']}～{meta['end_line']} 行 "
            f"[{r['collection']}]\n"
            f"```\n{r['content'][:500]}\n```\n"
        )

    return "\n".join(parts)


def execute_grep(vector_store: VectorStore, _embedder: Embedder, tool_input: dict[str, Any]) -> str:
    """使用正则表达式在仓库文件中搜索。"""
    pattern = tool_input["pattern"]
    repo = tool_input.get("repo")

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"无效的正则表达式模式：{e}"

    # 确定要搜索的仓库
    if repo:
        search_dirs = [REPOS_DIR / repo]
    else:
        search_dirs = [d for d in REPOS_DIR.iterdir() if d.is_dir()] if REPOS_DIR.exists() else []

    if not search_dirs:
        return "没有可用的仓库。请先使用 clone_and_index。"

    matches: list[str] = []
    context_lines = 2

    for repo_dir in search_dirs:
        if not repo_dir.exists():
            continue
        for filepath in repo_dir.rglob("*"):
            if not filepath.is_file() or filepath.suffix not in INDEXABLE_EXTENSIONS:
                continue

            try:
                lines = filepath.read_text(encoding="utf-8", errors="ignore").split("\n")
            except Exception:
                continue

            for i, line in enumerate(lines):
                if regex.search(line):
                    rel_path = filepath.relative_to(repo_dir)
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    context = "\n".join(
                        f"{'>' if j == i else ' '} {j + 1:4d} | {lines[j]}"
                        for j in range(start, end)
                    )
                    matches.append(f"**{rel_path}:{i + 1}**\n```\n{context}\n```")

                    if len(matches) >= 20:
                        break
            if len(matches) >= 20:
                break
        if len(matches) >= 20:
            break

    if not matches:
        return f"未找到与以下模式匹配的结果：{pattern}"

    header = f"找到 {len(matches)} 个与 `{pattern}` 匹配的结果"
    if len(matches) >= 20:
        header += "（显示前 20 个）"
    return header + "\n\n" + "\n\n".join(matches)
