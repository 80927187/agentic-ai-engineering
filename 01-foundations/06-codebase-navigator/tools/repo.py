"""
仓库工具

用于克隆 GitHub 仓库和管理已索引代码库的工具。
"""

import subprocess
from pathlib import Path
from typing import Any

from indexer.chunker import chunk_repository, collect_files
from indexer.embedder import Embedder, index_chunks
from store.vector import VectorStore

from common.logging_config import setup_logging

logger = setup_logging(__name__)

# 克隆仓库的存储位置
REPOS_DIR = Path(__file__).parent.parent / "repos"

REPO_TOOLS = [
    {
        "name": "clone_and_index",
        "description": (
            "克隆 GitHub 仓库并建立索引，以便进行语义搜索。"
            "请提供类似“pallets/flask”的 GitHub 仓库名称或本地路径。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "GitHub 仓库（所有者/仓库）或本地路径",
                },
            },
            "required": ["repo"],
        },
    },
    {
        "name": "list_repos",
        "description": "列出所有已索引仓库及其代码块数量。",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


def _normalize_collection_name(repo: str) -> str:
    """将仓库标识符转换为有效的 ChromaDB 集合名称。"""
    # ChromaDB 要求：3～63 个字符，以字母或数字开头和结尾，且只能包含字母、数字、下划线和连字符
    name = repo.replace("/", "-").replace(".", "-").replace(" ", "-")
    # 确保名称以字母或数字开头
    if name and not name[0].isalnum():
        name = "r-" + name
    # 截断为 63 个字符
    return name[:63]


def _resolve_repo_path(repo: str) -> tuple[Path, str, bool]:
    """将仓库解析为本地路径，返回（路径、集合名称、是否需要克隆）。"""
    local_path = Path(repo).expanduser()
    if local_path.is_dir():
        name = "local-" + local_path.name
        return local_path, _normalize_collection_name(name), False

    # 将其视为 GitHub 仓库
    name = _normalize_collection_name(repo)
    clone_dir = REPOS_DIR / name
    return clone_dir, name, not clone_dir.exists()


def execute_clone_and_index(
    vector_store: VectorStore, embedder: Embedder, tool_input: dict[str, Any]
) -> str:
    """克隆仓库并建立索引，以便进行语义搜索。"""
    repo = tool_input["repo"]
    repo_path, collection_name, needs_clone = _resolve_repo_path(repo)

    # 检查是否已建立索引
    if vector_store.collection_exists(collection_name):
        collections = vector_store.list_collections()
        for c in collections:
            if c["name"] == collection_name:
                return f"仓库“{repo}”已建立索引（{c['chunks']} 个代码块），现在可以搜索了！"

    # 根据需要克隆仓库
    if needs_clone:
        REPOS_DIR.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{repo}.git"
        logger.info("正在将 %s 克隆到 %s", url, repo_path)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(repo_path)],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            return f"克隆“{repo}”失败：{e.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return f"克隆“{repo}”超时。"

    # 统计文件并进行分块
    files = collect_files(repo_path)
    chunks = chunk_repository(repo_path, collection_name)

    if not chunks:
        return f"在“{repo}”中未找到可建立索引的文件。"

    # 生成嵌入并存储
    count = index_chunks(embedder, vector_store, collection_name, chunks)

    return (
        f"已为“{repo}”建立索引：{len(files)} 个文件，{count} 个代码块。"
        f"现在可以搜索了！请尝试提出有关代码库的问题。"
    )


def execute_list_repos(vector_store: VectorStore, _tool_input: dict[str, Any]) -> str:
    """列出所有已索引的仓库。"""
    collections = vector_store.list_collections()
    if not collections:
        return "尚未为任何仓库建立索引。请使用 clone_and_index 添加仓库。"

    lines = ["已索引的仓库："]
    for c in collections:
        lines.append(f"  - {c['name']}：{c['chunks']} 个代码块")
    return "\n".join(lines)
