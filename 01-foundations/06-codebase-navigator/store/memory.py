"""
记忆存储

基于 JSON 的持久化记忆，用于跨会话存储事实、见解和偏好。
这是增强型 LLM 模式中的“记忆”增强能力。
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.logging_config import setup_logging

logger = setup_logging(__name__)

# 默认记忆文件位置
DEFAULT_MEMORY_PATH = Path(__file__).parent.parent / "memory.json"


class MemoryStore:
    """以 JSON 文件为后端的持久化记忆存储。"""

    def __init__(self, path: Path = DEFAULT_MEMORY_PATH) -> None:
        self.path = path
        self.data: dict[str, list[dict[str, Any]]] = {
            "facts": [],
            "insights": [],
            "preferences": [],
        }
        self._load()

    def _load(self) -> None:
        """从磁盘加载记忆。"""
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
                total = sum(len(v) for v in self.data.values())
                logger.info("已从 %s 加载 %d 条记忆", self.path, total)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error("加载记忆文件失败：%s", e)
                self.data = {"facts": [], "insights": [], "preferences": []}

    def _save(self) -> None:
        """将记忆持久化到磁盘。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def save(self, category: str, content: str, repos: list[str] | None = None) -> str:
        """保存一条记忆。"""
        if category not in self.data:
            return f"无效类别：{category}。请使用 facts、insights 或 preferences"

        entry: dict[str, Any] = {
            "content": content,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        if repos:
            entry["repos"] = repos

        self.data[category].append(entry)
        self._save()
        logger.info("已保存 %s：%s", category, content[:80])
        return f"已保存 {category}：{content}"

    def recall(self, query: str | None = None) -> dict[str, list[dict[str, Any]]]:
        """回忆记忆，可以选择按关键词筛选。"""
        if not query:
            return self.data

        query_lower = query.lower()
        filtered: dict[str, list[dict[str, Any]]] = {}
        for category, entries in self.data.items():
            matches = [e for e in entries if query_lower in e["content"].lower()]
            if matches:
                filtered[category] = matches
        return filtered

    def summary(self) -> str:
        """返回一份简要摘要，以便加入系统提示词。"""
        parts = []
        category_names = {
            "facts": "事实",
            "insights": "见解",
            "preferences": "偏好",
        }
        for category, entries in self.data.items():
            if entries:
                parts.append(
                    f"{category_names.get(category, category)}（{len(entries)} 条）："
                    + "；".join(e["content"] for e in entries[-3:])
                )
        return "\n".join(parts) if parts else "尚未保存任何记忆。"
