"""情景记忆——持久化到 JSON 文件的带时间戳事件。"""

import json
from pathlib import Path

from common.logging_config import setup_logging

from .models import MemoryEntry, MemoryType

logger = setup_logging(__name__)


class EpisodicMemory:
    """以 JSON 文件为后端的长期事件记忆。"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path("data/episodic.json")
        self._entries: list[MemoryEntry] = []
        self._load()

    def _load(self) -> None:
        """从磁盘加载记忆。"""
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._entries = [MemoryEntry.from_dict(d) for d in raw]
                logger.info("已从 %s 加载 %d 条情景记忆", self.path, len(self._entries))
            except (json.JSONDecodeError, KeyError) as e:
                logger.error("加载情景记忆失败：%s", e)
                self._entries = []
        else:
            logger.info("%s 中没有现有的情景记忆", self.path)

    def _save(self) -> None:
        """将记忆持久化到磁盘。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self._entries]
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def save(self, entry: MemoryEntry) -> None:
        """将记忆条目保存到情景记忆库。"""
        entry.memory_type = MemoryType.EPISODIC
        self._entries.append(entry)
        self._save()
        logger.info("已保存情景记忆：%s", entry.content[:60])

    def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """通过关键词匹配搜索记忆。"""
        query_lower = query.lower()
        query_words = query_lower.split()

        scored: list[tuple[MemoryEntry, int]] = []
        for entry in self._entries:
            content_lower = entry.content.lower()
            # 根据找到的查询词数量评分
            score = sum(1 for word in query_words if word in content_lower)
            if score > 0:
                scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:limit]]

    def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        """返回最近的 N 条情景记忆。"""
        return sorted(self._entries, key=lambda e: e.timestamp, reverse=True)[:n]

    def delete(self, memory_id: str) -> bool:
        """根据 ID 删除记忆。"""
        for i, entry in enumerate(self._entries):
            if entry.id == memory_id:
                self._entries.pop(i)
                self._save()
                logger.info("已删除情景记忆：%s", memory_id)
                return True
        return False

    def list_all(self) -> list[MemoryEntry]:
        """返回所有按时间戳排序的情景记忆。"""
        return sorted(self._entries, key=lambda e: e.timestamp)

    def clear(self) -> None:
        """清除所有情景记忆。"""
        count = len(self._entries)
        self._entries.clear()
        self._save()
        logger.info("已清除 %d 条情景记忆", count)

    def stats(self) -> dict:
        """返回情景记忆统计信息。"""
        return {
            "count": len(self._entries),
            "file": str(self.path),
            "oldest": self._entries[0].timestamp.isoformat() if self._entries else None,
            "newest": self._entries[-1].timestamp.isoformat() if self._entries else None,
        }
