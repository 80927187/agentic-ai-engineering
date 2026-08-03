"""工作记忆——基于重要性淘汰的会话内缓冲区。"""

from common.logging_config import setup_logging

from .models import MemoryEntry, MemoryType

logger = setup_logging(__name__)


class WorkingMemory:
    """会话级缓冲区，容量已满时淘汰重要性最低的条目。"""

    def __init__(self, max_items: int = 50) -> None:
        self.max_items = max_items
        self._entries: list[MemoryEntry] = []

    def add(
        self,
        content: str,
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> MemoryEntry:
        """添加一条记忆；如果已达容量上限，则淘汰最不重要的条目。"""
        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType.WORKING,
            importance=importance,
            metadata=metadata or {},
        )

        if len(self._entries) >= self.max_items:
            # 淘汰重要性最低的条目
            self._entries.sort(key=lambda e: e.importance)
            evicted = self._entries.pop(0)
            logger.info("已淘汰工作记忆：%s", evicted.content[:60])

        self._entries.append(entry)
        return entry

    def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        """返回最近的 N 条记忆。"""
        return sorted(self._entries, key=lambda e: e.timestamp, reverse=True)[:n]

    def get_important(self, threshold: float = 0.7) -> list[MemoryEntry]:
        """返回重要性超过阈值的条目。"""
        return [e for e in self._entries if e.importance >= threshold]

    def get_all(self) -> list[MemoryEntry]:
        """返回所有按时间戳排序的条目。"""
        return sorted(self._entries, key=lambda e: e.timestamp)

    def clear(self) -> None:
        """清除所有工作记忆。"""
        count = len(self._entries)
        self._entries.clear()
        logger.info("已清除 %d 条工作记忆", count)

    def stats(self) -> dict:
        """返回工作记忆统计信息。"""
        return {
            "count": len(self._entries),
            "max_items": self.max_items,
            "avg_importance": (
                sum(e.importance for e in self._entries) / len(self._entries)
                if self._entries
                else 0.0
            ),
        }
