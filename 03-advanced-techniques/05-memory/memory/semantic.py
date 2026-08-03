"""语义记忆——存储在 ChromaDB 向量数据库中的事实和知识。"""

import chromadb
from common.logging_config import setup_logging

from .models import MemoryEntry, MemoryType

logger = setup_logging(__name__)


class SemanticMemory:
    """以 ChromaDB 为后端、支持余弦相似度搜索的长期事实记忆。"""

    def __init__(self, persist_dir: str = "data/chroma") -> None:
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="semantic_memory",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB 已在 %s 初始化（%d 条记录）", persist_dir, self.collection.count())

    def save(self, entry: MemoryEntry) -> None:
        """保存记忆——由 ChromaDB 自动处理嵌入。"""
        entry.memory_type = MemoryType.SEMANTIC
        self.collection.add(
            ids=[entry.id],
            documents=[entry.content],
            metadatas=[
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "importance": entry.importance,
                    **{k: str(v) for k, v in entry.metadata.items()},
                }
            ],
        )
        logger.info("已保存语义记忆：%s", entry.content[:60])

    def search(self, query: str, limit: int = 5) -> list[tuple[MemoryEntry, float]]:
        """按语义相似度搜索——返回（条目，相似度分数）对。"""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(limit, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        entries: list[tuple[MemoryEntry, float]] = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            # 余弦距离 → 相似度：similarity = 1 - distance
            similarity = 1.0 - results["distances"][0][i]
            entry = MemoryEntry(
                id=results["ids"][0][i],
                content=results["documents"][0][i],
                memory_type=MemoryType.SEMANTIC,
                importance=float(metadata.get("importance", 0.5)),
                metadata={
                    k: v for k, v in metadata.items() if k not in ("timestamp", "importance")
                },
            )
            entries.append((entry, similarity))

        return entries

    def delete(self, memory_id: str) -> bool:
        """根据 ID 删除记忆。"""
        try:
            self.collection.delete(ids=[memory_id])
            logger.info("已删除语义记忆：%s", memory_id)
            return True
        except Exception as e:
            logger.error("删除语义记忆 %s 失败：%s", memory_id, e)
            return False

    def list_all(self) -> list[MemoryEntry]:
        """返回所有语义记忆。"""
        if self.collection.count() == 0:
            return []

        results = self.collection.get(include=["documents", "metadatas"])
        entries: list[MemoryEntry] = []
        for i in range(len(results["ids"])):
            metadata = results["metadatas"][i]
            entry = MemoryEntry(
                id=results["ids"][i],
                content=results["documents"][i],
                memory_type=MemoryType.SEMANTIC,
                importance=float(metadata.get("importance", 0.5)),
                metadata={
                    k: v for k, v in metadata.items() if k not in ("timestamp", "importance")
                },
            )
            entries.append(entry)
        return entries

    def clear(self) -> None:
        """通过重建集合来清除所有语义记忆。"""
        self.client.delete_collection("semantic_memory")
        self.collection = self.client.get_or_create_collection(
            name="semantic_memory",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("已清除所有语义记忆")

    def stats(self) -> dict:
        """返回语义记忆统计信息。"""
        return {
            "count": self.collection.count(),
            "collection": self.collection.name,
        }
