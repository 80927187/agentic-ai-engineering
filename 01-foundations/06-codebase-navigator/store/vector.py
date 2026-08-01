"""
向量存储

基于 ChromaDB 的向量存储，用于对已索引代码库进行语义搜索。
这是增强型 LLM 模式中的“检索”增强能力。
"""

from pathlib import Path
from typing import Any

import chromadb

from common.logging_config import setup_logging

logger = setup_logging(__name__)

# 将 ChromaDB 数据持久化到本地目录
DEFAULT_CHROMA_PATH = str(Path(__file__).parent.parent / "data" / "chroma")


class VectorStore:
    """用于存储和查询代码嵌入向量的 ChromaDB 封装。"""

    def __init__(self, persist_dir: str = DEFAULT_CHROMA_PATH) -> None:
        self.client = chromadb.PersistentClient(path=persist_dir)
        logger.info("ChromaDB 已在 %s 初始化", persist_dir)

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """获取或创建仓库对应的集合。"""
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        collection_name: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """将带有预计算嵌入向量的代码块添加到集合。"""
        collection = self.get_or_create_collection(collection_name)
        # ChromaDB 有批次大小限制，每批添加 500 条
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )
        logger.info("已向集合“%s”添加 %d 个代码块", collection_name, len(ids))

    def search(
        self,
        query_embedding: list[float],
        collection_name: str | None = None,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """在一个或全部集合中搜索相似代码块。"""
        collections = (
            [self.client.get_collection(collection_name)]
            if collection_name
            else self.client.list_collections()
        )

        all_results: list[dict[str, Any]] = []
        for collection in collections:
            if collection.count() == 0:
                continue
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            for j in range(len(results["ids"][0])):
                all_results.append(
                    {
                        "id": results["ids"][0][j],
                        "content": results["documents"][0][j],
                        "metadata": results["metadatas"][0][j],
                        "distance": results["distances"][0][j],
                        "collection": collection.name,
                    }
                )

        # 按距离排序（使用余弦距离时，值越小表示越相似）
        all_results.sort(key=lambda x: x["distance"])
        return all_results[:n_results]

    def list_collections(self) -> list[dict[str, Any]]:
        """列出所有已索引仓库及其统计信息。"""
        result = []
        for collection in self.client.list_collections():
            result.append(
                {
                    "name": collection.name,
                    "chunks": collection.count(),
                }
            )
        return result

    def collection_exists(self, name: str) -> bool:
        """检查集合是否已存在。"""
        return any(c.name == name for c in self.client.list_collections())
