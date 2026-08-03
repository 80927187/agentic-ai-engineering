"""FlashRank 重排序器：轻量、仅使用 CPU，且无需 API 密钥。"""

import logging

from rag.chunker import Chunk

logger = logging.getLogger(__name__)


class Reranker:
    """使用 FlashRank 按文本块与查询的相关性重新排序。

    FlashRank 使用可在 CPU 上运行的小型 ONNX 模型（约 4MB），无需 API 密钥或
    GPU，非常适合教程和原型开发。
    """

    def __init__(self, model: str = "ms-marco-MiniLM-L-12-v2"):
        from flashrank import Ranker

        self.ranker = Ranker(model_name=model)
        logger.info("重排序器初始化完成，模型=%s", model)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        """按文本块与查询的相关性重新排序，并返回前 top_k 个结果。"""
        if not chunks:
            return []

        from flashrank import RerankRequest

        passages = [{"id": c.id, "text": c.content, "meta": {"source": c.source}} for c in chunks]
        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)

        # 映射回 Chunk 对象，并按重排序分数降序排列
        chunk_lookup = {c.id: c for c in chunks}
        reranked = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]:
            chunk = chunk_lookup.get(r["id"])
            if chunk:
                reranked.append(chunk)

        logger.info("已将 %d 个文本块重排序 → 取前 %d 个", len(chunks), len(reranked))
        return reranked
