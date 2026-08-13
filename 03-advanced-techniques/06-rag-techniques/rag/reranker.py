"""FlashRank 重排序器：轻量、仅使用 CPU，且无需 API 密钥。"""

import logging

from rag.chunker import Chunk

logger = logging.getLogger(__name__)


class Reranker:
    """使用 FlashRank 按文本块与查询的相关性重新排序。

    FlashRank 使用可在 CPU 上运行的小型 ONNX 模型（约 4MB），无需 API 密钥或
    GPU，非常适合教程和原型开发。
    """

    # FlashRank 支持列表中的多语言模型名称（注意没有 v2 后缀）。
    def __init__(self, model: str = "ms-marco-MultiBERT-L-12"):
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

        # 某些多语言 ONNX 模型在中文输入上会出现分数饱和（所有候选几乎同分）。
        # 此时模型排序没有判别力，保留上游 RRF 顺序通常更可靠。
        scores = [float(item["score"]) for item in results]
        if scores and max(scores) - min(scores) < 0.01:
            logger.info("重排序分数差异过小，保留 RRF 排名")
            return chunks[:top_k]

        # 映射回 Chunk 对象，并按重排序分数降序排列
        chunk_lookup = {c.id: c for c in chunks}
        reranked = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]:
            chunk = chunk_lookup.get(r["id"])
            if chunk:
                reranked.append(chunk)

        logger.info("已将 %d 个文本块重排序 → 取前 %d 个", len(chunks), len(reranked))
        return reranked
