"""结合向量搜索、BM25 和重排序的混合检索器。"""

import logging

from rag.chunker import Chunk
from rag.reranker import Reranker
from rag.store import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """通过倒数排名融合与重排序来结合向量搜索和 BM25。"""

    def __init__(self, store: VectorStore, reranker: Reranker | None = None):
        self.store = store
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 5, candidates: int = 20) -> list[Chunk]:
        """完整检索流水线：向量搜索 + BM25 → RRF → 重排序 → top_k。

        分别从两种检索方法中获取 `candidates` 个候选结果，使用 RRF 融合，
        然后按需重排序，生成最终的 top_k 个结果。
        """
        vector_results = self.store.vector_search(query, top_k=candidates)
        keyword_results = self.store.keyword_search(query, top_k=candidates)

        logger.info(
            "查询“%s”检索到 %d 个向量结果和 %d 个关键词结果",
            query[:60],
            len(vector_results),
            len(keyword_results),
        )

        # 使用倒数排名融合合并结果
        fused = self._reciprocal_rank_fusion(vector_results, keyword_results)
        fused_chunks = [chunk for chunk, _ in fused]

        # 如果配置了重排序器，则对结果重排序
        if self.reranker and fused_chunks:
            return self.reranker.rerank(query, fused_chunks, top_k=top_k)

        return fused_chunks[:top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[tuple[Chunk, float]],
        keyword_results: list[tuple[Chunk, float]],
        k: int = 60,
    ) -> list[tuple[Chunk, float]]:
        """使用倒数排名融合（Reciprocal Rank Fusion，RRF）合并排名列表。

        对结果出现过的所有列表求和：RRF 分数 = sum(1 / (k + rank))。
        k=60 是原始论文采用的标准常量。
        """
        scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, (chunk, _) in enumerate(vector_results):
            scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (k + rank + 1)
            chunk_map[chunk.id] = chunk

        for rank, (chunk, _) in enumerate(keyword_results):
            scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (k + rank + 1)
            chunk_map[chunk.id] = chunk

        # 按融合分数降序排列
        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        return [(chunk_map[cid], scores[cid]) for cid in sorted_ids]
