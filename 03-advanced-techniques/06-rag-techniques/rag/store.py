"""带有 BM25 关键词索引的 ChromaDB 向量存储。"""

import logging
import re

import jieba

import bm25s
import chromadb

from rag.chunker import Chunk
from rag.embedder import LocalEmbedder

logger = logging.getLogger(__name__)


def _mixed_tokens(text: str) -> str:
    """把中英混合文本转换成 BM25 可处理的空格分隔 token。

    英文按单词处理，中文先用 jieba 切词，同时保留相邻双字作为未登录词兜底，
    既能匹配“用户”，也能匹配没有空格的“用户的api有哪些”。
    """
    text = text.lower()
    tokens: list[str] = []
    for piece in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text):
        if re.fullmatch(r"[a-z0-9_]+", piece):
            tokens.append(piece)
        else:
            words = [word for word in jieba.lcut(piece, cut_all=False) if word.strip()]
            tokens.extend(words)
            # 双字片段只作为补充，避免纯字符 token 主导 BM25 排名。
            tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
    return " ".join(tokens)


class VectorStore:
    """双索引存储：ChromaDB 用于向量搜索，BM25 用于关键词搜索。"""

    def __init__(self, embedder: LocalEmbedder, persist_dir: str | None = None):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._chunk_lookup: dict[str, Chunk] = {}

        # ChromaDB：持久化存储或内存存储
        if persist_dir:
            self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.chroma_client = chromadb.Client()

        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

        # BM25：摄取文档后构建
        self.bm25: bm25s.BM25 | None = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """生成文本块的嵌入向量，并同时写入向量存储和 BM25 索引。"""
        if not chunks:
            return

        self.chunks = chunks
        self._chunk_lookup = {c.id: c for c in chunks}

        texts = [c.content for c in chunks]
        ids = [c.id for c in chunks]
        metadatas = [{"source": c.source, "chunk_index": c.chunk_index} for c in chunks]

        # 生成嵌入向量并添加到 ChromaDB
        embeddings = self.embedder.embed_documents(texts)
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("已在 ChromaDB 中索引 %d 个文本块", len(chunks))

        # 构建 BM25 索引
        tokenized = bm25s.tokenize(
            [_mixed_tokens(text) for text in texts], stopwords=None, show_progress=False
        )
        self.bm25 = bm25s.BM25()
        self.bm25.index(tokenized, show_progress=False)
        logger.info("已为 %d 个文本块构建 BM25 索引", len(chunks))

    def vector_search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """通过 ChromaDB 执行稠密向量相似度搜索。"""
        query_embedding = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(self.chunks)),
        )

        scored: list[tuple[Chunk, float]] = []
        if results["ids"] and results["ids"][0]:
            for chunk_id, distance in zip(results["ids"][0], results["distances"][0]):
                chunk = self._chunk_lookup.get(chunk_id)
                if chunk:
                    # ChromaDB 返回余弦距离，此处将其转换为相似度
                    similarity = 1.0 - distance
                    scored.append((chunk, similarity))

        return scored

    def keyword_search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """执行 BM25 关键词搜索。"""
        if self.bm25 is None or not self.chunks:
            return []

        tokenized_query = bm25s.tokenize(_mixed_tokens(query), stopwords=None, show_progress=False)
        results, scores = self.bm25.retrieve(tokenized_query, k=min(top_k, len(self.chunks)))

        scored: list[tuple[Chunk, float]] = []
        for idx, score in zip(results[0], scores[0]):
            if 0 <= idx < len(self.chunks) and score > 0:
                scored.append((self.chunks[idx], float(score)))

        return scored

    @property
    def chunk_count(self) -> int:
        """已索引的文本块数量。"""
        return len(self.chunks)
