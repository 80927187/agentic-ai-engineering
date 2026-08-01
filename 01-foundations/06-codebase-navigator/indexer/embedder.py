"""
嵌入流程

使用 sentence-transformers 为代码块生成嵌入，并将其存储在 ChromaDB 中。
使用轻量级 all-MiniLM-L6-v2 模型在本地快速生成嵌入。
"""

from typing import Any

from sentence_transformers import SentenceTransformer

from common.logging_config import setup_logging

logger = setup_logging(__name__)

# 适合代码搜索的轻量级模型
MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """封装 sentence-transformers 以生成嵌入向量。"""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        logger.info("正在加载嵌入模型：%s", model_name)
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """为一组文本生成嵌入向量。"""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return list(embeddings.tolist())

    def embed_query(self, query: str) -> list[float]:
        """为单个搜索查询生成嵌入向量。"""
        return list(self.model.encode(query).tolist())


def index_chunks(
    embedder: Embedder,
    vector_store: Any,
    collection_name: str,
    chunks: list[dict[str, Any]],
) -> int:
    """生成代码块的嵌入向量并将其存入向量存储。"""
    if not chunks:
        return 0

    # 为 ChromaDB 准备数据
    ids = [f"{collection_name}:{c['filepath']}:{c['start_line']}" for c in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [
        {
            "filepath": c["filepath"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "repo": c["repo"],
        }
        for c in chunks
    ]

    # 分批生成嵌入向量
    logger.info("正在为 %d 个代码块生成嵌入向量……", len(chunks))
    if len(chunks) > 1000:
        logger.info("检测到大型仓库——此过程可能需要几分钟……")
    batch_size = 128
    all_embeddings: list[list[float]] = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        all_embeddings.extend(embedder.embed(batch))

    # 存入向量数据库
    vector_store.add_chunks(
        collection_name=collection_name,
        ids=ids,
        documents=documents,
        embeddings=all_embeddings,
        metadatas=metadatas,
    )

    return len(chunks)
