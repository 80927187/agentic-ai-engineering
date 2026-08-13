"""本地 sentence-transformer 嵌入，无需 API 密钥。"""

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# 支持中文、英文及中英混合查询；比纯英文 MiniLM 更适合本示例文档。
# 首次运行会下载约 120MB。
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class LocalEmbedder:
    """使用本地 sentence-transformers 模型生成嵌入向量。"""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        logger.info("正在加载嵌入模型：%s", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info(
            "嵌入模型加载完成（维度=%d）", self.model.get_sentence_embedding_dimension()
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为文档生成用于索引的嵌入向量。"""
        if not texts:
            return []

        embeddings = self.model.encode(texts, show_progress_bar=False)
        logger.info("已为 %d 份文档生成嵌入向量", len(texts))
        return [e.tolist() for e in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """为搜索查询生成嵌入向量。"""
        embedding = self.model.encode(query)
        result: list[float] = embedding.tolist()
        return result
