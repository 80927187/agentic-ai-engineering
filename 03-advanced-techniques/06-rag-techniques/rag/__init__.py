"""RAG 流水线组件：分块、嵌入、存储、检索和重排序。"""

import logging
import os

# 在第三方库初始化前关闭冗余日志和进度条
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["SAFETENSORS_LOG_LEVEL"] = "error"
for _lib in (
    "sentence_transformers",
    "transformers",
    "torch",
    "huggingface_hub",
    "chromadb",
    "bm25s",
    "safetensors",
):
    logging.getLogger(_lib).setLevel(logging.ERROR)

from rag.chunker import Chunk, recursive_split  # noqa: E402
from rag.embedder import LocalEmbedder  # noqa: E402
from rag.reranker import Reranker  # noqa: E402
from rag.retriever import HybridRetriever  # noqa: E402
from rag.store import VectorStore  # noqa: E402

__all__ = [
    "Chunk",
    "HybridRetriever",
    "Reranker",
    "VectorStore",
    "LocalEmbedder",
    "recursive_split",
]
