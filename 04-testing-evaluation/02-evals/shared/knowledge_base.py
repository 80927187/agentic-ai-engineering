"""共享的知识库、工具定义和搜索功能。

提供所有评估教程脚本共用的研究语料库、系统提示词和 Anthropic 工具模式。
"""

import re
from typing import Any

from common import setup_logging

logger = setup_logging(__name__)

_CHINESE_STOPWORDS = {"如何", "使用", "搭建", "前端", "有哪些", "哪种", "最适合"}

# ---------------------------------------------------------------------------
# 知识库语料
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    {
        "id": "doc_001",
        "title": "微服务架构",
        "content": (
            "微服务架构将应用程序拆分为小型、独立的服务。每个服务在自己的进程中运行，"
            "通过 API 通信，并且可以独立部署。其优势包括可扩展性、故障隔离和技术灵活性。"
            "其挑战包括分布式系统的复杂性、数据一致性和运维开销。"
        ),
        "tags": ["架构", "微服务", "分布式系统"],
    },
    {
        "id": "doc_002",
        "title": "REST API 设计",
        "content": (
            "REST API 遵循面向资源的设计原则。端点使用名词（例如 /users、/orders），"
            "操作使用 HTTP 方法（GET、POST、PUT、DELETE），结果使用状态码。最佳实践包括"
            "版本控制（例如 /v1/）、集合分页，以及统一的错误响应格式。"
        ),
        "tags": ["API", "REST", "设计"],
    },
    {
        "id": "doc_003",
        "title": "数据库索引",
        "content": (
            "数据库索引通过创建高效的查找结构来提升查询性能。B 树索引适用于等值查询和范围查询。"
            "复合索引支持多列查询，但列的顺序十分重要。索引过多会降低写入速度并浪费存储空间。"
            "可以使用 EXPLAIN 分析查询计划并发现缺失的索引。"
        ),
        "tags": ["数据库", "性能", "索引"],
    },
    {
        "id": "doc_004",
        "title": "身份认证与授权",
        "content": (
            "身份认证用于验证身份（你是谁），授权用于控制访问权限（你能做什么）。JWT 令牌支持"
            "基于声明的无状态身份认证和授权。OAuth 2.0 提供委托访问。密码必须始终使用 bcrypt "
            "或 argon2 进行哈希处理，并应实施速率限制和账户锁定来防止暴力破解攻击。"
        ),
        "tags": ["安全", "身份认证", "授权"],
    },
    {
        "id": "doc_005",
        "title": "CI/CD 流水线",
        "content": (
            "持续集成（CI）会在每次提交时自动构建和测试代码。持续部署（CD）会将通过测试的构建"
            "自动部署到生产环境。关键实践包括快速反馈循环、基于主干的开发、通过功能开关逐步发布，"
            "以及失败时自动回滚。常用工具包括 GitHub Actions、GitLab CI 和 Jenkins。"
        ),
        "tags": ["DevOps", "CI/CD", "自动化"],
    },
    {
        "id": "doc_006",
        "title": "使用 Kubernetes 编排容器",
        "content": (
            "Kubernetes 管理跨集群的容器化工作负载。核心概念包括 Pod（最小可部署单元）、"
            "Service（网络抽象）、Deployment（声明式更新），以及 ConfigMap/Secret（配置）。"
            "主要功能包括自动扩缩容、自愈、滚动更新和服务发现。"
        ),
        "tags": ["DevOps", "Kubernetes", "容器"],
    },
    {
        "id": "doc_007",
        "title": "事件驱动架构",
        "content": (
            "事件驱动架构使用事件来触发服务并在服务之间通信。常见模式包括事件溯源（以事件形式"
            "存储状态）、CQRS（读写分离）和发布/订阅消息。其优势包括松耦合、可扩展性和审计轨迹；"
            "挑战包括最终一致性、事件顺序，以及分布式流程的调试。"
        ),
        "tags": ["架构", "事件", "消息传递"],
    },
    {
        "id": "doc_008",
        "title": "缓存策略",
        "content": (
            "缓存通过在内存中存储频繁访问的数据来降低延迟和数据库负载。相关策略包括旁路缓存"
            "（由应用管理缓存）、写穿（写入时更新缓存）和写回（异步写入缓存）。分布式缓存可使用 "
            "Redis 或 Memcached。应设置适当的 TTL，并谨慎实现缓存失效机制。"
        ),
        "tags": ["性能", "缓存", "Redis"],
    },
]

# ---------------------------------------------------------------------------
# 系统提示词和工具定义
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是一名研究助手。只能使用工具返回的搜索结果回答问题。始终使用文档 ID 标注信息来源。"
    "如果没有找到相关信息，请明确说明。不要编造信息。"
)

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "在知识库中搜索与查询匹配的文档，返回相关文档及其内容。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用于查找相关文档的搜索查询",
                },
                "max_results": {
                    "type": "integer",
                    "description": "最多返回的文档数（默认值：3）",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
]


# ---------------------------------------------------------------------------
# 搜索函数
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """提取中英文检索词，避免中文连续文本被当成一个整词。"""
    tokens: set[str] = set()
    for segment in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            # 中文没有空格分隔；使用二、三字片段保留短语匹配能力。
            tokens.add(segment)
            tokens.update(segment[i : i + 2] for i in range(len(segment) - 1))
            tokens.update(segment[i : i + 3] for i in range(len(segment) - 2))
        else:
            tokens.add(segment)
    return {token for token in tokens if token not in _CHINESE_STOPWORDS}


def search_knowledge_base(
    query: str, max_results: int = 3, corpus: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """使用关键词匹配搜索知识库。"""
    docs = corpus if corpus is not None else KNOWLEDGE_BASE
    query_words = _tokenize(query)
    scored: list[tuple[int, dict[str, Any]]] = []
    for doc in docs:
        text = f"{doc['title']} {doc['content']} {' '.join(doc['tags'])}"
        score = len(query_words & _tokenize(text))
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:max_results]]
