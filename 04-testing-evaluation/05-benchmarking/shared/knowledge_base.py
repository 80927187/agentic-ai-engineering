"""
共享知识库、工具定义、系统提示词和搜索函数。

供本教程模块中的所有基准测试脚本使用。
"""

from typing import Any

from common import setup_logging

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# 知识库（研究助手共享语料库）
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    {
        "id": "doc_001",
        "title": "微服务架构",
        "content": (
            "微服务架构将应用程序拆分为小型、独立的服务。每项服务都在自己的进程中运行，"
            "通过 API 通信，并且可以独立部署。其优势包括可扩展性、故障隔离和技术灵活性。"
            "其挑战包括分布式系统的复杂性、数据一致性和运维开销。"
        ),
        "tags": ["架构", "微服务", "分布式系统"],
    },
    {
        "id": "doc_002",
        "title": "REST API 设计",
        "content": (
            "REST API 遵循面向资源的设计原则。端点使用名词，操作使用 HTTP 方法，"
            "结果使用状态码。最佳实践包括版本控制、集合分页和统一的错误响应格式。"
        ),
        "tags": ["API", "REST", "设计"],
    },
    {
        "id": "doc_003",
        "title": "数据库索引",
        "content": (
            "数据库索引通过创建高效的查找结构来提高查询性能。B 树索引可处理等值查询和"
            "范围查询。复合索引支持多列查询，但列的顺序很重要。索引过多会降低写入速度"
            "并浪费存储空间。"
        ),
        "tags": ["数据库", "性能", "索引"],
    },
    {
        "id": "doc_004",
        "title": "身份认证与授权",
        "content": (
            "身份认证用于验证身份，授权用于控制访问权限。JWT 令牌支持无状态身份认证。"
            "OAuth 2.0 提供委托访问。密码应始终使用 bcrypt 或 argon2 进行哈希处理。"
        ),
        "tags": ["安全", "身份认证", "授权"],
    },
    {
        "id": "doc_005",
        "title": "CI/CD 流水线",
        "content": (
            "CI 会在每次提交时自动构建和测试代码。CD 会自动部署通过测试的构建。"
            "关键实践包括快速反馈循环、基于主干的开发、功能开关和自动回滚。"
        ),
        "tags": ["DevOps", "CI/CD", "自动化"],
    },
    {
        "id": "doc_006",
        "title": "使用 Kubernetes 编排容器",
        "content": (
            "Kubernetes 管理容器化工作负载。核心概念包括 Pod、Service、Deployment、"
            "ConfigMap 和 Secret。主要功能包括自动扩缩容、自愈、滚动更新和服务发现。"
        ),
        "tags": ["DevOps", "Kubernetes", "容器"],
    },
    {
        "id": "doc_007",
        "title": "事件驱动架构",
        "content": (
            "事件驱动架构使用事件触发服务之间的通信。常见模式包括事件溯源、CQRS 和"
            "发布/订阅。其优势包括松耦合、可扩展性和审计跟踪。"
        ),
        "tags": ["架构", "事件", "消息传递"],
    },
    {
        "id": "doc_008",
        "title": "缓存策略",
        "content": (
            "缓存通过在内存中存储经常访问的数据来降低延迟。常见策略包括旁路缓存、"
            "直写缓存和回写缓存。分布式缓存可使用 Redis 或 Memcached。"
        ),
        "tags": ["性能", "缓存", "Redis"],
    },
]

SYSTEM_PROMPT = (
    "你是一名研究助手。只能使用工具所提供的搜索结果来回答问题。"
    "始终使用文档 ID 标注信息来源。如果没有找到相关信息，请明确说明，不得编造信息。"
)

# Anthropic 工具格式
TOOLS_ANTHROPIC = [
    {
        "name": "search_knowledge_base",
        "description": "在知识库中搜索与查询匹配的文档。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
                "max_results": {
                    "type": "integer",
                    "description": "返回的最大文档数",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
]

# OpenAI 工具格式
TOOLS_OPENAI = [
    {
        "type": "function",
        "name": "search_knowledge_base",
        "description": "在知识库中搜索与查询匹配的文档。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
                "max_results": {
                    "type": "integer",
                    "description": "返回的最大文档数",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
]

# ---------------------------------------------------------------------------
# 基准测试任务（黄金数据集子集）
# ---------------------------------------------------------------------------

BENCHMARK_TASKS = [
    {
        "id": "bench_001",
        "question": "微服务架构有哪些主要优势？",
        "expected_keywords": ["可扩展性", "故障隔离", "独立"],
        "category": "架构",
    },
    {
        "id": "bench_002",
        "question": "应该如何设计 REST API 端点？",
        "expected_keywords": ["名词", "http 方法", "状态码"],
        "category": "API",
    },
    {
        "id": "bench_003",
        "question": "数据库索引有哪些策略？",
        "expected_keywords": ["b 树", "复合", "查询性能"],
        "category": "数据库",
    },
    {
        "id": "bench_004",
        "question": "请解释身份认证与授权之间的区别。",
        "expected_keywords": ["身份", "访问", "jwt", "oauth"],
        "category": "安全",
    },
    {
        "id": "bench_005",
        "question": "CI/CD 有哪些关键实践？",
        "expected_keywords": ["持续", "自动", "反馈"],
        "category": "DevOps",
    },
]

# ---------------------------------------------------------------------------
# 知识库搜索工具
# ---------------------------------------------------------------------------


def search_knowledge_base(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """使用关键词匹配搜索知识库。"""
    query_normalized = query.lower()
    query_words = set(query_normalized.split())
    scored: list[tuple[int, dict[str, Any]]] = []
    for doc in KNOWLEDGE_BASE:
        text = f"{doc['title']} {doc['content']} {' '.join(doc['tags'])}".lower()
        score = sum(1 for word in query_words if word in text)
        # 中文通常不使用空格分词，因此同时检查标签是否出现在完整查询中。
        score += sum(1 for tag in doc["tags"] if tag.lower() in query_normalized)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:max_results]]


def score_answer(answer: str, expected_keywords: list[str]) -> float:
    """根据预期关键词的覆盖率为回答评分。"""
    answer_normalized = "".join(answer.lower().split())
    found = sum(
        1 for kw in expected_keywords if "".join(kw.lower().split()) in answer_normalized
    )
    return found / len(expected_keywords) if expected_keywords else 1.0
