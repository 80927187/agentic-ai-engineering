"""
共享知识库、工具定义和工具执行逻辑。

供各追踪教程中的带追踪研究助手使用。
"""

from collections.abc import Callable
from typing import Any

from common import setup_logging

logger = setup_logging(__name__)

KNOWLEDGE_BASE = [
    {
        "id": "doc_001",
        "title": "微服务架构",
        "content": (
            "微服务架构将应用程序拆分为小型、独立的服务。每个服务都在自己的进程中运行，"
            "通过 API 通信，并且可以独立部署。其优势包括可扩展性、故障隔离和技术灵活性。"
            "其挑战包括分布式系统的复杂性、数据一致性和运维开销。"
        ),
        "tags": ["architecture", "microservices", "distributed-systems"],
    },
    {
        "id": "doc_002",
        "title": "REST API 设计",
        "content": (
            "REST API 遵循面向资源的设计原则。端点使用名词，操作使用 HTTP 方法，"
            "结果使用状态码。最佳实践包括版本控制、集合分页和一致的错误响应格式。"
        ),
        "tags": ["api", "rest", "design"],
    },
    {
        "id": "doc_003",
        "title": "数据库索引",
        "content": (
            "数据库索引通过创建高效的查找结构来提升查询性能。B 树索引适用于等值查询和"
            "范围查询。复合索引支持多列查询，但列的顺序很重要。索引过多会拖慢写入并浪费"
            "存储空间。可以使用 EXPLAIN 分析查询计划。"
        ),
        "tags": ["database", "performance", "indexing"],
    },
    {
        "id": "doc_004",
        "title": "身份认证与授权",
        "content": (
            "身份认证用于验证身份，授权用于控制访问。JWT 令牌支持无状态身份认证，"
            "OAuth 2.0 提供委托访问。密码应始终使用 bcrypt 或 argon2 进行哈希处理，"
            "并实现速率限制和账户锁定。"
        ),
        "tags": ["security", "authentication", "authorization"],
    },
    {
        "id": "doc_005",
        "title": "CI/CD 流水线",
        "content": (
            "CI 会在每次提交时自动构建和测试代码。CD 会自动部署通过测试的构建。"
            "关键实践包括快速反馈循环、基于主干的开发、功能开关和自动回滚。"
        ),
        "tags": ["devops", "ci-cd", "automation"],
    },
    {
        "id": "doc_006",
        "title": "使用 Kubernetes 编排容器",
        "content": (
            "Kubernetes 管理容器化工作负载。核心概念包括 Pod、Service、Deployment、"
            "ConfigMap 和 Secret。主要功能包括自动扩缩容、自愈、滚动更新和服务发现。"
        ),
        "tags": ["devops", "kubernetes", "containers"],
    },
    {
        "id": "doc_007",
        "title": "事件驱动架构",
        "content": (
            "事件驱动架构使用事件触发服务之间的通信。常见模式包括事件溯源、CQRS 和发布/订阅。"
            "其优势包括松耦合、可扩展性和审计跟踪；挑战包括最终一致性和事件顺序。"
        ),
        "tags": ["architecture", "events", "messaging"],
    },
    {
        "id": "doc_008",
        "title": "缓存策略",
        "content": (
            "缓存通过在内存中存储频繁访问的数据来降低延迟。常见策略包括旁路缓存、"
            "写穿和写回。可以使用 Redis 或 Memcached。应设置适当的 TTL 并实现缓存失效机制。"
        ),
        "tags": ["performance", "caching", "redis"],
    },
]

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "在知识库中搜索与查询匹配的文档。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
                "max_results": {"type": "integer", "description": "最大结果数", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_document",
        "description": "根据 ID 获取指定文档。",
        "input_schema": {
            "type": "object",
            "properties": {"doc_id": {"type": "string", "description": "文档 ID"}},
            "required": ["doc_id"],
        },
    },
]

SYSTEM_PROMPT = (
    "你是一名研究助手。回答前请使用 search_knowledge_base 和 get_document 工具查找信息。"
    "回答必须始终以找到的文档为依据。如果找不到相关文档，请明确说明。"
)


def search_knowledge_base(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """将查询词与标题、内容和标签匹配，以搜索知识库。"""
    query_terms = query.lower().split()
    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in KNOWLEDGE_BASE:
        searchable = f"{doc['title']} {doc['content']} {' '.join(doc['tags'])}".lower()
        score = sum(1 for term in query_terms if term in searchable)
        if score > 0:
            scored.append((score, {"id": doc["id"], "title": doc["title"], "score": score}))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:max_results]]


def get_document(doc_id: str) -> dict[str, Any]:
    """根据 ID 获取文档。"""
    for doc in KNOWLEDGE_BASE:
        if doc["id"] == doc_id:
            return doc
    return {"error": f"未找到文档：{doc_id}"}


TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "search_knowledge_base": search_knowledge_base,
    "get_document": get_document,
}


def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> Any:
    """执行工具并返回结果。"""
    if tool_name not in TOOL_FUNCTIONS:
        return {"error": f"未知工具：{tool_name}"}
    try:
        return TOOL_FUNCTIONS[tool_name](**tool_input)
    except Exception as e:
        logger.error("工具执行错误：%s", e)
        return {"error": str(e)}
