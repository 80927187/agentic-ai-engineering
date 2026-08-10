"""
评测框架教程共用的知识库和研究助手。

提供一个返回模拟响应的简单研究助手，使教程无需 API 密钥即可专注于
各框架特有的评测模式。
"""

from typing import Any

from common import setup_logging

logger = setup_logging(__name__)

KNOWLEDGE_BASE = [
    {
        "id": "doc_001",
        "title": "微服务架构",
        "content": (
            "微服务架构将应用程序拆分为小型、独立的服务。每个服务都在自己的进程中运行，"
            "通过 API 通信，并可独立部署。其优势包括可扩展性、故障隔离和技术选型灵活性。"
        ),
        "tags": ["架构", "微服务", "分布式系统"],
    },
    {
        "id": "doc_002",
        "title": "REST API 设计",
        "content": (
            "REST API 遵循面向资源的设计原则。端点使用名词，操作使用 HTTP 方法，结果使用"
            "状态码。最佳实践包括版本控制、为集合提供分页以及采用一致的错误响应格式。"
        ),
        "tags": ["API", "REST", "设计"],
    },
    {
        "id": "doc_003",
        "title": "数据库索引",
        "content": (
            "数据库索引通过创建高效的查找结构来提升查询性能。B 树索引可处理等值查询和"
            "范围查询。复合索引支持多列查询，但列顺序很重要。索引过多会降低写入速度并"
            "浪费存储空间。"
        ),
        "tags": ["数据库", "性能", "索引"],
    },
    {
        "id": "doc_004",
        "title": "身份认证与权限授权",
        "content": (
            "身份认证用于验证身份，权限授权用于控制访问。JWT 令牌支持无状态身份认证，"
            "OAuth 2.0 支持委托访问。密码必须始终使用 bcrypt 或 argon2 进行哈希处理。"
        ),
        "tags": ["安全", "身份认证", "权限授权"],
    },
]

# 评测数据集：问题、预期答案及其元数据
EVAL_TASKS: list[dict[str, Any]] = [
    {
        "id": "task_001",
        "question": "微服务架构的主要优势有哪些？",
        "reference_answer": (
            "主要优势包括可扩展性、故障隔离以及服务可独立部署（doc_001）。"
        ),
        "expected_keywords": ["可扩展性", "故障隔离", "独立部署"],
        "expected_source_ids": ["doc_001"],
    },
    {
        "id": "task_002",
        "question": "REST API 的设计最佳实践有哪些？",
        "reference_answer": (
            "端点使用名词，操作使用 HTTP 方法，采用恰当的状态码、版本控制和分页（doc_002）。"
        ),
        "expected_keywords": ["名词", "端点", "HTTP 方法", "状态码"],
        "expected_source_ids": ["doc_002"],
    },
    {
        "id": "task_003",
        "question": "数据库索引如何提升查询性能？",
        "reference_answer": (
            "索引使用 B 树结构实现高效查找，减少全表扫描并加快查询速度（doc_003）。"
        ),
        "expected_keywords": ["B 树", "查找", "查询"],
        "expected_source_ids": ["doc_003"],
    },
    {
        "id": "task_004",
        "question": "身份认证与权限授权有什么区别？",
        "reference_answer": (
            "身份认证用于验证身份；权限授权用于控制访问权限。使用 JWT，并通过 bcrypt 对密码"
            "进行哈希处理（doc_004）。"
        ),
        "expected_keywords": ["身份", "访问", "身份认证", "权限授权"],
        "expected_source_ids": ["doc_004"],
    },
    {
        "id": "task_005",
        "question": "哪种编程语言最适合机器学习？",
        "reference_answer": "知识库不包含与机器学习编程语言有关的信息。",
        "expected_keywords": [],
        "expected_source_ids": [],
    },
]

# 模拟智能体响应（由所有框架教程共用）
SIMULATED_RESPONSES: dict[str, dict[str, Any]] = {
    "task_001": {
        "answer": (
            "根据 doc_001，微服务架构的主要优势包括可扩展性、故障隔离以及服务可独立部署。"
        ),
        "tool_calls": [{"name": "search_knowledge_base", "input": {"query": "微服务"}}],
        "sources": ["doc_001"],
    },
    "task_002": {
        "answer": (
            "根据 doc_002，REST API 的最佳实践包括端点使用名词、操作使用 HTTP 方法、采用"
            "恰当的状态码、版本控制和分页。"
        ),
        "tool_calls": [{"name": "search_knowledge_base", "input": {"query": "REST API"}}],
        "sources": ["doc_002"],
    },
    "task_003": {
        "answer": (
            "根据 doc_003，数据库索引利用高效的 B 树查找结构处理等值查询和范围查询，从而"
            "提升查询性能。"
        ),
        "tool_calls": [{"name": "search_knowledge_base", "input": {"query": "数据库索引"}}],
        "sources": ["doc_003"],
    },
    "task_004": {
        "answer": (
            "身份认证用于验证身份，而权限授权用于控制访问。JWT 令牌支持无状态身份认证；"
            "密码应使用 bcrypt 进行哈希处理（doc_004）。"
        ),
        "tool_calls": [{"name": "search_knowledge_base", "input": {"query": "认证授权"}}],
        "sources": ["doc_004"],
    },
    "task_005": {
        "answer": "我无法在知识库中找到与机器学习编程语言有关的信息。",
        "tool_calls": [{"name": "search_knowledge_base", "input": {"query": "机器学习"}}],
        "sources": [],
    },
}


def get_agent_response(task_id: str) -> dict[str, Any]:
    """返回给定任务 ID 对应的模拟智能体响应。"""
    return SIMULATED_RESPONSES.get(
        task_id,
        {"answer": "没有可用的响应。", "tool_calls": [], "sources": []},
    )


def search_knowledge_base(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """通过关键词匹配搜索知识库。"""
    query_terms = query.lower().split()
    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in KNOWLEDGE_BASE:
        searchable = f"{doc['title']} {doc['content']} {' '.join(doc['tags'])}".lower()
        score = sum(1 for term in query_terms if term in searchable)
        if score > 0:
            scored.append((score, {"id": doc["id"], "title": doc["title"], "score": score}))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:max_results]]
