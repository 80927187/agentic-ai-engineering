"""MemoryManager——协调三个记忆层。"""

import json

import anthropic
from common.logging_config import setup_logging

from .episodic import EpisodicMemory
from .models import MemoryEntry, MemoryType
from .semantic import SemanticMemory
from .working import WorkingMemory

logger = setup_logging(__name__)

# 从对话中提取重要项目的提示词
CONSOLIDATION_PROMPT = """\
分析以下对话，提取值得长期记住的重要信息。
返回一个 JSON 对象数组，每个对象包含：
- "content"：要记住的事实或事件（一个简洁的句子）
- "importance"：0.0-1.0 的浮点数（记住它有多重要？）
- "type"："episodic"（事件、互动、发生过的事）或 \
"semantic"（事实、偏好、知识）

只提取真正重要的信息。如果没有值得保存的内容，返回空数组 []。

对话：
{conversation}

只回复 JSON 数组，不要包含其他文本。"""


class MemoryManager:
    """协调工作记忆、情景记忆和语义记忆层。"""

    def __init__(self) -> None:
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()

    def remember(
        self,
        content: str,
        memory_type: str = "working",
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> str:
        """在指定层中存储记忆。"""
        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType(memory_type),
            importance=importance,
            metadata=metadata or {},
        )

        if memory_type == "working":
            self.working.add(content, importance, metadata)
        elif memory_type == "episodic":
            self.episodic.save(entry)
        elif memory_type == "semantic":
            self.semantic.save(entry)
        else:
            return f"未知记忆类型：{memory_type}"

        memory_label = {
            "working": "工作记忆",
            "episodic": "情景记忆",
            "semantic": "语义记忆",
        }[memory_type]
        return f"已记入{memory_label}：{content}"

    def recall(self, query: str, limit: int = 5) -> str:
        """跨层搜索——结合情景关键词和语义相似度结果。"""
        results: list[tuple[str, str, float]] = []  # (来源, 内容, 分数)

        # 情景关键词搜索
        episodic_matches = self.episodic.search(query, limit=limit)
        for entry in episodic_matches:
            results.append(("情景记忆", entry.content, entry.importance))

        # 语义相似度搜索
        semantic_matches = self.semantic.search(query, limit=limit)
        for entry, similarity in semantic_matches:
            # 按“相似度 × 重要性”排名
            score = similarity * entry.importance
            results.append(("语义记忆", entry.content, score))

        # 按分数降序排列
        results.sort(key=lambda x: x[2], reverse=True)
        results = results[:limit]

        if not results:
            return "未找到相关记忆。"

        lines = []
        for source, content, score in results:
            lines.append(f"[{source}]（分数：{score:.2f}）{content}")
        return "\n".join(lines)

    def forget(self, memory_id: str, memory_type: str) -> str:
        """从指定层删除特定记忆。"""
        if memory_type == "episodic":
            success = self.episodic.delete(memory_id)
        elif memory_type == "semantic":
            success = self.semantic.delete(memory_id)
        elif memory_type == "working":
            return "工作记忆会在会话结束时自动清除。"
        else:
            return f"未知记忆类型：{memory_type}"

        status = "已删除" if success else "未找到"
        memory_label = {"episodic": "情景记忆", "semantic": "语义记忆"}[memory_type]
        return f"{status}：{memory_label}中的 {memory_id}"

    def build_memory_context(self) -> str:
        """构建要注入系统提示词的记忆上下文字符串。"""
        sections: list[str] = []

        # 最近的情景记忆
        recent = self.episodic.get_recent(5)
        if recent:
            episodic_lines = [f"- {e.content}" for e in recent]
            sections.append("## 最近事件\n" + "\n".join(episodic_lines))

        # 最重要的语义记忆（最相关的常识）
        semantic_all = self.semantic.list_all()
        if semantic_all:
            # 按重要性排序，取排名最高的条目
            top = sorted(semantic_all, key=lambda e: e.importance, reverse=True)[:5]
            semantic_lines = [f"- {e.content}" for e in top]
            sections.append("## 已知事实\n" + "\n".join(semantic_lines))

        if not sections:
            return ""

        return "# 回忆起的记忆\n\n" + "\n\n".join(sections)

    def consolidate(
        self,
        conversation_messages: list[dict],
        client: anthropic.Anthropic,
        model: str,
    ) -> list[str]:
        """使用 LLM 从对话中提取重要项目并存入持久记忆。"""
        # 根据消息构建对话文本
        parts: list[str] = []
        for msg in conversation_messages:
            role = msg["role"]
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                text_parts = [
                    b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"
                ]
                if text_parts:
                    parts.append(f"{role}: {' '.join(text_parts)}")

        conversation_text = "\n".join(parts)
        if not conversation_text.strip():
            return []

        prompt = CONSOLIDATION_PROMPT.format(conversation=conversation_text)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=21333,
                messages=[{"role": "user", "content": prompt}],
            )
            text_parts = [block.text for block in response.content if block.type == "text"]
            if not text_parts:
                block_types = [block.type for block in response.content]
                raise ValueError(
                    f"模型响应中没有文本内容（stop_reason={response.stop_reason}，"
                    f"内容块={block_types}，output_tokens={response.usage.output_tokens}）。"
                )
            raw = "\n\n".join(text_parts).strip()

            # 如果存在 Markdown 代码围栏，则将其去除
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3].strip()

            # 从响应中解析 JSON 数组
            items = json.loads(raw)
            if not isinstance(items, list):
                return []

        except (json.JSONDecodeError, anthropic.APIError, ValueError) as e:
            logger.error("记忆整合失败：%s", e)
            return []

        saved: list[str] = []
        for item in items:
            content = item.get("content", "")
            importance = float(item.get("importance", 0.5))
            mem_type = item.get("type", "episodic")

            if mem_type not in ("episodic", "semantic"):
                mem_type = "episodic"

            self.remember(content, memory_type=mem_type, importance=importance)
            memory_label = {"episodic": "情景记忆", "semantic": "语义记忆"}[mem_type]
            saved.append(f"[{memory_label}] {content}")

        logger.info("已从对话中整合 %d 条记忆", len(saved))
        return saved

    def get_stats(self) -> dict:
        """汇总所有记忆层的统计信息。"""
        return {
            "working": self.working.stats(),
            "episodic": self.episodic.stats(),
            "semantic": self.semantic.stats(),
        }
