"""三层记忆系统的共享类型。"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class MemoryType(Enum):
    """智能体记忆的三个层级。"""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class MemoryEntry:
    """包含内容、元数据和重要性分数的单条记忆。"""

    content: str
    memory_type: MemoryType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)
    importance: float = 0.5
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        """序列化为可安全转换为 JSON 的字典。"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """从字典反序列化。"""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            importance=data.get("importance", 0.5),
        )
