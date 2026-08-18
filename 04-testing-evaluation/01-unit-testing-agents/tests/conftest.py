"""
集成测试共用的测试夹具和响应录制基础设施。

提供用于记录/重放测试的 CassetteClient、预构建的录制数据，
以及各测试模块共用的夹具。
"""

import json
from pathlib import Path
from typing import Any

import pytest
from common import setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# 响应录制系统——记录并重放 API 响应
# ---------------------------------------------------------------------------


class CassetteResponse:
    """重建后的响应对象，模拟 Anthropic API 响应的结构。"""

    def __init__(self, data: dict[str, Any]) -> None:
        self.stop_reason = data["stop_reason"]
        self.content = [_AttrDict(block) for block in data["content"]]
        self.usage = _AttrDict(
            {
                "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                "output_tokens": data.get("usage", {}).get("output_tokens", 0),
                "cache_read_input_tokens": data.get("usage", {}).get("cache_read_input_tokens"),
                "cache_creation_input_tokens": data.get("usage", {}).get(
                    "cache_creation_input_tokens"
                ),
            }
        )


class _AttrDict:
    """将字典键公开为属性的轻量对象。"""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        for key, value in data.items():
            setattr(self, key, value)


class CassetteClient:
    """从录制文件中重放响应的虚假 Anthropic 客户端。"""

    def __init__(self, cassette_path: Path) -> None:
        with cassette_path.open(encoding="utf-8") as f:
            self._interactions = json.load(f)
        self._call_index = 0
        self.messages = self

    def create(self, **kwargs: Any) -> CassetteResponse:
        """重放下一条已记录的响应。"""
        if self._call_index >= len(self._interactions):
            raise RuntimeError(
                f"录制响应已用尽：最多应调用 API {len(self._interactions)} 次，"
                f"但当前已发起第 {self._call_index + 1} 次调用。"
                "代理行为已偏离录制内容。"
            )
        interaction = self._interactions[self._call_index]
        self._call_index += 1
        logger.info(
            "正在重放录制响应 %d/%d",
            self._call_index,
            len(self._interactions),
        )
        return CassetteResponse(interaction["response"])

    @property
    def calls_remaining(self) -> int:
        """尚未重放的已记录响应数量。"""
        return len(self._interactions) - self._call_index

def serialize_response(response: Any) -> dict[str, Any]:

    """将 Anthropic API 响应序列化为适合写入 JSON 的字典。"""
    content = []
    for block in response.content:
        if hasattr(block, "text"):
            content.append({"text": block.text})
        elif hasattr(block, "name") and hasattr(block, "input"):
            content.append({"id": block.id, "name": block.name, "input": block.input})
    return {
        "stop_reason": response.stop_reason,
        "content": content,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", None),
            "cache_creation_input_tokens": getattr(
                response.usage, "cache_creation_input_tokens", None
            ),
        },
    }


# ---------------------------------------------------------------------------
# 预构建的响应录制数据
# ---------------------------------------------------------------------------

# 录制数据：简单文本响应（不使用工具）
CASSETTE_TEXT_ONLY = [
    {
        "response": {
            "stop_reason": "end_turn",
            "content": [{"text": "你好！我可以帮你进行计算。"}],
            "usage": {"input_tokens": 120, "output_tokens": 15},
        }
    }
]

# 录制数据：调用一次计算器工具，然后返回文本响应
CASSETTE_CALCULATOR = [
    {
        "response": {
            "stop_reason": "tool_use",
            "content": [
                {
                    "id": "toolu_01ABC",
                    "name": "calculator",
                    "input": {"operation": "multiply", "a": 12, "b": 15},
                }
            ],
            "usage": {"input_tokens": 150, "output_tokens": 40},
        }
    },
    {
        "response": {
            "stop_reason": "end_turn",
            "content": [{"text": "12 乘以 15 等于 180。"}],
            "usage": {"input_tokens": 200, "output_tokens": 12},
        }
    },
]

# 录制数据：多轮工具调用——连续调用两次计算器
CASSETTE_MULTI_TOOL = [
    {
        "response": {
            "stop_reason": "tool_use",
            "content": [
                {
                    "id": "toolu_01STEP1",
                    "name": "calculator",
                    "input": {"operation": "add", "a": 100, "b": 200},
                }
            ],
            "usage": {"input_tokens": 160, "output_tokens": 35},
        }
    },
    {
        "response": {
            "stop_reason": "tool_use",
            "content": [
                {
                    "id": "toolu_01STEP2",
                    "name": "calculator",
                    "input": {"operation": "multiply", "a": 300, "b": 2},
                }
            ],
            "usage": {"input_tokens": 220, "output_tokens": 38},
        }
    },
    {
        "response": {
            "stop_reason": "end_turn",
            "content": [{"text": "我先计算 100 + 200 = 300，再乘以 2，得到 600。"}],
            "usage": {"input_tokens": 280, "output_tokens": 25},
        }
    },
]

# 录制数据：被禁止的命令——大语言模型请求执行 rm，代理将其阻止，模型随后致歉
CASSETTE_BLOCKED_COMMAND = [
    {
        "response": {
            "stop_reason": "tool_use",
            "content": [
                {
                    "id": "toolu_01DANGER",
                    "name": "run_bash",
                    "input": {"command": "rm -rf /tmp/data"},
                }
            ],
            "usage": {"input_tokens": 140, "output_tokens": 30},
        }
    },
    {
        "response": {
            "stop_reason": "end_turn",
            "content": [{"text": "抱歉，出于安全考虑，该命令已被阻止。"}],
            "usage": {"input_tokens": 210, "output_tokens": 18},
        }
    },
]


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture()
def cassette_dir(tmp_path: Path) -> Path:
    """创建临时的响应录制目录。"""
    d = tmp_path / "cassettes"
    d.mkdir()
    return d


def write_cassette(cassette_dir: Path, name: str, data: list[dict[str, Any]]) -> Path:
    """写入响应录制文件并返回其路径。"""
    path = cassette_dir / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
