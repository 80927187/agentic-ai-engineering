"""采用递归切分和重叠策略进行文本分块。"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """带有来源元数据的文本块。"""

    content: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """该文本块的唯一标识符。"""
        return f"{self.source}:{self.chunk_index}"


# 按顺序尝试分隔符，优先在语义最完整的边界处切分
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


def recursive_split(
    text: str,
    source: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: list[str] | None = None,
) -> list[Chunk]:
    """沿自然边界递归切分文本，并保留重叠内容。

    依次尝试双换行、单换行、句末和空格；若均不可用，最后按字符硬切分。
    """
    if not text.strip():
        return []

    seps = separators or DEFAULT_SEPARATORS
    raw_chunks = _split_recursive(text, chunk_size, seps)

    # 在相邻文本块之间添加重叠内容
    chunks = []
    for i, raw in enumerate(raw_chunks):
        # 将上一个文本块的末尾作为重叠内容添加到当前块开头
        if i > 0 and chunk_overlap > 0:
            prev = raw_chunks[i - 1]
            overlap_text = prev[-chunk_overlap:]
            raw = overlap_text + raw

        start_char = (
            text.find(raw_chunks[i][:50]) if i == 0 else max(0, text.find(raw_chunks[i][:50]))
        )
        chunks.append(
            Chunk(
                content=raw.strip(),
                source=source,
                chunk_index=i,
                start_char=start_char,
                end_char=start_char + len(raw_chunks[i]),
            )
        )

    return [c for c in chunks if c.content]


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """使用粒度逐渐变细的分隔符递归切分文本。"""
    if len(text) <= chunk_size:
        return [text]

    # 依次尝试每个分隔符
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            result = []
            current = ""

            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        result.append(current)
                    # 如果单个片段超过 chunk_size，则使用粒度更细的分隔符继续切分
                    if len(part) > chunk_size:
                        remaining_seps = separators[separators.index(sep) + 1 :]
                        if remaining_seps:
                            result.extend(_split_recursive(part, chunk_size, remaining_seps))
                        else:
                            # 最后的备选方案：按字符硬切分
                            for j in range(0, len(part), chunk_size):
                                result.append(part[j : j + chunk_size])
                    else:
                        current = part

            if current:
                result.append(current)

            return result

    # 找不到分隔符时进行硬切分
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
