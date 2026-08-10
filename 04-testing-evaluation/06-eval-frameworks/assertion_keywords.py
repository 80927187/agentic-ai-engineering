"""
用于评定关键词覆盖率的自定义 Promptfoo 断言。

Promptfoo 会为每个 `python` 类型的断言调用 get_assert()。
返回包含 pass、score 和 reason 的字典。
"""


def get_assert(output, context):
    """检查智能体输出的关键词覆盖率。"""
    metadata = context.get("test", {}).get("metadata", {})
    keywords = metadata.get("keywords", [])

    if not keywords:
        return {"pass": True, "score": 1.0, "reason": "没有需要检查的关键词"}

    output_lower = output.lower()
    found = [kw for kw in keywords if kw.lower() in output_lower]
    missing = [kw for kw in keywords if kw.lower() not in output_lower]

    score = len(found) / len(keywords)
    passed = score >= 0.5

    reason = f"找到 {len(found)}/{len(keywords)} 个关键词"
    if missing:
        reason += f"（缺少：{', '.join(missing)}）"

    return {"pass": passed, "score": score, "reason": reason}
