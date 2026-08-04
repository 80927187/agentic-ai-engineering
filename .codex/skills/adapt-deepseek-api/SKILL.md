---
name: adapt-deepseek-api
description: 将本项目原本调用 Claude 的 Python 提示词链适配为 DeepSeek API。用于切换模型、兼容 ThinkingBlock、保留但不发送 tool_choice、调大输出 token 与工具调用上限，以及处理并行网页搜索错误。
---

# 适配 DeepSeek API

先查看目标文件和当前 diff，只修改实际相关代码。

## 1. 切换模型

使用用户指定的 DeepSeek 模型：

```python
MODEL = "deepseek-v4-flash"
LIGHT_MODEL = "deepseek-v4-flash"
```

## 2. 保留但不发送 `tool_choice`

保留 `_call_llm` 形参、上层传参和原教学结构，只在最终请求中停止加入 `tool_choice`：

```python
kwargs: dict[str, Any] = {}
if tools:
    kwargs["tools"] = tools

# DeepSeek 思考模式不支持 tool_choice；保留形参和调用点用于对照学习。
# if tool_choice:
#     kwargs["tool_choice"] = tool_choice
```

确认 `messages.create(...)` 的参数中完全没有 `tool_choice`。

## 3. 兼容 `ThinkingBlock`

不要读取固定位置的 `.text`，只提取文本块：

```python
text_parts = [block.text for block in response.content if block.type == "text"]
if not text_parts:
    block_types = [block.type for block in response.content]
    raise ValueError(
        f"模型响应中没有文本内容（stop_reason={response.stop_reason}，"
        f"内容块={block_types}，output_tokens={response.usage.output_tokens}）。"
    )
return "\n\n".join(text_parts)
```

删除不再使用的 `cast` 导入。

## 4. 调大限制

搜索默认值和所有显式覆盖，统一提高。保留字段和明确数字，不要删除、改成 `None` 或无限循环。

```python
def _call_llm(..., max_tokens: int = 21333, ...):
    ...

response = self._call_llm(..., max_tokens=21333)

WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 100,
}

def run_agent(..., max_turns: int = 100):
    ...
```

- `max_tokens`：非流式 `messages.create(...)` 使用 `21333`，这是当前 SDK 10 分钟限制内的最大整数。
- `max_uses`：使用 `100` 或网关允许的更大值。
- `max_turns`、`max_iterations`、`max_steps`、`max_tool_calls`、`recursion_limit`：使用 `100` 或框架允许的更大值。
- 保留无工具调用、`end_turn`、重复调用和不可恢复错误等正常退出条件。

## 5. 处理搜索错误

```python
for result in block.content:
    if result.type == "web_search_result":
        searches.append({"title": result.title, "url": result.url})
    elif result.type == "web_search_tool_result_error":
        logger.warning("网络搜索失败：%s", result.error_code)
```

不要从错误对象读取 `title` 或 `url`。

## 6. 验证

1. 运行 `uv run python -m py_compile <目标文件>`。
2. 确认请求中没有 `tool_choice`，但原形参和调用点仍在。
3. 确认能跳过 `ThinkingBlock` 并取得文本。
4. 确认所有输出和工具循环限制都已调大，没有低值覆盖。
5. 用真实长输入完整运行，确认没有非流式长请求错误、`max_tokens`、`max_uses_exceeded` 或循环次数耗尽。
