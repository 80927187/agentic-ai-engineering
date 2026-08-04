---
name: adapt-deepseek-api
description: 根据本项目的实际改造经验，将原本调用 Claude 的 Python 提示词链适配为 DeepSeek API。用于修改模型配置、兼容 ThinkingBlock、诊断思考阶段耗尽输出 token，以及处理 DeepSeek 并行网页搜索和搜索错误结果。
---

# 适配 DeepSeek API

先查看目标文件和当前 diff，只处理代码实际出现的问题，不扩展为 SDK 重构。

## 1. 切换模型

将主模型和轻量模型改为用户指定的 DeepSeek 模型：

```python
MODEL = "deepseek-v4-flash"
LIGHT_MODEL = "deepseek-v4-flash"
```

保留现有调用结构，先运行程序观察真实返回值。

## 2. 避免在思考模式下传入 `tool_choice`

DeepSeek 在思考模式下不支持 `tool_choice` 参数。启用思考模式时，从请求参数中完全省略 `tool_choice`；不要仅将其设为 `None`，以免 SDK 仍将该字段序列化并发送。若同一套调用代码还需支持非思考模式，则根据思考模式是否启用来条件性添加该参数。

## 3. 提取文本块

DeepSeek 响应可能先包含 `ThinkingBlock`，不能固定读取：

```python
response.content[0].text
```

只提取 `text` 类型内容：

```python
response = self._call_llm(system, messages)
text_parts = [block.text for block in response.content if block.type == "text"]
if not text_parts:
    block_types = [block.type for block in response.content]
    raise ValueError(
        f"模型响应中没有文本内容（stop_reason={response.stop_reason}，"
        f"内容块={block_types}，output_tokens={response.usage.output_tokens}）。"
    )
return "\n\n".join(text_parts)
```

删除因此不再使用的 `cast` 导入。

## 4. 快速诊断无文本响应

不要只报告“没有文本内容”。至少记录：

- `response.stop_reason`
- `response.content` 中的块类型
- `response.usage.output_tokens`

若出现 `stop_reason=max_tokens` 且内容块只有 `thinking`，说明 DeepSeek 在思考阶段耗尽了输出预算，不是文本提取代码漏掉了结果。适当提高 `max_tokens`，例如从 `2048` 提高到 `8192`，再用原始长输入复现验证。`max_tokens` 是上限，不会要求模型固定消耗这么多 token。

优先验证最容易触发问题的调用：长输入、复杂提示词以及最高温度的并行任务。不要仅用简短测试提示词判断适配成功。

## 5. 调整网页搜索次数

DeepSeek 可能根据多个大纲方向在一次响应中并行发起多次搜索。若 `max_uses=1`，后续搜索会返回 `max_uses_exceeded`。

先检查实际 `server_tool_use` 数量，再适当提高：

```python
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 10,
}
```

不要假设增加次数后永远不会失败。收集来源时区分正常结果和错误结果：

```python
for result in block.content:
    if result.type == "web_search_result":
        searches.append({"title": result.title, "url": result.url})
    elif result.type == "web_search_tool_result_error":
        logger.warning("网络搜索失败：%s", result.error_code)
```

错误对象没有 `title` 和 `url`，不要直接读取。

## 6. 验证

1. 运行 `uv run python -m py_compile <目标文件>`。
2. 确认思考模式的请求中没有 `tool_choice` 字段。
3. 单独验证大纲阶段能跳过 `ThinkingBlock` 并取得文本。
4. 使用真实长输入和最高温度检查 `stop_reason`、内容块类型与输出 token，确认不会在思考阶段耗尽预算。
5. 使用真实写作提示词检查搜索调用数和 `error_code`。
6. 完整运行提示词链，确认各阶段都取得非空文本。
