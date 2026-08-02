---
name: adapt-deepseek-api
description: 根据本项目的实际改造经验，将原本调用 Claude 的 Python 提示词链适配为 DeepSeek API。用于修改模型配置、兼容 ThinkingBlock，以及处理 DeepSeek 并行网页搜索和搜索错误结果。
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

## 2. 提取文本块

DeepSeek 响应可能先包含 `ThinkingBlock`，不能固定读取：

```python
response.content[0].text
```

只提取 `text` 类型内容：

```python
response = self._call_llm(system, messages)
text_parts = [block.text for block in response.content if block.type == "text"]
if not text_parts:
    raise ValueError("模型响应中没有文本内容。")
return "\n\n".join(text_parts)
```

删除因此不再使用的 `cast` 导入。

## 3. 调整网页搜索次数

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

## 4. 验证

1. 运行 `uv run python -m py_compile <目标文件>`。
2. 单独验证大纲阶段能跳过 `ThinkingBlock` 并取得文本。
3. 使用真实写作提示词检查搜索调用数和 `error_code`。
4. 完整运行提示词链，确认各阶段都取得非空文本。
