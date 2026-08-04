---
name: adapt-deepseek-api
description: 根据本项目的实际改造经验，将原本调用 Claude 的 Python 提示词链适配为 DeepSeek API。用于修改模型配置、保留 tool_choice 教学痕迹但避免发送、兼容 ThinkingBlock、调大输出 token 和工具调用上限，以及处理并行网页搜索错误。
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

## 2. 保留 `tool_choice` 痕迹，只在请求前停止发送

DeepSeek 在思考模式下不支持 `tool_choice` 参数。为保留 Claude 原始示例的教学和对照价值，不要删除以下痕迹：

- `_call_llm` 的 `tool_choice` 形参；
- `_plan` 等上层调用传入的 `tool_choice={...}`；
- 原代码中关于强制工具调用的结构。

只在构造最终请求参数时停止加入 `tool_choice`，并保留注释说明原因：

```python
kwargs: dict[str, Any] = {}
if tools:
    kwargs["tools"] = tools

# DeepSeek 思考模式不支持 tool_choice；保留形参和调用点用于对照学习。
# if tool_choice:
#     kwargs["tool_choice"] = tool_choice
```

不能把 `tool_choice` 设为 `None` 后仍传给 SDK，因为 SDK 可能继续序列化该字段。最终检查 `messages.create(...)` 收到的参数，确认其中完全没有 `tool_choice`。

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

## 4. 调大输出 token 上限并诊断无文本响应

将通用调用的 `max_tokens` 默认值调到 `8192`。同时搜索所有调用点；若大纲、规划或并行任务显式使用 `1024`、`2048`、`4096` 等较低值，也根据真实输入调到 `8192`，不要只改默认值而遗漏显式覆盖：

```python
def _call_llm(..., max_tokens: int = 8192, ...) -> anthropic.types.Message:
    ...

response = self._call_llm(..., max_tokens=8192)
```

不要只报告“没有文本内容”。至少记录：

- `response.stop_reason`
- `response.content` 中的块类型
- `response.usage.output_tokens`

若出现 `stop_reason=max_tokens` 且内容块只有 `thinking`，说明 DeepSeek 在思考阶段耗尽了输出预算，不是文本提取代码漏掉了结果。继续根据真实长输入提高预算并复现验证。`max_tokens` 是上限，不会要求模型固定消耗这么多 token。

优先验证最容易触发问题的调用：长输入、复杂提示词以及最高温度的并行任务。不要仅用简短测试提示词判断适配成功。

## 5. 调大网页搜索调用上限

DeepSeek 可能根据多个大纲方向在一次响应中并行发起多次搜索。若 `max_uses=1`，后续搜索会返回 `max_uses_exceeded`。

不要保留 Claude 示例中的低上限。先把 `max_uses` 从 `1` 调到至少 `10`；编排器—工作器这类复杂研究任务可直接调到 `20`，再根据实际 `server_tool_use` 数量验证：

```python
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 20,
}
```

提高 `max_uses` 后，网关仍可能返回 `max_uses_exceeded`。收集来源时区分正常结果和错误结果：

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
2. 确认 `tool_choice` 形参和上层调用仍保留，但思考模式的最终请求中没有该字段。
3. 单独验证大纲阶段能跳过 `ThinkingBlock` 并取得文本。
4. 检查默认值和所有显式调用点的 `max_tokens`，再用真实长输入和最高温度确认不会在思考阶段耗尽预算。
5. 确认网页搜索 `max_uses` 已调大，并用真实写作提示词检查 `server_tool_use` 数量和 `error_code`。
6. 完整运行提示词链，确认各阶段都取得非空文本。
