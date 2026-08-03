<!-- ---
title: "流式传输与实时输出"
description: "逐 Token 流式响应，以及在流式传输过程中处理工具调用"
icon: "zap"
--- -->

# 流式传输与实时输出

通过逐 Token 的实时响应，让智能体显得更加生动。此前的所有教程都使用阻塞式 API 调用——用户只能在沉默中等待完整响应返回。本教程将加入流式传输，把体验从“是不是卡住了？”变为“它正在思考，而且我能看到过程”。

真正的挑战并非基础流式传输，而是如何在流式传输中处理工具调用。当 Claude 决定在响应途中调用工具时，你需要检测该调用、执行工具、将结果反馈给模型，然后恢复流式传输。本教程将以易于理解的方式完成这一过程。

## 🎯 你将学到什么

- 使用 `client.messages.stream()` 逐 Token 流式输出 Claude 的响应
- 使用 Rich 的 `Live` 显示组件在终端中渲染流式 Markdown
- 理解完整的流式事件生命周期（message_start → content_block_delta → message_stop）
- 在流式传输途中处理 tool_use 块——检测、执行并恢复传输
- 构建支持工具调用的完整流式智能体循环
- 跟踪流式传输的 Token 用量（用量信息会在流结束时到达）

## 📦 可用示例

| 提供商                                          | 文件                                                         | 说明                       |
| ----------------------------------------------- | ------------------------------------------------------------ | -------------------------- |
| ![Anthropic](../../common/badges/anthropic.svg) | [01_streaming_fundamentals.py](01_streaming_fundamentals.py) | 文本流式传输与多轮对话     |
| ![Anthropic](../../common/badges/anthropic.svg) | [02_streaming_agent.py](02_streaming_agent.py)               | 支持工具调用的流式智能体   |

## 🚀 快速开始

> **前置条件：** Python 3.11+、API 密钥和 uv。完整配置说明请参阅 [SETUP.md](../../SETUP.md)。

```bash
uv run --directory 03-advanced-techniques/02-streaming python {script_name}

# 先从基础示例开始
uv run --directory 03-advanced-techniques/02-streaming python 01_streaming_fundamentals.py

# 然后尝试流式智能体
uv run --directory 03-advanced-techniques/02-streaming python 02_streaming_agent.py
```

你也可以使用 VS Code 的 [Code Runner](https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner) 扩展，单击一次即可运行当前打开的脚本。

## 🔑 核心概念

### 1. 两种流式传输方式

Anthropic 提供两种流式传输方式。可以先使用简单方式，需要更多控制时再改用事件方式。

**简单方式——`.text_stream` 迭代器：**

```python
with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=2048,
    messages=messages,
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)  # 每个分片包含几个字符
```

这是最简单的流式传输方式。迭代器会产出纯文本字符串，也就是内容增量。它非常适合不需要事件级控制的简单用例。

**事件方式——完整的生命周期控制：**

```python
with client.messages.stream(...) as stream:
    for event in stream:
        if event.type == "content_block_start":
            # 一个新的内容块（text 或 tool_use）即将开始
            pass
        elif event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                print(event.delta.text, end="")
            elif event.delta.type == "input_json_delta":
                # 工具输入参数正在流式传入
                pass
        elif event.type == "content_block_stop":
            # 内容块已结束
            pass
        elif event.type == "message_delta":
            # 此时已经可以获取 stop_reason
            print(f"\n停止原因：{event.delta.stop_reason}")
```

当你需要检测工具调用、跟踪内容块边界或构建自定义渲染逻辑时，请使用基于事件的迭代方式。

### 2. 流式事件生命周期

每个流都遵循以下顺序：

```
message_start                          ← 流开始
│
├─ content_block_start (index=0)       ← 第一个块（通常是文本）
│  ├─ content_block_delta              ← 文本分片到达
│  ├─ content_block_delta              ← 更多文本到达
│  └─ content_block_stop               ← 内容块完成
│
├─ content_block_start (index=1)       ← 可能是另一个 text 或 tool_use 块
│  ├─ content_block_delta              ← text 或 input_json 增量
│  └─ content_block_stop
│
├─ message_delta                       ← stop_reason 和最终用量统计
└─ message_stop                        ← 流结束
```

关键点在于：单个响应可以包含**多个内容块**，文本块与 tool_use 块可以交错出现。这正是带工具的流式传输有趣之处。

### 3. 在流式传输中调用工具

当 Claude 想要调用工具时，流中会包含一个 `tool_use` 内容块。处理流程如下：

```
用户：“东京的天气怎么样？”
        │
        ▼
  ┌─ 流开始 ────────────────────────────────────┐
  │ 文本块：“让我查一下天气……”                    │  ← 流式输出到终端
  │ tool_use 块：get_weather(city="Tokyo")       │  ← 在流传输途中被检测到
  │ stop_reason: "tool_use"                      │
  └──────────────────────────────────────────────┘
        │
        ▼ 执行工具
  ┌─ 工具结果 ─────────────────────────┐
  │ {"city": "Tokyo", "temp_f": 58} │
  └────────────────────────────────────┘
        │
        ▼ 反馈结果并启动新的流
  ┌─ 恢复流式传输 ───────────────────────────────┐
  │ 文本块：“东京今天是 58°F，天气晴朗。”          │  ← 流式输出到终端
  │ stop_reason: "end_turn"                      │
  └──────────────────────────────────────────────┘
```

智能体循环会在每次流结束后检查 `stop_reason`：

- `"end_turn"` → 已完成，返回响应
- `"tool_use"` → 执行工具，反馈结果，然后再次开始流式传输
- `"max_tokens"` → 响应已被截断

### 4. 使用 Rich Live 渲染

直接使用 `print()` 可以获得流式文本，但无法在传输过程中处理 Markdown 格式。Rich 的 `Live` 显示组件可以解决这个问题——每次更新时，它都会重新渲染已累积的全部 Markdown：

```python
from rich.live import Live
from rich.markdown import Markdown

accumulated = ""
with Live(Markdown(""), refresh_per_second=15, console=console) as live:
    for text in stream.text_stream:
        accumulated += text
        live.update(Markdown(accumulated))
```

`refresh_per_second=15` 参数会限制更新频率，使渲染保持流畅。用户可以看到格式化后的 Markdown 实时生成——标题、项目符号和粗体文本都能在流式传输过程中正确渲染。

### 5. 在流式传输中跟踪 Token

在流完成之前，Token 用量不可用。请使用 `get_final_message()` 获取用量：

```python
with client.messages.stream(...) as stream:
    for text in stream.text_stream:
        print(text, end="")

    # 流完成后即可获取用量
    final_message = stream.get_final_message()
    token_tracker.track(final_message.usage)
    print(f"\nToken：输入 {final_message.usage.input_tokens}，输出 {final_message.usage.output_tokens}")
```

`get_final_message()` 返回完整累积的 `Message` 对象——它与 `client.messages.create()` 返回的对象相同，只不过你已经先以流式方式接收了它。

## 🏗️ 代码结构

### 脚本 01——流式传输基础

```python
class StreamingChat:
    """提供流式响应的交互式聊天。"""

    def stream_simple(self, user_input, console) -> str:
        """使用 .text_stream 进行流式传输——简单方式。"""
        with client.messages.stream(...) as stream:
            for text in stream.text_stream:   # 只有文本字符串
                # 使用 Rich Live 渲染
            final = stream.get_final_message()
            # 跟踪 Token

    def stream_with_events(self, user_input, console) -> str:
        """使用基于事件的迭代进行流式传输——完全控制。"""
        with client.messages.stream(...) as stream:
            for event in stream:              # 带类型的事件对象
                if event.type == "content_block_delta":
                    # 处理 text_delta 和 input_json_delta
```

### 脚本 02——流式智能体

```python
class StreamingAgent:
    """支持工具调用处理的流式智能体。"""

    def run(self, user_input, console) -> str:
        """智能体循环：流式传输 → 检测工具 → 执行 → 恢复传输。"""
        while True:
            response = self._stream_response(console)
            if response.stop_reason == "tool_use":
                results = self._execute_tool_calls(response.content, console)
                # 反馈结果，然后再次循环
            else:
                return extract_text(response)   # 完成

    def _stream_response(self, console) -> Message:
        """执行一次流式 API 调用，并渲染文本和工具指示信息。"""
        with client.messages.stream(tools=TOOLS, ...) as stream:
            self._render_mixed_stream(stream, console)
            return stream.get_final_message()

    def _render_mixed_stream(self, stream, console) -> None:
        """关键方法：处理交错的文本块和 tool_use 块。"""
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "text":
                    # 启动 Rich Live 显示
                elif event.content_block.type == "tool_use":
                    # 显示“正在调用 tool_name……”
            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    # 更新实时 Markdown 显示
```

## ⚠️ 重要注意事项

- **流式传输不会减少总延迟**——Token 数量与处理时间都不会改变。它通过立即显示进度来降低用户的*感知延迟*。
- **错误处理**——流可能在途中失败。务必使用 try/except，并处理 `APIError`。必须在 `finally` 块中停止 `Live` 显示，以免终端显示异常。
- **`stop_reason` 至关重要**——务必检查它。`"tool_use"` 表示执行工具并继续，`"end_turn"` 表示完成，`"max_tokens"` 表示响应被截断。
- **Token 跟踪时机**——只有流完成后，才能通过 `get_final_message()` 获取用量统计。无法在流传输途中跟踪 Token。
- **对话历史**——流式传输结束后，需要保存完整的响应内容作为消息历史。使用 `get_final_message().content` 获取完整的内容块列表。

## 👉 后续步骤

掌握流式传输后，可以继续：

- **[上下文工程](../03-context-engineering/)**——通过滑动窗口和摘要管理有限的上下文窗口
- **动手实验**——为流式智能体添加更多工具，并尝试在一次响应中触发多个工具调用的提示词
- **深入探索**——尝试在 `stream.text_stream` 和事件迭代之间切换，体会两者在控制能力和简洁性上的差异
