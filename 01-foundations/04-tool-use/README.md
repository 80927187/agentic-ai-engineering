<!-- ---
title: "工具使用"
description: "让 LLM 能够调用函数并与外部系统交互"
icon: "wrench"
--- -->

# 工具使用

学习如何赋予 LLM 调用函数（工具）的能力，使其能够与现实世界交互。本教程演示如何定义工具、处理工具调用，以及代表模型执行函数。

## 🎯 你将学到什么

- 使用 JSON Schema 定义供 LLM 使用的工具
- 处理工具调用循环（请求 -> 执行 -> 回复）
- 借助安全防护机制执行函数
- 处理单次回复中的多个工具调用

## 📦 可用示例

| 提供商                                          | 文件                                                 | 说明                                 |
| ----------------------------------------------- | ---------------------------------------------------- | ------------------------------------ |
| ![Anthropic](../../common/badges/anthropic.svg) | [01_tool_use_anthropic.py](01_tool_use_anthropic.py) | 使用 Claude Messages API 调用工具    |
| ![OpenAI](../../common/badges/openai.svg)       | [02_tool_use_openai.py](02_tool_use_openai.py)       | 使用 OpenAI Responses API 调用工具   |

## 🚀 快速开始

> **前置条件：** Python 3.11+、API 密钥和 uv。完整的设置说明请参阅 [SETUP.md](../../SETUP.md)。

```bash
uv run --directory 01-foundations/04-tool-use python {script_name}

# 示例
uv run --directory 01-foundations/04-tool-use python 01_tool_use_anthropic.py
```

你也可以使用 VS Code 的 [Code Runner](https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner) 扩展，一键运行当前打开的脚本。

## 🔑 核心概念

### 1. 工具定义

工具使用 JSON Schema 定义，以便 LLM 了解有哪些函数可用：

**Anthropic：**

```python
TOOLS = [
    {
        "name": "calculator",
        "description": "执行基本算术运算。",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        },
    },
]
```

**OpenAI：**

```python
TOOLS = [
    {
        "type": "function",
        "name": "calculator",
        "description": "执行基本算术运算。",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        },
    },
]
```

### 2. 工具调用循环

LLM 不会直接执行工具，而是发出工具调用请求，由你负责执行：

```text
用户消息
    |
LLM 回复（包含 tool_use）
    |
执行工具 -> 获取结果
    |
将结果发回 LLM
    |
LLM 回复（最终答案）
```

**Anthropic：**

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    tools=TOOLS,
    messages=messages,
)

if response.stop_reason == "tool_use":
    for block in response.content:
        if isinstance(block, ToolUseBlock):
            result = execute_tool(block.name, block.input)
            # 使用 tool_use_id 将结果发回
```

**OpenAI：**

```python
response = client.responses.create(
    model="gpt-4.1",
    tools=TOOLS,
    input=messages,
)

for output in response.output:
    if output.type == "function_call":
        result = execute_tool(output.name, json.loads(output.arguments))
        # 使用 call_id 将结果发回
```

### 3. 带安全防护的工具实现

务必验证并清理工具输入，尤其是系统级工具的输入：

```python
BLOCKED_COMMANDS = ["rm", "sudo", "chmod", "shutdown", ">", ">>"]

def run_bash(command: str, timeout: int = 30) -> dict:
    """在安全防护机制下执行 Bash 命令。"""
    # 拦截危险命令
    for blocked in BLOCKED_COMMANDS:
        if blocked in command.lower():
            return {"error": f"命令已被拦截：包含 '{blocked}'"}

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        timeout=timeout,
    )
    return {"stdout": result.stdout, "stderr": result.stderr}
```

### 4. 处理多个工具调用

LLM 可以在单次回复中请求调用多个工具。继续处理之前，应先完成所有工具调用：

**Anthropic：**

```python
tool_results = []
for tool_use in tool_uses:
    result = execute_tool(tool_use.name, tool_use.input)
    tool_results.append({
        "type": "tool_result",
        "tool_use_id": tool_use.id,
        "content": json.dumps(result),
    })
messages.append({"role": "user", "content": tool_results})
```

**OpenAI：**

```python
# 先将函数调用添加到消息列表
messages.extend(response.output)

# 再添加结果
for func_call in function_calls:
    result = execute_tool(func_call.name, json.loads(func_call.arguments))
    messages.append({
        "type": "function_call_output",
        "call_id": func_call.call_id,
        "output": json.dumps(result),
    })
```

## 🧰 本教程中的工具

| 工具         | 说明                                 |
| ------------ | ------------------------------------ |
| `calculator` | 基本算术运算（加、减、乘、除）       |
| `read_file`  | 从文件系统读取文件内容               |
| `run_bash`   | 执行 Shell 命令（带安全防护）        |

## 🏗️ 代码结构

两个示例采用一致的结构：

```python
# 1. 使用 JSON Schema 定义工具
TOOLS = [...]

# 2. 实现工具函数
def calculator(operation: str, a: float, b: float) -> dict:
    ...

def read_file(path: str) -> dict:
    ...

def run_bash(command: str) -> dict:
    ...

# 3. 工具执行分派器
TOOL_FUNCTIONS = {"calculator": calculator, "read_file": read_file, ...}

def execute_tool(name: str, input: dict) -> Any:
    return TOOL_FUNCTIONS[name](**input)


# 4. 包含工具调用循环的聊天类
class ToolUseChat:
    def send_message(self, message: str) -> str:
        while True:
            response = self.client.create(tools=TOOLS, ...)

            if has_tool_calls(response):
                execute_tools_and_add_results()
                continue
            else:
                return response.text


# 5. 主编排逻辑
def main():
    chat = ToolUseChat(model, token_tracker, console)
    while True:
        user_input = input()
        response = chat.send_message(user_input)
        print(response)
```

## 👉 后续步骤

掌握工具使用后，可以继续：

- **[智能体循环](../05-agent-loop/README.md)**——构建能够使用工具完成任务的自主智能体
- **动手实验**——添加网络搜索、数据库查询或 API 调用等更多工具
- **深入探索**——实现工具选择模式（`auto`、`required`、`none`）
