<!-- ---
title: "多模态智能体"
description: "在处理文本的同时处理图像、生成视觉内容并处理音频"
icon: "image"
--- -->

# 多模态智能体

突破纯文本智能体的限制。本教程介绍三项核心多模态能力——图像理解（视觉）、图像生成和音频（文本转语音 + 语音转文本），每项能力均使用最适合的服务商。

## 🎯 学习目标

- 使用 URL 和 Base64 数据源将图像发送给 Claude 进行视觉分析
- 在一次请求中比较多张图像
- 检测 MIME 类型并为视觉 API 编码本地文件
- 使用 Gemini 的原生图像生成功能根据文本提示词生成图像
- 使用 `response_modalities` 请求文本与图像的混合输出
- 使用 OpenAI 提供的 6 种声音将文本转换为语音
- 使用 Whisper 转录音频文件
- 验证往返保真度：文本 → 语音 → 转录 → 比较

## 📦 示例一览

| 服务商 | 文件 | 说明 |
| --- | --- | --- |
| ![Anthropic](../../common/badges/anthropic.svg) | [01_vision_anthropic.py](01_vision_anthropic.py) | 使用 Claude 视觉能力分析图像 |
| ![Gemini](../../common/badges/gemini.svg) | [02_image_generation_gemini.py](02_image_generation_gemini.py) | 使用 Gemini 原生生成图像 |
| ![OpenAI](../../common/badges/openai.svg) | [03_audio_openai.py](03_audio_openai.py) | 文本转语音与语音转文本 |

## 🚀 快速开始

> **前置条件：** Python 3.11+、API 密钥和 uv。完整设置说明请参阅 [SETUP.md](../../SETUP.md)。

```bash
# 视觉——使用 Claude 分析图像
uv run --directory 03-advanced-techniques/07-multimodal python 01_vision_anthropic.py

# 图像生成——使用 Gemini 创建图像
uv run --directory 03-advanced-techniques/07-multimodal python 02_image_generation_gemini.py

# 音频——使用 OpenAI 进行文本转语音和转录
uv run --directory 03-advanced-techniques/07-multimodal python 03_audio_openai.py
```

也可以使用 [Code Runner](https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner) VS Code 扩展，一键运行当前打开的脚本。

## 🔑 核心概念

### 1. 图像输入格式

Claude 接受两种图像格式：URL 引用或 Base64 编码：

```python
# URL 数据源——Claude 直接获取图像
{"type": "image", "source": {"type": "url", "url": "https://example.com/photo.jpg"}}

# Base64 数据源——将图像数据内联发送
{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "<base64>"}}
```

支持 JPEG、PNG、GIF 和 WebP 格式。图像会缩放到不超过 1568×1568 像素。

### 2. 视觉 API 模式

图像作为内容块与文本一起放入消息数组：

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=2048,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "url", "url": image_url}},
            {"type": "text", "text": "描述这张图像。"},
        ],
    }],
)
```

比较多张图像时，在图像块之间交错插入文字标签：

```python
content = [
    {"type": "text", "text": "图像 1："},
    {"type": "image", "source": {"type": "url", "url": url_1}},
    {"type": "text", "text": "图像 2："},
    {"type": "image", "source": {"type": "url", "url": url_2}},
    {"type": "text", "text": "比较这两张图像。"},
]
```

### 3. 图像生成

Gemini 很独特：同一个模型既能理解图像，也能创建图像。设置 `response_modalities` 即可请求图像输出：

```python
from google import genai
from google.genai import types

client = genai.Client()  # 自动读取环境变量 GOOGLE_API_KEY

response = client.models.generate_content(
    model="gemini-2.0-flash-exp-image-generation",
    contents="日出时分的山间景色",
    config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
)

# 响应部分包含文本和/或 inline_data（图像字节）
for part in response.candidates[0].content.parts:
    if part.inline_data:
        Path("output.png").write_bytes(part.inline_data.data)
```

### 4. 音频流水线

```mermaid
---
config:
  look: handDrawn
  theme: neutral
---
flowchart LR
    A["📝 文本输入   "] -->|"TTS"| B["🔊 音频文件   "]
    B -->|"STT"| C["📝 转录文本 "]
    C -->|"比较"| D["✅ 验证结果  "]
```

OpenAI 为两个方向提供独立端点：

```python
# 文本 → 语音（TTS）
response = client.audio.speech.create(model="tts-1", voice="alloy", input="你好！")
response.write_to_file("output.mp3")

# 语音 → 文本（STT / Whisper）
with open("output.mp3", "rb") as f:
    result = client.audio.transcriptions.create(model="whisper-1", file=f)
print(result.text)
```

### 5. 多模态令牌成本

| 内容类型 | 约略成本 |
| --- | --- |
| 1568×1568 图像（最大尺寸） | 约 1,600 个令牌 |
| 768×768 图像 | 约 400 个令牌 |
| 文本（1 个单词） | 约 1.3 个令牌 |
| 音频 TTS（1,000 个字符） | $0.015 |
| 音频 STT（1 分钟） | $0.006 |

### 6. 服务商选择指南

| 任务 | 最佳服务商 | 原因 |
| --- | --- | --- |
| 图像分析 / OCR | **Anthropic**（Claude） | 一流的视觉能力和简洁的内容块 API |
| 图像生成 | **Gemini** | 原生生成——无需独立端点 |
| 文本转语音 | **OpenAI** | 6 种不同声音，TTS API 成熟 |
| 语音转文本 | **OpenAI**（Whisper） | 行业标准级转录准确度 |
| 视频理解 | **Gemini** | 原生支持视频输入（最长 1 小时） |

## 🏗️ 代码结构

每个脚本都遵循标准模式：使用类封装 LLM/API 逻辑，并通过交互式菜单驱动 `main()`。

| 脚本 | 类 | 主要方法 |
| --- | --- | --- |
| `01_vision_anthropic.py` | `VisionAnalyst` | `analyze_url()`、`analyze_file()`、`compare_images()` |
| `02_image_generation_gemini.py` | `ImageGenerator` | `generate()`、`save_image()` |
| `03_audio_openai.py` | `VoiceAssistant` | `speak()`、`transcribe()`、`round_trip()`、`voice_comparison()` |

## ⚠️ 注意事项

- **图像令牌成本**——每张图像根据分辨率消耗 400–1,600 个令牌，多图像请求会迅速叠加。
- **音频文件大小**——TTS 每句话生成约 32KB 的 MP3 文件；声音比较会一次创建 6 个文件，输出保存在 `output/`。
- **生成质量**——Gemini 图像生成功能仍处于实验阶段，结果可能有所差异，模型偶尔也会拒绝请求。
- **速率限制**——图像和音频 API 的速率限制比文本更严格。在生产环境中请在请求之间增加延迟。
- **音频不跟踪令牌**——OpenAI 音频 API 按字符数（TTS）和分钟数（STT）计费，而非按令牌计费；音频脚本改为跟踪 API 调用次数。

## 👉 后续实践

- **[安全护栏](../08-guardrails/)**——为生产级智能体添加输入和输出安全层
- **可以尝试的实验：**
  - 添加图像编辑：向 Gemini 发送图像和文本提示词进行修改
  - 构建视觉问答循环：上传图像并围绕图像提出后续问题
  - 串联多种模态：分析图像 → 生成描述 → 将描述朗读出来
  - 为 Whisper 转录添加语言检测，以支持多语言
