<!-- ---
title: "评测框架"
description: "集成外部评测框架：Promptfoo、Braintrust AutoEvals、Langfuse"
icon: "puzzle-piece"
--- -->

# 评测框架

探索 Anthropic 在[揭秘 AI 智能体评测](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)中推荐的**生产级评测框架**。每个脚本都展示一种评测同一研究助手智能体的方法，涵盖 YAML 驱动配置、预置评分器和链路追踪平台。

## 🎯 你将学到什么

- 使用 **Promptfoo 的 YAML 配置**以声明式方式定义评测套件
- 使用 **Braintrust AutoEvals** 的预置评分器（字符串相似度、事实性和自定义分类器）
- 使用 **Langfuse** 为智能体添加链路追踪和程序化评分
- 比较不同框架的权衡：CLI 与 SDK、本地与云端、字符串评分与基于 LLM 的评分

## 📦 可用示例

| 提供方 | 文件 | 说明 |
| ------ | ---- | ---- |
| Promptfoo | [01_promptfoo.py](01_promptfoo.py) | YAML 配置、自定义 Python 提供器和断言 |
| Braintrust | [02_braintrust_autoevals.py](02_braintrust_autoevals.py) | 预置评分器：Levenshtein、Factuality 和自定义分类器 |
| Langfuse | [03_langfuse.py](03_langfuse.py) | 基于装饰器的链路追踪和多类型评分 |

## 🚀 快速开始

> **前置条件：** Python 3.11+ 和 uv。完整配置说明请参阅 [SETUP.md](../../SETUP.md)。

每个脚本都可以在无外部依赖的**模拟模式**下运行。若要使用真实框架：

```bash
# 核心依赖（无需安装下列可选依赖即可运行所有脚本）
uv run --directory 04-testing-evaluation/06-eval-frameworks python 01_promptfoo.py

# 安装框架依赖
uv sync --extra promptfoo    # 添加 pyyaml，用于生成 YAML
uv sync --extra braintrust   # 添加 autoevals 评分器
uv sync --extra langfuse     # 添加 langfuse SDK
uv sync --extra all          # 安装所有框架


# 个人总结
export PROMPTFOO_PYTHON="$(cygpath -m "$PWD/04-testing-evaluation/06-eval-frameworks/.venv/Scripts/python.exe")"
promptfoo eval -c 04-testing-evaluation/06-eval-frameworks/promptfooconfig.yaml
```




也可以使用 VS Code 的 [Code Runner](https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner) 扩展，一键运行当前打开的脚本。

## 🔑 核心概念

### 框架对比

| 维度 | Promptfoo | Braintrust AutoEvals | Langfuse |
| ---- | --------- | -------------------- | -------- |
| **类型** | CLI 工具（Node.js） | Python SDK | Python SDK |
| **配置方式** | YAML 驱动 | 代码驱动 | 代码驱动 |
| **可本地运行？** | 是（完全本地） | 字符串评分器：是 | 需要服务器（或自行托管） |
| **API 密钥** | 仅 LLM 提供方需要 | LLM 评分器需要 `OPENAI_API_KEY` | `LANGFUSE_*` 密钥 |
| **最适合** | 声明式评测套件 | 预置评分 | 链路追踪与评分 |

### 1. Promptfoo：声明式 YAML 评测

在 YAML 中定义提供器、提示词、测试用例和断言：

```yaml
providers:
  - id: "file://provider_agent.py"    # 自定义 Python 提供器
tests:
  - vars:
      question: "微服务有哪些优势？"
    assert:
      - type: python                   # 自定义 Python 断言
        value: "file://assertion_keywords.py"
      - type: contains                 # 内置字符串检查
        value: "doc_001"
      - type: llm-rubric               # LLM 充当裁判
        value: "回答应涵盖可扩展性和故障隔离。"
```

### 2. Braintrust AutoEvals：预置评分器

无需从头构建，直接使用经过实践检验的评分器：

```python
from autoevals import Factuality, Levenshtein

# 本地评分器——无需 API 密钥
lev = Levenshtein()
result = lev.eval(output="你好世届", expected="你好世界")

# LLM 评分器——需要 OPENAI_API_KEY
fact = Factuality()
result = fact.eval(input="问题", output="回答", expected="参考答案")
```

### 3. Langfuse：链路追踪与评分

使用装饰器检测代码，并以编程方式添加评分：

```python
from langfuse import observe, get_client

@observe()  # 自动创建包含嵌套跨度的追踪
def my_agent(question: str) -> str:
    return search_and_answer(question)

# 为追踪评分
langfuse = get_client()
langfuse.create_score(
    trace_id=trace_id,
    name="correctness",
    value=0.95,
    data_type="NUMERIC",
)
```

## ⚠️ 重要注意事项

- **选定一个框架并持续迭代**——原文建议把精力投入高质量测试用例和评分器，而不是纠结框架选择
- **许多团队会组合使用工具**——用 Promptfoo 完成 CI/CD 断言，用 Langfuse 追踪生产环境，用 AutoEvals 快速评分
- **所有脚本都可在无外部依赖时运行**——模拟模式用于演示模式；准备实际使用时再安装对应框架
- **基于 LLM 的评分器会产生费用**——Factuality/ClosedQA 评分器会调用 OpenAI；大规模评测时应规划好预算

## 🔗 资源

- [Promptfoo Python 提供器文档](https://www.promptfoo.dev/docs/providers/python/)
- [Braintrust AutoEvals GitHub 仓库](https://github.com/braintrustdata/autoevals)
- [Langfuse Python SDK](https://langfuse.com/docs/sdk/python/decorators)
- [揭秘评测——评测框架附录](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

## 👉 后续步骤

- **应用**——选择适合工作流的框架，为自己的智能体定义评测任务
- **组合**——使用 Promptfoo 完成 CI 断言，并使用 Langfuse 追踪生产环境
- **综合项目**——参阅[评测工具链](../07-eval-harness/)，了解融合所有技术且不依赖特定框架的评测流水线
