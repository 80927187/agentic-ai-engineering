<!-- ---
title: "评估"
description: "构建用于持续衡量准确性、质量和回归情况的评估套件"
icon: "bar-chart"
--- -->

# 评估

超越确定性断言，对智能体质量进行**统计评估**。本教程遵循 Anthropic 的评估驱动开发方法：将成功标准定义为评估任务，使用多种评分器打分，持续跟踪质量，并自动发现回归。

## 🎯 你将学到什么

- 构建基于代码的评分器：关键词匹配、正则表达式、来源引用和工具调用验证
- 使用结构化评分标准和思维链评判实现 **LLM 充当评委**模式
- 设计**黄金数据集**——用于回归测试的精选输入/输出对
- 构建使用多评分器打分的端到端评估流水线
- 通过比较当前通过率与基线来检测回归
- 理解 Anthropic 的评估术语：任务、试验、轨迹、结果和评分器

## 📦 示例

| 脚本 | 文件 | 说明 |
| ---- | ---- | ---- |
| 基于代码的评分器 | [01_code_based_graders.py](01_code_based_graders.py) | 关键词、正则表达式、引用和工具调用评分器 |
| LLM 充当评委 | [02_llm_as_judge.py](02_llm_as_judge.py) | 使用思维链和结构化评分标准打分 |
| 评估流水线 | [03_eval_pipeline.py](03_eval_pipeline.py) | 端到端流程：数据集 → 试验 → 评分 → 回归检测 |

## 🚀 快速开始

> **前置条件：** Python 3.11+、API 密钥和 uv。完整配置说明请参阅 [SETUP.md](../../SETUP.md)。

```bash
uv run --directory 04-testing-evaluation/02-evals python 01_code_based_graders.py

# 示例
uv run --directory 04-testing-evaluation/02-evals python 03_eval_pipeline.py
```

所有脚本均可在没有 API 密钥时以**模拟模式**运行（使用预定义响应），也可在配置 `ANTHROPIC_API_KEY` 后以**在线模式**运行。

你也可以安装 VS Code 的 [Code Runner](https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner) 扩展，单击即可运行当前打开的脚本。

## 🔑 核心概念

### 1. 评估术语（来自 Anthropic）

| 术语 | 定义 |
| ---- | ---- |
| **任务（Task）** | 包含输入和成功标准的测试用例 |
| **试验（Trial）** | 对一个任务的一次随机运行（通过多次运行捕捉差异） |
| **轨迹（Transcript）** | 智能体所有操作的完整记录 |
| **结果（Outcome）** | 智能体完成后的最终环境状态 |
| **评分器（Grader）** | 对智能体表现的某一方面进行评分的逻辑 |
| **pass@k** | k 次试验中至少成功一次的概率 |
| **pass^k** | k 次试验必须全部成功（用于检验一致性） |

### 2. 三种评分器

```python
# 基于代码：快速、确定、成本低
class KeywordGrader:
    def grade(self, answer, expected_keywords) -> GraderResult: ...

# 基于模型：灵活、细致、成本高
class LLMJudge:
    def evaluate(self, question, answer, reference) -> JudgeResult: ...

# 人工：黄金标准，成本很高，难以规模化
# （这里只提及而未实现——可用于校准自动评分器）
```

### 3. 通过结构化输出让 LLM 充当评委

使用 `tool_choice` 强制模型输出结构化评分：

```python
JUDGE_TOOLS = [{
    "name": "submit_evaluation",
    "input_schema": {
        "properties": {
            "reasoning": {"type": "string"},        # 先推理
            "accuracy_score": {"type": "integer"}, # 再评分
            "completeness_score": {"type": "integer"},
            "grounding_score": {"type": "integer"},
        }
    }
}]
# 使用 tool_choice={"type": "tool", "name": "submit_evaluation"}
```

### 4. 黄金数据集设计

从 15～20 个精选任务开始（Anthropic 建议初期准备 20～50 个）：

```json
{
    "id": "task_001",
    "question": "微服务有哪些主要优势？",
    "expected_keywords": ["可扩展性", "故障隔离"],
    "expected_source_ids": ["doc_001"],
    "difficulty": "简单",
    "category": "架构"
}
```

任务应兼顾：简单的单文档任务、困难的跨文档综合任务，以及应当拒答的超出范围问题。

## ⚠️ 重要事项

- **从小规模开始**——15～20 个精心挑选的任务胜过 1000 个泛化任务
- **校准评分器**——将自动评分结果与人工判断进行比较
- **LLM 充当评委并非零成本**——每次评估都会消耗 token；应优先使用基于代码的评分器
- **持续跟踪通过率**——下降 5% 就表示出现了值得调查的回归

## 🔗 资源

- [揭开 AI 智能体评估的神秘面纱 — Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — 本教程使用的核心评估术语（任务、试验、评分器）、评分器分类和八步评估路线图
- [使用 MT-Bench 和 Chatbot Arena 评判 LLM 评委 — Zheng 等，2023](https://arxiv.org/abs/2306.05685) — 系统研究 LLM 评委与人工的一致率、位置偏差和结构化评分标准方法
- [语言模型整体评估（HELM）— Liang 等，2022](https://arxiv.org/abs/2211.09110) — 涵盖准确性、校准、稳健性、公平性和效率的多指标评估框架
- [OpenAI 评估最佳实践](https://platform.openai.com/docs/guides/evaluation-best-practices) — 关于评估设计、黄金数据集和评分策略的实用指南
- [评估驱动开发](https://evaldriven.org/) — 在开发功能前先构建评估的工程方法

## 👉 后续步骤

掌握评估后，可以继续：

- **[追踪与调试](../03-tracing-debugging/)**——评估失败时，追踪信息会准确显示失败原因
- **动手实验**——将你自己的智能体失败案例加入黄金数据集
- **深入探索**——尝试不同的评分标准设计，并比较评委的一致性
