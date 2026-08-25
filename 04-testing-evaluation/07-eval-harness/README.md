<!-- ---
title: "评测工具"
description: "综合项目：融合所有测试技术的完整评测流水线"
icon: "award"
--- -->

# 评测工具

这是本模块的综合项目，将**全部五项技术**整合为一套可复用的评测工具。单元测试模式、评测、追踪、红队测试和基准测试共同构成一个面向真实智能体的完整质量体系。

## 🎯 你将学到什么

- 将全部 5 项测试技术串联成统一的评测流水线
- 使用 Pydantic 数据模型构建类型安全的评测基础设施
- 构建职责清晰、模块化的 `eval_harness` 包
- 生成汇总质量、安全性和基准指标的综合报告
- 端到端实践**评测驱动开发**

## 📦 可用示例

| 脚本 | 文件 | 说明 |
| ---- | ---- | ---- |
| 评测工具 | [01_eval_harness.py](01_eval_harness.py) | 运行完整评测流水线 |

### 包模块

| 模块 | 文件 | 说明 |
| ---- | ---- | ---- |
| 模型 | [eval_harness/models.py](eval_harness/models.py) | Pydantic 模型：EvalTask、EvalTrial、EvalResult 等 |
| 智能体 | [eval_harness/agent.py](eval_harness/agent.py) | 研究助手（实时模式和模拟模式） |
| 评分器 | [eval_harness/graders.py](eval_harness/graders.py) | 关键词、引用和复合评分器 |
| 追踪器 | [eval_harness/tracer.py](eval_harness/tracer.py) | 基于 span 的轻量级追踪收集器 |
| 红队测试 | [eval_harness/red_team.py](eval_harness/red_team.py) | 使用对抗性输入进行安全测试 |
| 基准测试 | [eval_harness/benchmark.py](eval_harness/benchmark.py) | 使用帕累托分析比较模型 |
| 报告器 | [eval_harness/reporter.py](eval_harness/reporter.py) | 生成 Rich 终端报告 |

## 🚀 快速开始

> **前置条件：** Python 3.11+、API 密钥和 uv。完整的设置说明请参阅 [SETUP.md](../../SETUP.md)。

```bash
uv run --directory 04-testing-evaluation/07-eval-harness python 01_eval_harness.py
```

启动后，可通过交互式菜单选择：

- **模拟模式**（默认）——使用预定义回答，不调用 API，立即得到结果
- **实时模式**——通过工具调用智能体循环发起真实的模型 API 请求（需要 `ANTHROPIC_API_KEY`）。实时模式可选择 `deepseek-v4-flash`、`glm-5.2` 或 `glm-4.7`。评测试验和安全测试会调用 API；基准测试需要比较多种模型配置，因此仍使用模拟数据。

也可以使用 VS Code 的 [Code Runner](https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner) 扩展，单击即可运行当前打开的脚本。

## 🏗️ 架构

```
加载任务 → 运行智能体（同时追踪）→ 评分（多评分器）→ 安全测试 → 基准测试 → 报告
```

每个阶段都对应前面的一个教程：

| 流水线阶段 | 模块 | 对应教程 |
| ---------- | ---- | -------- |
| 可测试的智能体设计 | `agent.py` | [01 - 智能体单元测试](../01-unit-testing-agents/) |
| 黄金数据集与评分 | `graders.py` | [02 - 评测](../02-evals/) |
| 执行追踪 | `tracer.py` | [03 - 追踪](../03-tracing-debugging/) |
| 对抗性测试 | `red_team.py` | [04 - 红队测试](../04-red-teaming-safety/) |
| 模型比较 | `benchmark.py` | [05 - 基准测试](../05-benchmarking/) |
| 统一报告 | `reporter.py` | 综合项目新增内容 |

## 🔑 核心概念

### 1. Pydantic 数据模型

带验证的类型安全评测基础设施：

```python
class EvalTask(BaseModel):
    id: str
    question: str
    expected_keywords: list[str]
    difficulty: str = "medium"

class EvalResult(BaseModel):
    task_id: str
    trials: list[EvalTrial]
    grader_scores: list[GraderScore]
    pass_rate: float
```

### 2. 复合评分

每项任务使用多种评分器并进行加权评分：

```python
class CompositeGrader:
    """按可配置权重组合关键词评分器和引用评分器。"""

    def grade(self, trial, task) -> list[GraderScore]:
        keyword_score = self.keyword_grader.grade(trial.answer, task.expected_keywords)
        citation_score = self.citation_grader.grade(trial.answer, task.expected_source_ids)
        # 通过加权组合决定整体通过或失败
```

### 3. 评测报告

该工具会生成统一报告：

```
╭─────────────── 评测报告：研究助手 ───────────────╮
│                                                 │
│  📊 质量评测          12/15 项任务通过（80.0%） │
│  🔒 安全分数          阻止 7/8 次攻击（87.5%）  │
│  ⏱️  平均延迟          每项任务 1.5 秒           │
│  💰 总成本            $0.045                    │
│                                                 │
╰─────────────────────────────────────────────────╯
```

## ⚠️ 重要注意事项

- **评测是持续演进的基础设施**——像维护生产代码一样维护黄金数据集
- **安全性是一等指标**——红队测试结果应与准确性结果并列展示
- **成本追踪必不可少**——在 CI/CD 中运行评测套件前，先了解其成本
- **回归警报需要基线**——保存一次已知良好运行的结果作为比较基准

## 🔗 资源

- [解密 AI 智能体评测 — Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) —— 本工具实现的评测方法：任务 → 试验 → 评分 → 回归检测
- [EleutherAI 语言模型评测工具](https://github.com/EleutherAI/lm-evaluation-harness) —— 面向大语言模型的开源评测框架；模块化工具架构的设计灵感来源
- [评测驱动开发](https://evaldriven.org/) —— 在构建功能之前，先通过评测定义成功标准的开发方法
- [构建高效智能体 — Anthropic](https://www.anthropic.com/research/building-effective-agents) —— 通过依赖注入和清晰接口提高智能体可测试性的设计模式

## 👉 后续步骤

这是综合项目——你已经完成“测试与评估”模块！接下来可以：

- **应用**——为自己的智能体构建评测工具
- **扩展**——参考[教程 02](../02-evals/)，为复合评分器添加“以大语言模型为裁判”的评分方式
- **集成**——在 CI/CD 中运行评测工具，自动发现回归问题
- **框架**——接入适用于生产环境的[评测框架](../06-eval-frameworks/)（Promptfoo、Braintrust、Langfuse）
- **探索**——查看[模块 02：高效智能体](../../02-effective-agents/)，了解可供测试的更复杂智能体
