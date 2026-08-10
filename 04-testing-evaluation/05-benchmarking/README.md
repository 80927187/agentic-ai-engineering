<!-- ---
title: "基准测试"
description: "对模型、提示词和架构进行系统化的正面对比"
icon: "bar-chart-2"
--- -->

# 基准测试

当你需要在 Claude Sonnet 与 GPT-4o-mini 之间，或在两种提示词策略之间做出选择时，基准测试能为你提供**数据，而不是凭感觉判断**。你可以围绕准确率、延迟、成本和可靠性等重要维度进行系统化的正面对比。

## 🎯 你将学到什么

- 比较不同模型的准确率、延迟、Token 用量和成本
- 评估零样本、少样本和思维链等提示词策略
- 构建配置矩阵（模型 × 提示词）并开展受控实验
- 找出**帕累托最优**配置（在给定成本预算下准确率最高的配置）
- 根据数据选择模型

## 📦 示例一览

| 示例 | 文件 | 说明 |
| ---- | ---- | ---- |
| 模型对比 | [01_model_comparison.py](01_model_comparison.py) | 使用相同任务比较 Claude Sonnet、Haiku 和 GPT-4.1 mini |
| 提示词对比 | [02_prompt_comparison.py](02_prompt_comparison.py) | 比较零样本、少样本与思维链策略 |
| 基准测试套件 | [03_benchmark_suite.py](03_benchmark_suite.py) | 完整配置矩阵、帕累托分析与报告生成 |

## 🚀 快速开始

> **前置条件：** Python 3.11+、API 密钥和 uv。完整配置说明请参阅 [SETUP.md](../../SETUP.md)。

```bash
uv run --directory 04-testing-evaluation/05-benchmarking python 01_model_comparison.py

# 示例
uv run --directory 04-testing-evaluation/05-benchmarking python 03_benchmark_suite.py
```

所有脚本都内置了**模拟结果**，无需 API 密钥也能运行。设置 `ANTHROPIC_API_KEY`（以及可选的 `OPENAI_API_KEY`）后，将启用实时模式。

你也可以使用 VS Code 的 [Code Runner](https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner) 扩展，一键运行当前打开的脚本。

## 🔑 核心概念

### 1. 受控实验

每次只改变一个变量，其余条件保持不变：

| 基准测试类型 | 变量 | 保持不变的条件 |
| ------------ | ---- | -------------- |
| 模型对比 | 模型 | 相同任务、相同提示词、相同评分器 |
| 提示词对比 | 提示词策略 | 相同模型、相同任务、相同评分器 |
| 完整矩阵 | 模型 × 提示词 | 相同任务、相同评分器 |

### 2. 多维评估

仅看准确率并不足够：

```python
@dataclass
class BenchmarkResult:
    keyword_score: float   # 质量：回答是否包含预期信息？
    latency_ms: float      # 速度：响应有多快？
    input_tokens: int      # 效率：消耗了多少 Token？
    cost_usd: float        # 成本：本次运行花费多少？
    tool_calls: int        # 行为：需要调用多少次工具？
```

### 3. 帕累托最优

如果不存在另一个配置在所有维度上都更优，那么该配置就是**帕累托最优**配置：

```
准确率 ↑
    │   ★ Sonnet+思维链（质量最高、成本最高）
    │
    │       ★ Sonnet+零样本（均衡性好）
    │
    │           ★ Haiku+少样本（优质选项中成本最低）
    │
    └──────────────────────────── 成本 →
```

帕累托前沿可以帮助回答：“每个任务预算为 X 美元时，我能获得的最佳效果是什么？”

### 4. 提示词策略的影响

不同提示词策略会在质量和成本之间做出不同权衡：

| 策略 | 准确率 | 成本 | 适用场景 |
| ---- | ------ | ---- | -------- |
| 零样本 | 基准水平 | 最低 | 简单且定义明确的任务 |
| 少样本 | +10–15% | 中等 | 模式清晰的任务 |
| 思维链 | +15–25% | 最高 | 复杂推理任务 |

## ⚠️ 重要注意事项

- **进行多次试验**——一次试验不能称为基准测试；每个配置至少运行 3～5 次
- **考虑方差**——输出具有非确定性，因此不同运行之间的结果会有差异
- **成本会迅速累积**——完整矩阵基准测试可能很昂贵，建议先从模拟模式开始
- **Token 定价会变化**——服务商更新价格后，及时调整 `cost_per_input_token` 和 `cost_per_output_token`

## 🔗 参考资料

- [Chatbot Arena：基于人类偏好评估大语言模型的开放平台——Chiang 等，2024](https://arxiv.org/abs/2403.04132)——介绍基于 Elo 评分的人类偏好基准测试方法与开放排行榜方案
- [语言模型的整体评估（HELM）——Liang 等，2022](https://arxiv.org/abs/2211.09110)——从准确率、稳健性、公平性和效率等维度进行综合评估
- [思维链提示激发大语言模型的推理能力——Wei 等，2022](https://arxiv.org/abs/2201.11903)——思维链提示领域的奠基论文，展示其对推理任务准确率的显著提升
- [语言模型是少样本学习器——Brown 等，2020](https://arxiv.org/abs/2005.14165)——提出以少样本上下文学习作为提示范式的 GPT-3 论文
- [AI 智能体基准测试——Evidently AI](https://www.evidentlyai.com/blog/ai-agent-benchmarks)——AI 智能体基准测试生态概览

## 👉 后续步骤

掌握基准测试后，可以继续：

- **[评估工具链](../07-eval-harness/)**——把全部 5 种技术整合到统一流水线中的综合项目
- **动手实验**——把你自己的模型和提示词策略添加到基准测试中
- **继续探索**——将基准结果与[教程 02](../02-evals/) 的评估分数结合起来
