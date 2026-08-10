<!-- ---
title: "追踪与调试"
description: "追踪每一次 LLM 调用、工具调用和决策点"
icon: "search"
--- -->

# 追踪与调试

当智能体做出意外行为时，你需要准确知道原因。追踪会捕获完整的执行流程——每一次 LLM 调用、工具调用、决策点和中间结果——让你能够在事后重建智能体的推理路径。

本教程使用纯 Python，讲解如何将可观测性作为一等公民。无需外部依赖——先学习概念，再将其应用于生产工具。

## 🎯 你将学到什么

- 使用上下文管理器和装饰器构建基于跨度（span）的追踪收集器
- 将执行追踪可视化为 Rich 树形层次结构
- 检测反模式：调用过多、循环、重复搜索和令牌用量过高
- 比较同一任务在不同运行中的追踪
- 通过逐步检查已记录的追踪来调试智能体故障
- 从检查点重放智能体执行过程

## 📦 可用示例

| 脚本 | 文件 | 说明 |
| ---- | ---- | ---- |
| 追踪收集器 | [01_trace_collector.py](01_trace_collector.py) | 使用跨度、上下文管理器和装饰器构建 `TraceCollector` |
| 追踪分析 | [02_trace_analysis.py](02_trace_analysis.py) | 加载追踪、检测反模式并计算指标 |
| 追踪调试 | [03_trace_debugging.py](03_trace_debugging.py) | 定位故障点、提取决策路径并进行重放 |

## 🚀 快速开始

> **前置条件：** Python 3.11+、API 密钥和 uv。完整设置说明请参阅 [SETUP.md](../../SETUP.md)。

```bash
uv run --directory 04-testing-evaluation/03-tracing-debugging python 01_trace_collector.py

# 示例
uv run --directory 04-testing-evaluation/03-tracing-debugging python 02_trace_analysis.py
```

所有脚本都包含示例追踪数据，无需 API 密钥即可运行。设置 `ANTHROPIC_API_KEY` 后会自动启用实时模式。

你也可以使用 VS Code 的 [Code Runner](https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner) 扩展，一键运行当前打开的脚本。

## 🔑 核心概念

### 1. 基于跨度的追踪

每个操作都是一个跨度，其中包含耗时、输入、输出和子跨度：

```python
@dataclass
class Span:
    name: str           # "llm_call_1"、"search_knowledge_base"
    span_type: str      # "llm_call"、"tool_call"、"agent_step"
    start_time: float
    end_time: float
    tokens: dict        # {"input": 150, "output": 80}
    children: list      # 嵌套的子跨度
    error: str | None   # 跨度失败时的错误消息
```

### 2. 使用上下文管理器追踪

`TraceCollector` 使用上下文管理器自动管理跨度的生命周期：

```python
tracer = TraceCollector()

with tracer.span("answer_question", "agent_step") as root:
    with tracer.span("llm_call", "llm_call") as llm_span:
        response = client.messages.create(...)
        llm_span.tokens = {"input": 150, "output": 80}

    with tracer.span("search", "tool_call") as tool_span:
        results = search_knowledge_base(query)
```

### 3. 反模式检测

自动分析能够发现常见的智能体问题：

| 反模式 | 表现 | 常见原因 |
| ------ | ---- | -------- |
| LLM 调用过多 | 一个简单问题调用超过 5 次 | 缺少停止条件 |
| 重复搜索 | 同一查询搜索两次 | 未缓存结果 |
| 令牌用量过高 | 简单任务使用超过 2000 个令牌 | 提示词冗长或出现循环 |
| 工具调用失败 | 工具出错后未重试 | 缺少错误处理 |
| 跨度耗时过长 | 单个操作超过 10 秒 | API 超时或出现循环 |

### 4. 基于追踪的调试

当评估失败时，追踪会显示故障路径：

```text
1. 从失败结果开始向前回溯
2. 找到第一个发生错误或输出异常的跨度
3. 检查导致错误决策的输入
4. 使用 TraceReplay 从该检查点重新执行
```

## ⚠️ 重要注意事项

- **纯 Python，并非生产实现**——本教程用于讲解概念。生产环境请使用 [Langfuse](https://langfuse.com/)、[Datadog](https://www.datadoghq.com/) 或 [OpenTelemetry](https://opentelemetry.io/)
- **追踪存储量增长很快**——生产环境中应对追踪进行采样，并设置保留策略
- **成本归因很重要**——了解哪个步骤成本最高，有助于指导优化

## 🔗 资源

- [OpenTelemetry 文档](https://opentelemetry.io/docs/)——生产可观测性中用于跨度、追踪和上下文传播的行业标准规范
- [Langfuse——开源 LLM 可观测性平台](https://langfuse.com/)——支持成本追踪、评分和提示词管理的生产级 LLM 追踪工具
- [Dapper：大规模分布式系统追踪基础设施——Sigelman 等，2010](https://research.google/pubs/dapper-a-large-scale-distributed-systems-tracing-infrastructure/)——Google 关于基于跨度的分布式追踪的奠基性论文，并启发了 OpenTelemetry
- [可观测性的三大支柱——Charity Majors](https://www.oreilly.com/library/view/distributed-systems-observability/9781492033431/ch04.html)——将指标、日志和追踪作为互补的可观测性信号

## 👉 后续步骤

掌握追踪后，可以继续：

- **[红队测试与安全](../04-red-teaming-safety/)**——测试智能体抵御对抗性攻击的能力
- **动手实验**——使用 `TraceCollector` 为你自己的智能体添加追踪
- **深入探索**——将追踪与[教程 02](../02-evals/)中的评估失败关联起来
