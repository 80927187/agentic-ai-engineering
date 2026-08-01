# 构建智能体式 AI 系统：Anthropic 智能体框架的 5 种核心模式

五种核心工作流模式构成了生产级智能体系统的基础，每种模式都针对特定的运营需求和复杂度进行了优化。Anthropic 的研究表明，成功并非来自复杂性，而在于为具体工作选择正确的模式。

## 模式 1：使用提示链分解顺序任务

提示链将复杂任务分解为若干独立步骤，每次 LLM 调用都会处理上一步的输出。Anthropic 的实践显示，在多步骤工作流中，与使用单一提示词相比，这种模式可将任务失败率降低 34%。

以处理一万页监管申报材料的文档审查流水线为例。步骤 1 提取相关章节（输出包含章节 ID 和页码的 JSON）；步骤 2 总结每个章节（输出带有 1 至 10 风险评分的结构化摘要）；步骤 3 标记合规问题（输出布尔标记及具体法规引用）；步骤 4 生成最终建议（输出按优先级排序的行动项）。

每一步都有清晰的输入/输出约定。当摘要步骤失败时，你可以立即知道应该在哪里调试，而不必等到整个文档处理完毕。

**延迟注意事项**：提示链最多保持在 3 至 5 个步骤。超过这个阈值后，大多数应用的累计延迟会超过 15 秒，而且每增加一个步骤，上下文劣化都会使准确率下降 12% 至 18%。

**错误处理**：在每一步实现断路器。如果步骤 2 连续失败三次，应转交人工审核，而不是继续执行提示链。

## 模式 2：具备生产级错误处理的稳健工具集成

未经封装的 API 集成在生产环境中很容易失败。Anthropic 对企业部署的分析表明，60% 的智能体故障源于未处理的工具错误，而非 LLM 推理问题。

为每个外部 API 封装便于智能体处理的错误机制：

```python
def call_payment_api(amount: float, account_id: str) -> dict:
    """使用指数退避和断路器处理付款。"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.payment.service/charge",
                json={"amount": amount, "account": account_id},
                timeout=5
            )
            response.raise_for_status()
            return {"status": "success", "transaction_id": response.json()["id"]}
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
                continue
            return {"status": "failed", "reason": "timeout_after_retries"}
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # 速率限制
                return {"status": "failed", "reason": "rate_limited", "retry_after": 60}
            return {"status": "error", "code": e.response.status_code}
```

**工具粒度**：构建原子化工具，而不是大而全的函数。支付系统需要分别提供验证、处理和确认工具，而不是一个包办所有工作的支付工具。Anthropic 的内部测试表明，这种方法可将工具选择错误减少 40%。

**模型上下文协议（MCP）**：使用 Slack、Google Drive 和 Salesforce 等提供商已有的 MCP 服务器，而不是自行开发集成。使用 MCP 的团队报告称，集成速度提高了 70%，维护问题减少了 50%。

## 模式 3：通过专业化角色编排多个智能体

当单个智能体拥有 50 多个工具时，工具选择的混淆率会提高 60%。解决方法是使用具有专注工具集的专业智能体，并由一个编排器进行协调。

客户支持自动化的架构示例：

- **分诊智能体**：5 个工具（工单分类、紧急程度评分、路由决策）
- **研究智能体**：8 个工具（知识库搜索、客户历史查询、产品文档）
- **解决方案智能体**：6 个工具（退款处理、账户更新、邮件撰写）

编排器根据工单类型进行路由，并在置信度低于 0.7 时升级处理。与处理相同工单量的单智能体系统相比，这种方式可将响应时间缩短 45%。

**通信协议**：为智能体之间的交接定义明确的模式。智能体 A 输出 `{"ticket_id": "123", "classification": "billing", "confidence": 0.85, "extracted_data": {...}}`，智能体 B 则严格按此格式接收，不留歧义。

**冲突解决**：当智能体意见不一致时（例如研究智能体标记为“低风险”，而分诊智能体标记为“高风险”），应将两份评估结果同时展示并转交人工审核。

## 模式 4：策略性设置人在回路检查点

人在回路（HITL）检查点既能防止代价高昂的错误，又能维持自动化效率。关键在于按阈值实施：常规决策自动执行，高风险操作则必须获得批准。

金融服务智能体的实现示例：

```python
# 智能体起草退款决定
refund_proposal = agent.decide_refund(customer_id="123")

# 金额低于 100 美元时自动批准
if refund_proposal["amount"] < 100:
    execute_refund(refund_proposal)
else:
    # 金额较大时需要人工批准
    approval = await request_approval(
        action="process_refund",
        amount=refund_proposal["amount"],
        reason=refund_proposal["reason"],
        customer_history=refund_proposal["context"]
    )
```

**阈值优化**：采用 500 美元阈值的公司报告称，其自动化率达到 94%，且从未发生超过 500 美元的未授权退款。较低的阈值（低于 100 美元）会带来 99.8% 的批准率，使人工审核失去效率。

**批准延迟**：应设计异步工作流。平均批准时间为 12 分钟；智能体应该妥善暂停，并在收到批准后恢复，而不是超时或重新启动。

## 模式 5：超越任务完成情况的全面评估

大多数团队只衡量任务是否完成（“智能体完成工作了吗？”），却忽略行为是否对齐（“它正确地完成工作了吗？”）。Anthropic 的 Bloom 框架同时涵盖了这两个维度。

客户服务智能体的**任务完成指标**：

- 解决率：87% 的工单无需升级即可关闭
- 准确率：92% 的事实陈述经验证正确
- 延迟：平均响应时间低于 30 秒

**行为对齐指标**：

- 语气分析：94% 的回复被评为“专业且有帮助”
- 政策合规：99.2% 的回复遵守公司退款政策
- 幻觉率：包含无法验证陈述的回复少于 2%

Bloom 的自动评估会为每种行为生成 500 多个测试场景、模拟用户交互，并为智能体回复打分。使用 Bloom 的团队报告称，与人工评估相比，他们识别行为漂移的速度提高了 40%。

**生产监控**：在实时智能体旁部署影子评估。将 5% 的生产流量交给并行评估智能体，让其对真实交互评分，但不影响客户体验。

## 要点总结

- **对于包含 3 个以上顺序步骤的工作流，先从提示链开始**——在每一步实现明确的输入/输出模式和断路器，以妥善处理故障。

- **为每个外部 API 封装带有指数退避和超时逻辑的错误处理工具**——使用已有 MCP 服务器代替自行开发集成，将开发时间缩短 70%。

- **仅在单个智能体使用的工具超过 15 个时部署多智能体架构**——创建各自专注于 5 至 8 个工具的专业智能体，并使用明确的 JSON 模式进行智能体间通信。

- **对超过风险阈值的操作实施人在回路审批**——设置金额限制（例如退款上限为 500 美元），并设计可在等待批准期间妥善暂停的异步工作流。

- **从第一天起同时衡量任务完成情况和行为对齐情况**——跟踪准确率、延迟和政策合规率，并对 5% 的生产流量运行影子评估，以尽早发现行为漂移。

## 资料来源

- [使用 Anthropic 的 6 种可组合模式构建 AI 智能体](https://aimultiple.com/building-ai-agents)
- [Anthropic 为 Claude 推出 Skills 开放标准](https://aibusiness.com/foundation-models/anthropic-launches-skills-open-standard-claude)
- [Agent Skills：Anthropic 定义 AI 标准的下一步尝试](https://thenewstack.io/agent-skills-anthropics-next-bid-to-define-ai-standards/)
- [Anthropic 智能体 | Microsoft Learn](https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-types/anthropic-agent)
- [Anthropic 发布 Bloom：用于前沿 AI 模型自动行为评估的开源智能体框架](https://www.marktechpost.com/2025/12/21/anthropic-ai-releases-bloom-an-open-source-agentic-framework-for-automated-behavioral-evaluations-of-frontier-ai-models/)
- [Anthropic](https://www.anthropic.com/research/bloom)
- [Anthropic 将 Agent Skills 设为开放标准](https://siliconangle.com/2025/12/18/anthropic-makes-agent-skills-open-standard/)
- [2026 智能体编程趋势报告：编程智能体如何重塑开发](https://resources.anthropic.com/hubfs/2026%20Agentic%20Coding%20Trends%20Report.pdf?hsLang=en)
- [五种智能体工作流模式：Anthropic 的构建框架](https://danieldavenport.medium.com/five-agentic-workflow-patterns-9f03e356d031)
- [AI 智能体究竟如何工作：2025 年 7 月](https://bertomill.medium.com/how-ai-agents-actually-work-july-2025-fe44405be906)
- [提示链 | 提示工程指南](https://www.promptingguide.ai/techniques/prompt_chaining)
- [提示链工作流——AWS 规范性指导](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/workflow-for-prompt-chaining.html)
- [AI 智能体中的提示链：模块化、可靠且可扩展的工作流](https://medium.com/@nivalabs.ai/prompt-chaining-for-the-ai-agents-modular-reliable-and-scalable-workflows-a22d15fd5d33)
- [如何使用提示链和 AI 原语构建自主智能体（不使用框架）](https://www.freecodecamp.org/news/build-autonomous-agents-using-prompt-chaining-with-ai-primitives/)
- [构建高效的 AI 智能体](https://www.anthropic.com/research/building-effective-agents)
- [提示链与智能体式 AI：最佳用例对比](https://aicompetence.org/prompt-chaining-vs-agentic-ai-use-cases-compared/)
- [智能体式 AI 提示工程：面向 CTO 和数据负责人的最佳实践](https://ubtiinc.com/agentic-ai-prompt-engineering-key-concepts-techniques-and-best-practices/)
- [智能体式 AI 中的提示链：复杂思考如何逐步涌现](https://medium.com/@pillaisatheesh74/prompt-chaining-in-agentic-ai-how-complex-thinking-emerges-one-step-at-a-time-2b53e2834033)
- [什么是 AI 提示链？（2026 教程）](https://www.voiceflow.com/blog/prompt-chaining)
- [AI 提示与提示链完整指南](https://metaflow.life/blog/prompt-chaining)
- [AI 智能体 API：5 种集成模式（2026 指南）](https://composio.dev/blog/apis-ai-agents-integration-patterns)
- [如何为 AI 智能体集成工具？完整实施策略](https://zenvanriel.nl/ai-engineer-blog/how-to-integrate-tools-with-ai-agents-implementation-guide/)
- [智能体 SDK](https://chatbotkit.com/manuals/agent-sdk)
- [AI 智能体工具：教程与示例](https://www.patronus.ai/ai-agent-development/ai-agent-tools)
- [AI 智能体工具集成：使用 Spring AI 构建强大的智能体](https://medium.com/@mehhmetoz/ai-agent-tool-integration-building-powerful-agents-with-spring-ai-74e70a10e3bb)
- [编排复杂 AI 工作流：高级集成模式](https://www.getknit.dev/blog/orchestrating-complex-ai-workflows-advanced-integration-patterns)
- [使用逻辑与控制构建自定义 AI 智能体 | n8n 自动化平台](https://n8n.io/ai-agents/)
- [赋予 AI 智能体行动能力：掌握工具调用与函数执行](https://www.getknit.dev/blog/empowering-ai-agents-to-act-mastering-tool-calling-function-execution)
- [AI 智能体入门：开始构建 AI 智能体的 12 节课](https://microsoft.github.io/ai-agents-for-beginners/04-tool-use/)
- [企业中的 AI 智能体：Amazon Bedrock AgentCore 最佳实践](https://aws.amazon.com/blogs/machine-learning/ai-agents-in-enterprises-best-practices-with-amazon-bedrock-agentcore/)
