<!-- ---
title: "代理单元测试"
description: "模拟大语言模型响应，以确定性的方式测试代理行为"
icon: "check-square"
--- -->

# 代理单元测试

学习如何在不调用 API 的情况下测试 AI 代理。通过模拟大语言模型响应，并测试模型外围的工具执行、决策路由、消息构造和错误处理，可以构建快速、确定性的测试，捕获真实缺陷。

## 🎯 你将学到什么

- 模拟大语言模型响应，创建确定性的测试场景
- 隔离测试工具函数，包括输入验证、输出格式和错误处理
- 定义并验证行为契约，即代理始终必须做或绝不能做的事情
- 使用 pytest 运行代理测试套件
- 通过依赖注入构建易于测试的代理
- 使用录制文件记录和重放 API 响应，完成集成测试
- 通过快照测试和令牌预算断言发现回归

## 📦 可用示例

| 脚本 | 文件 | 说明 |
| ---- | ---- | ---- |
| 运行测试 | [01_run_tests.py](01_run_tests.py) | 使用 Rich 界面和确认提示运行完整测试套件 |

### 测试模块

| 测试 | 文件 | 说明 |
| ---- | ---- | ---- |
| 模拟大语言模型 | [tests/test_mock_llm.py](tests/test_mock_llm.py) | 模拟 `client.messages.create()`，测试代理循环逻辑 |
| 工具 | [tests/test_tools.py](tests/test_tools.py) | 隔离测试工具函数及其边界情况 |
| 契约 | [tests/test_behavioral_contracts.py](tests/test_behavioral_contracts.py) | 定义并验证代理的不变量 |
| 集成 | [tests/test_integration.py](tests/test_integration.py) | 记录并重放 API 响应，执行快照回归测试 |

## 🚀 快速开始

> **前置条件：** Python 3.11+、API 密钥和 uv。完整配置说明请参阅 [SETUP.md](../../SETUP.md)。本教程的测试本身不需要 API 密钥。

```bash
# 使用 Rich 界面运行测试套件（显示概览并请求确认）
uv run --directory 04-testing-evaluation/01-unit-testing-agents python 01_run_tests.py

# 通过 pytest 直接运行全部测试（42 个测试）
uv run --directory 04-testing-evaluation/01-unit-testing-agents pytest tests/ -v

# 运行单个测试模块
uv run --directory 04-testing-evaluation/01-unit-testing-agents pytest tests/test_mock_llm.py -v

# 运行指定的测试类
uv run --directory 04-testing-evaluation/01-unit-testing-agents pytest tests/test_behavioral_contracts.py::TestSafetyContracts -v
```

也可以使用 VS Code 的 [Code Runner](https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner) 扩展，一键运行当前打开的脚本。

## 🔑 核心概念

### 1. 代理测试金字塔

传统的测试金字塔同样适用于代理，但带有 AI 特有的变化：层级越高，就越依赖实时大语言模型调用和统计断言，而不是确定性检查。

| 层级 | 速度 | 成本 | 测试内容 |
| ---- | ---- | ---- | -------- |
| **单元（模拟）** ← test_mock_llm、test_tools | 快 | 免费 | 工具执行、路由、消息构造、错误处理 |
| **契约** ← test_behavioral_contracts | 快 | 免费 | 安全、终止和历史记录不变量 |
| **集成** ← test_integration | 中等 | 免费 | 使用已录制或缓存的大语言模型响应测试端到端流程 |
| **评估** | 慢 | $$ | 通过实时调用和统计断言评估大语言模型的推理质量 |

### 2. 通过依赖注入提高可测试性

让代理易于测试的关键是**注入大语言模型客户端**，而不是在代理内部创建客户端：

```python
class ToolUseAgent:
    """通过依赖注入实现可测试性的工具调用代理。"""

    def __init__(self, client: Any, model: str = "claude-sonnet-4-5-20250929") -> None:
        self.client = client  # 注入的客户端——测试时可以替换为模拟对象
        self.model = model
```

### 3. 模拟大语言模型响应

使用 `unittest.mock` 创建虚假的 Anthropic 响应：

```python
def create_mock_response(content, stop_reason="end_turn"):
    response = MagicMock()
    response.content = content
    response.stop_reason = stop_reason
    response.usage = Mock(input_tokens=100, output_tokens=50)
    return response

# 在测试中
mock_client = MagicMock()
mock_client.messages.create.return_value = create_mock_response(...)
agent = ToolUseAgent(client=mock_client)
```

### 4. 行为契约

定义不变量，即代理始终必须做或绝不能做的事情：

```python
def test_agent_never_executes_blocked_commands():
    """代理绝不能执行 rm、sudo、chmod 等命令。"""
    # 模拟大语言模型请求被禁止的命令
    # 验证工具返回错误，而不是执行结果

def test_agent_stops_after_max_iterations():
    """代理必须在 N 次迭代后终止，不能无限循环。"""
    agent = SafeToolUseAgent(client=mock, max_iterations=3)
    # 模拟大语言模型始终请求使用工具
    # 验证代理恰好在 3 次迭代后停止
```

### 5. 响应录制文件（记录/重放）

与其维护脆弱的 `MagicMock` 数据结构，不如将真实 API 响应记录到 JSON 文件中，并在测试时重放。录制系统还能发现行为偏离：如果代理的调用次数超过已记录数量，测试就会失败。

```python
class CassetteClient:
    """从录制文件中重放响应。"""

    def __init__(self, cassette_path: Path) -> None:
        with cassette_path.open() as f:
            self._interactions = json.load(f)
        self._call_index = 0
        self.messages = self  # 像真实客户端一样提供 messages.create

    def create(self, **kwargs) -> CassetteResponse:
        if self._call_index >= len(self._interactions):
            raise RuntimeError("录制响应已用尽——代理行为已发生偏离")
        response = self._interactions[self._call_index]["response"]
        self._call_index += 1
        return CassetteResponse(response)
```

### 6. 快照回归测试

将代理输出与黄金基准进行比较，以发现回归。如果代码变更导致输出不同、令牌用量激增或消息历史结构改变，快照测试就会捕获这些问题。

```python
def test_output_matches_snapshot(cassette_dir):
    golden_snapshot = "12 乘以 15 等于 180。"
    client = CassetteClient(write_cassette(cassette_dir, "calc", CASSETTE_DATA))
    agent = ToolUseAgent(client=client)
    result = agent.send_message("12 * 15 等于多少？")
    assert result == golden_snapshot, f"输出发生偏移：{result!r}"
```

## ⚠️ 重要注意事项

- **模拟测试的是脚手架，而不是智能**——模拟测试验证代理外围逻辑，而不是大语言模型的推理能力。质量验证仍需要评估测试（教程 02）。
- **保持模拟响应真实**——模拟响应应与实际 API 格式一致。不真实的模拟会带来虚假的信心。
- **测试错误路径**——API 故障、格式错误的工具输出和超时往往最容易隐藏缺陷。

## 🔗 资源

- [超越准确率：使用 CheckList 对 NLP 模型进行行为测试——Ribeiro 等，2020](https://arxiv.org/abs/2005.04118)——介绍不变性、方向性和最小功能测试的行为测试奠基论文
- [构建高效代理——Anthropic](https://www.anthropic.com/research/building-effective-agents)——介绍工具调用、循环和委派等代理模式，帮助确定测试内容及依赖注入方式
- [pytest 文档](https://docs.pytest.org/)——测试运行器、夹具、参数化和模拟集成
- [unittest.mock——Python 文档](https://docs.python.org/3/library/unittest.mock.html)——用于模拟大语言模型客户端的 `MagicMock`、`patch` 和 `side_effect`

## 👉 后续步骤

掌握单元测试后，可以继续：

- **[评估](../02-evals/)**——从确定性断言扩展到使用黄金数据集和“大语言模型裁判”的统计评估
- **实验**——为自己的代理添加更多行为契约
- **练习**——为[模块 01](../../01-foundations/) 中的代理编写单元测试
