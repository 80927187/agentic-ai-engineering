"""
Promptfoo——YAML 驱动的评测框架。

演示如何将 Promptfoo 与基于 Python 的智能体集成。Promptfoo 是一个支持自定义
Python 提供器和断言的 Node.js CLI 工具，可以使用 YAML 以声明式方式定义评测套件。

本脚本将：
1. 根据黄金数据集中的测试用例生成 promptfooconfig.yaml
2. 创建封装研究助手的自定义 Python 提供器
3. 创建用于关键词评分的自定义 Python 断言
4. 展示如何通过 `npx promptfoo eval` 运行评测

注意：Promptfoo 需要 Node.js。可通过 `npm install -g promptfoo` 安装，或使用
`npx promptfoo@latest eval`。`pip install promptfoo` 软件包封装了 Node 二进制文件。
"""

import json
from pathlib import Path

from common import setup_logging
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from shared.knowledge_base import EVAL_TASKS, get_agent_response

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Promptfoo YAML 配置生成器
# ---------------------------------------------------------------------------


def generate_promptfoo_config(tasks: list[dict], output_dir: Path) -> str:
    """为研究助手生成 promptfooconfig.yaml。"""
    tests = []
    for task in tasks:
        test_case: dict = {
            "vars": {"question": task["question"]},
            "assert": [],
        }

        # 基于代码的断言：通过自定义 Python 函数检查关键词
        if task["expected_keywords"]:
            test_case["assert"].append(
                {
                    "type": "python",
                    "value": "file://assertion_keywords.py",
                    "metadata": {"keywords": task["expected_keywords"]},
                }
            )
        else:
            # 超出范围的问题：应包含拒答文本
            test_case["assert"].append(
                {
                    "type": "icontains",
                    "value": "无法在知识库中找到",
                }
            )

        # 来源引用检查
        for source_id in task.get("expected_source_ids", []):
            test_case["assert"].append(
                {
                    "type": "contains",
                    "value": source_id,
                }
            )

        # LLM 裁判评分标准（使用 Promptfoo 内置的 llm-rubric 断言）
        if task["expected_keywords"]:
            test_case["assert"].append(
                {
                    "type": "llm-rubric",
                    "value": (
                        f"回答应准确解答：‘{task['question']}’。"
                        f"回答应引用来源并涵盖以下主题："
                        f"{', '.join(task['expected_keywords'])}。"
                    ),
                }
            )

        tests.append(test_case)

    config = {
        "description": "研究助手评测套件",
        "providers": [
            {
                "id": "file://provider_agent.py",
                "label": "研究助手",
                "config": {"mode": "simulated"},
            }
        ],
        "prompts": ["{{question}}"],
        "tests": tests,
        "defaultTest": {
            "options": {
                "provider": "anthropic:messages:deepseek-v4-flash",
            }
        },
    }

    # 写入 YAML 配置
    import yaml  # type: ignore[import-untyped]

    config_path = output_dir / "promptfooconfig.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return config_path.name


def generate_provider_script(output_dir: Path) -> None:
    """生成 Promptfoo 的自定义 Python 提供器脚本。"""
    # Promptfoo 会为每个测试用例调用此脚本的 call_api 函数
    provider_code = '''"""
封装研究助手的自定义 Promptfoo 提供器。

Promptfoo 会为每个测试用例调用 call_api()。该函数接收：
- prompt：渲染后的提示词字符串
- options：包含 YAML 中 `config` 的字典
- context：包含测试用例中 `vars` 的字典
"""


def call_api(prompt, options, context):
    """Promptfoo 提供器入口。"""
    from shared.knowledge_base import get_agent_response, EVAL_TASKS

    question = context.get("vars", {}).get("question", prompt)

    # 根据问题文本查找匹配的任务
    task_id = None
    for task in EVAL_TASKS:
        if task["question"] == question:
            task_id = task["id"]
            break

    if task_id is None:
        return {"output": "未找到匹配的任务。"}

    response = get_agent_response(task_id)

    return {
        "output": response["answer"],
        "tokenUsage": {"total": 100, "prompt": 50, "completion": 50},
    }
'''
    (output_dir / "provider_agent.py").write_text(provider_code, encoding="utf-8")


def generate_assertion_script(output_dir: Path) -> None:
    """生成用于关键词评分的自定义 Python 断言脚本。"""
    assertion_code = '''"""
用于评定关键词覆盖率的自定义 Promptfoo 断言。

Promptfoo 会为每个 `python` 类型的断言调用 get_assert()。
返回包含 pass、score 和 reason 的字典。
"""


def get_assert(output, context):
    """检查智能体输出的关键词覆盖率。"""
    metadata = context.get("test", {}).get("metadata", {})
    keywords = metadata.get("keywords", [])

    if not keywords:
        return {"pass": True, "score": 1.0, "reason": "没有需要检查的关键词"}

    output_lower = output.lower()
    found = [kw for kw in keywords if kw.lower() in output_lower]
    missing = [kw for kw in keywords if kw.lower() not in output_lower]

    score = len(found) / len(keywords)
    passed = score >= 0.5

    reason = f"找到 {len(found)}/{len(keywords)} 个关键词"
    if missing:
        reason += f"（缺少：{', '.join(missing)}）"

    return {"pass": passed, "score": score, "reason": reason}
'''
    (output_dir / "assertion_keywords.py").write_text(assertion_code, encoding="utf-8")


# ---------------------------------------------------------------------------
# 主程序——生成配置并演示设置方式
# ---------------------------------------------------------------------------


def main() -> None:
    """生成 Promptfoo 配置并演示 YAML 驱动的评测模式。"""
    console = Console()
    console.print(
        Panel(
            "[bold cyan]Promptfoo——YAML 驱动的评测框架[/bold cyan]\n\n"
            "生成包含以下内容的 Promptfoo 评测套件：\n"
            "  - 自定义 Python 提供器（封装研究助手）\n"
            "  - 自定义 Python 断言（关键词评分）\n"
            "  - 内置断言（contains、icontains、llm-rubric）\n\n"
            "Promptfoo 是原生支持 Python 的 Node.js CLI。\n"
            "安装：npm install -g promptfoo",
            title="01 - Promptfoo",
        )
    )

    output_dir = Path(__file__).parent
    try:
        import yaml  # noqa: F401  # type: ignore[import-untyped]
    except ImportError:
        console.print(
            "[yellow]未安装 PyYAML——改为显示 JSON 配置。[/yellow]\n"
            "[dim]安装命令：pip install pyyaml[/dim]\n"
        )

    # 生成 Promptfoo 文件
    generate_provider_script(output_dir)
    generate_assertion_script(output_dir)
    console.print("[green]已生成 provider_agent.py 和 assertion_keywords.py[/green]\n")

    # 展示配置内容
    console.print("[bold]Promptfoo 配置（promptfooconfig.yaml）[/bold]\n")

    # 构建用于展示的示例配置（展示时避免依赖 yaml）
    sample_config = {
        "description": "研究助手评测套件",
        "providers": [
            {
                "id": "file://provider_agent.py",
                "label": "研究助手",
                "config": {"mode": "simulated"},
            }
        ],
        "prompts": ["{{question}}"],
        "tests": [
            {
                "vars": {"question": EVAL_TASKS[0]["question"]},
                "assert": [
                    {
                        "type": "python",
                        "value": "file://assertion_keywords.py",
                        "metadata": {"keywords": EVAL_TASKS[0]["expected_keywords"]},
                    },
                    {"type": "contains", "value": "doc_001"},
                    {
                        "type": "llm-rubric",
                        "value": "回答应准确涵盖微服务的优势。",
                    },
                ],
            },
            {"vars": {"question": "..."}, "assert": [{"type": "..."}]},
        ],
    }
    config_yaml = json.dumps(sample_config, indent=2, ensure_ascii=False)
    console.print(Syntax(config_yaml, "json", theme="monokai", line_numbers=True))

    # 如果 pyyaml 可用，则尝试生成 YAML 配置
    try:
        config_name = generate_promptfoo_config(EVAL_TASKS, output_dir)
        console.print(f"\n[green]已生成 {config_name}[/green]")
    except ImportError:
        console.print("\n[yellow]跳过 YAML 生成（未安装 pyyaml）[/yellow]")

    # 在本地运行断言以演示评分逻辑
    console.print("\n[bold]在本地运行断言（模拟模式）：[/bold]\n")

    # 内联运行生成的断言逻辑
    for task in EVAL_TASKS:
        response = get_agent_response(task["id"])
        output = response["answer"]

        # 关键词检查
        if task["expected_keywords"]:
            output_lower = output.lower()
            found = [kw for kw in task["expected_keywords"] if kw.lower() in output_lower]
            score = len(found) / len(task["expected_keywords"])
            status = "[green]通过[/green]" if score >= 0.5 else "[red]失败[/red]"
            console.print(f"  {task['id']}：{status} 关键词={score:.0%}", end="")
        else:
            has_refusal = "无法在知识库中找到" in output or "没有相关" in output
            status = "[green]通过[/green]" if has_refusal else "[red]失败[/red]"
            refusal_text = "是" if has_refusal else "否"
            console.print(f"  {task['id']}：{status} 拒答={refusal_text}", end="")

        # 来源引用检查
        for sid in task.get("expected_source_ids", []):
            cited = sid in output
            cite_status = "[green]是[/green]" if cited else "[red]否[/red]"
            console.print(f"  {sid}={cite_status}", end="")

        console.print()

    # 展示运行方式
    console.print(
        "\n[bold]使用 Promptfoo CLI 运行：[/bold]\n"
        "  [dim]npx promptfoo@latest eval[/dim]\n"
        "  [dim]npx promptfoo@latest view  # 打开网页界面查看结果[/dim]"
    )


if __name__ == "__main__":
    main()
