"""
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
