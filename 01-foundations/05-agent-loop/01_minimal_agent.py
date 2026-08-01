import subprocess
import anthropic
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "bash",
        "description": "运行一条 bash 命令",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
]


def agent(goal: str) -> str:
    messages = [{"role": "user", "content": goal}]

    for _ in range(10):
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=4096, messages=messages, tools=TOOLS
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return response.content[0].text  # type: ignore[no-any-return]

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  -> 工具：{block.name}({block.input})")
                # 执行前进行人机协同确认
                if input("  是否批准？(y/n)：").strip().lower() != "y":
                    return "已由用户取消。"
                result = subprocess.run(
                    block.input["command"], shell=True, capture_output=True, text=True, timeout=30
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result.stdout or result.stderr,
                    }
                )
        messages.append({"role": "user", "content": tool_results})  # type: ignore[dict-item]

    return "已达到最大迭代次数"


if __name__ == "__main__":
    print("最小智能体（输入 'quit' 退出）")
    print("可以尝试：'总结当前目录中的文件内容' 或 '我运行的是什么操作系统？'")
    while True:
        task = input("\n你：").strip()
        if task.lower() in ("exit", "quit", "q", ""):
            break
        print(f"\n智能体：{agent(task)}")
