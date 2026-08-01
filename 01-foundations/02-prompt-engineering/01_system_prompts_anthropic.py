"""
系统提示词与角色工程（Anthropic）

通过比较三种配置，演示系统提示词如何控制大语言模型的行为：
- 通用助手（基线）
- 指定角色的专家
- 角色 + 约束 + 输出格式

三种配置会对相同的支持工单进行分类，以展示提示工程的影响。
"""

import anthropic
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel

from common import AnthropicTokenTracker, interactive_menu, setup_logging

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

# 含义模糊的支持工单，迫使系统提示词决定如何解读
SUPPORT_TICKETS = [
    {
        "label": "工单 1 — 性能问题投诉",
        "text": (
            "主题：应用更新后变得非常慢\n\n"
            "你好，自从最近一次更新后，应用加载任何内容都要花很长时间。"
            "以前能立即打开的页面现在会卡住 10 多秒。我使用的是 Wi-Fi，"
            "其他一切都运行正常。这实在让人沮丧——我工作需要用它。"
            "能请你们尽快修复吗？"
        ),
    },
    {
        "label": "工单 2 — 功能无法使用",
        "text": (
            "主题：导出按钮不起作用\n\n"
            "我一直在尝试导出报告，但点击导出按钮后没有任何反应。"
            "我已经试了很多次。我在 Windows 上使用 Chrome。"
            "我的同事说他们可以正常使用，但我不知道自己哪里操作错了。"
            "这是一个已知问题吗？"
        ),
    },
]

TICKET_LABELS = [t["label"] for t in SUPPORT_TICKETS]

# 三种系统提示词配置，展示逐步优化的过程
PROMPT_CONFIGS = [
    {
        "label": "A：通用助手",
        "system": "你是一名乐于助人的助手。请帮助分析这张支持工单。",
    },
    {
        "label": "B：指定角色的专家",
        "system": (
            "你是一家 SaaS 公司的高级支持工程师，已经对数千张工单进行过分类。"
            "分析工单时，你会找出最可能的根本原因、评估严重程度，并建议下一步行动。"
            "不要含糊其辞——请根据经验作出明确判断。"
        ),
    },
    {
        "label": "C：角色 + 约束 + 格式",
        "system": (
            "你是一家 SaaS 公司的高级支持工程师，已经对数千张工单进行过分类。\n\n"
            "严格按照以下分节作答：\n\n"
            "类别：缺陷 / 用户错误 / 功能请求 / 配置\n\n"
            "根本原因：一句话。\n\n"
            "严重程度：P1-P4\n\n"
            "下一步行动：支持团队可执行的一个具体步骤。\n\n"
            "保持简洁。不要提供要求之外的解释。"
        ),
    },
]

CONFIG_LABELS = [c["label"] for c in PROMPT_CONFIGS]


class PromptEngineer:
    """演示系统提示词如何塑造大语言模型的响应。"""

    def __init__(self, model: str, token_tracker: AnthropicTokenTracker):
        self.client = anthropic.Anthropic()
        self.model = model
        self.token_tracker = token_tracker

    def run(self, system_prompt: str, user_prompt: str) -> str:
        """使用给定的系统提示词和用户提示词执行一次大语言模型调用。"""
        logger.info("正在调用模型：%s", self.model)

        response = self.client.messages.create(
            model=self.model,
            temperature=0.1,
            max_tokens=200,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        self.token_tracker.track(response.usage)
        logger.info(
            "Token 数量 - 输入：%d，输出：%d",
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        return str(response.content[0].text)


def main() -> None:
    """使用三种不同的系统提示词执行支持工单分类。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()
    engineer = PromptEngineer("claude-sonnet-4-6", token_tracker)

    header = Panel(
        "[bold cyan]系统提示词与角色工程[/bold cyan]\n\n"
        "比较 3 种系统提示词配置在支持工单分类任务中的表现。\n"
        "观察随着提示词改进，响应风格和可操作性如何变化。",
        title="提示工程 — Anthropic",
    )

    try:
        while True:
            # 第 1 步：选择一张支持工单
            selection = interactive_menu(
                console,
                TICKET_LABELS,
                title="选择一张支持工单",
                header=header,
                allow_custom=True,
                custom_prompt="输入自定义支持工单",
            )
            if not selection:
                break

            ticket = next((t for t in SUPPORT_TICKETS if t["label"] == selection), None)
            ticket_text = ticket["text"] if ticket else selection
            ticket_label = ticket["label"] if ticket else "自定义工单"
            user_prompt = f"分析这张支持工单：\n\n{ticket_text}"

            # 第 2 步：选择要用于这张工单的提示词配置
            ticket_header = Panel(
                f"[bold magenta]{ticket_label}[/bold magenta]\n[dim]{ticket_text}[/dim]",
                title="已选工单",
                border_style="magenta",
            )

            while True:
                config_selection = interactive_menu(
                    console,
                    CONFIG_LABELS,
                    title="选择一种提示词配置",
                    header=ticket_header,
                )
                if not config_selection:
                    break

                config = next(c for c in PROMPT_CONFIGS if c["label"] == config_selection)

                console.print(f"\n[bold yellow]━━━ {config['label']} ━━━[/bold yellow]")
                console.print(Panel(config["system"], title="系统提示词", border_style="dim"))

                try:
                    response = engineer.run(config["system"], user_prompt)
                    console.print(Panel(response, title=config["label"], border_style="green"))
                except Exception as e:
                    logger.error("配置 %s 出错：%s", config["label"], e)

                token_tracker.report()
                token_tracker.reset()

                console.print("\n[dim]按 Enter 键继续……[/dim]")
                input()

    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")


if __name__ == "__main__":
    main()
