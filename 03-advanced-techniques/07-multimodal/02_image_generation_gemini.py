"""
使用 Gemini 生成图像

演示 Google Gemini 的原生图像生成功能——同一个模型既能理解也能创建图像。
无需独立端点，只需将 IMAGE 加入 response_modalities。
"""

from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from common import GeminiTokenTracker, interactive_menu, setup_logging

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

# 支持原生图像生成的 Gemini 模型
MODEL = "gemini-2.0-flash-exp-image-generation"

SAMPLE_PROMPTS = {
    "风景": (
        "日出时分宁静的山间湖泊，水面升起薄雾，静止的湖面倒映着松树，写实风格"
    ),
    "肖像": (
        "一位演奏萨克斯的爵士乐手水彩肖像，暖色调，富有表现力的笔触，艺术风格"
    ),
    "抽象": (
        "使用鲜明原色的抽象几何艺术，重叠的圆形和三角形，简洁线条，现代极简风格"
    ),
    "产品": (
        "木桌上的陶瓷咖啡杯极简产品照片，柔和自然光，浅景深，影棚品质"
    ),
    "建筑": (
        "带有垂直花园和玻璃幕墙的未来建筑，周围环绕树木，蓝天，建筑可视化风格"
    ),
}

OUTPUT_DIR = Path("output")


class ImageGenerator:
    """使用 Gemini 的原生图像生成功能。"""

    def __init__(self, model: str, token_tracker: GeminiTokenTracker) -> None:
        # genai.Client() 会自动从环境变量读取 GOOGLE_API_KEY
        self.client = genai.Client()
        self.model = model
        self.token_tracker = token_tracker

    def generate(self, prompt: str) -> tuple[str, bytes | None]:
        """根据文本提示词生成图像。

        返回（文本响应、图像字节）；生成失败时 image_bytes 为 None。
        """
        logger.info("Generating image for prompt: %s", prompt[:60])

        # 关键：response_modalities=["TEXT", "IMAGE"] 请求混合输出
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )

        # 通过 Gemini 的 usage_metadata 跟踪令牌用量
        if response.usage_metadata:
            self.token_tracker.track(response.usage_metadata)
            logger.info(
                "Tokens — input: %d, output: %d",
                response.usage_metadata.prompt_token_count or 0,
                response.usage_metadata.candidates_token_count or 0,
            )

        # 解析响应部分——可能包含文本和/或 inline_data（图像）
        text_parts: list[str] = []
        image_bytes: bytes | None = None

        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.text:
                    text_parts.append(part.text)
                elif part.inline_data:
                    # inline_data 包含 .data（字节）和 .mime_type
                    image_bytes = part.inline_data.data
                    logger.info(
                        "Received image: %s, %d bytes",
                        part.inline_data.mime_type,
                        len(image_bytes),
                    )

        text_response = "\n".join(text_parts) if text_parts else "图像生成成功。"
        return text_response, image_bytes

    def save_image(self, image_bytes: bytes, filename: str | None = None) -> str:
        """将生成的图像保存到输出目录。"""
        OUTPUT_DIR.mkdir(exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{timestamp}.png"

        file_path = OUTPUT_DIR / filename
        file_path.write_bytes(image_bytes)
        logger.info("Saved image to %s (%d bytes)", file_path, len(image_bytes))
        return str(file_path)


def main() -> None:
    """交互式图像生成演示。"""
    console = Console()
    token_tracker = GeminiTokenTracker()
    generator = ImageGenerator(MODEL, token_tracker)

    welcome = Panel(
        "[bold cyan]使用 Gemini 生成图像[/bold cyan]\n\n"
        "使用 Gemini 的原生生成功能根据文本提示词创建图像：\n"
        "  [green]•[/green] 同一模型既理解又创建图像\n"
        "  [green]•[/green] 无需独立图像 API——使用 generate_content\n"
        "  [green]•[/green] 图像保存到 output/ 目录\n\n"
        "[dim]需要在 .env 中设置 GOOGLE_API_KEY[/dim]",
        title="多模态——图像生成",
        border_style="blue",
    )

    prompt_items = list(SAMPLE_PROMPTS.keys())

    try:
        while True:
            # Select or enter a prompt
            choice = interactive_menu(
                console,
                prompt_items,
                title="选择提示词",
                header=welcome,
                allow_custom=True,
                custom_label="自定义提示词……",
                custom_prompt="描述你想生成的图像",
            )

            if choice is None:
                break

            prompt = SAMPLE_PROMPTS.get(choice, choice)

            console.clear()
            console.print(Panel(prompt, title="[bold]提示词[/bold]", border_style="cyan"))
            console.print("\n[yellow]正在生成图像……（可能需要一点时间）[/yellow]\n")

            try:
                text_response, image_bytes = generator.generate(prompt)

                if image_bytes:
                    file_path = generator.save_image(image_bytes)
                    console.print(
                        Panel(
                            Markdown(text_response),
                            title="[bold blue]Gemini 响应[/bold blue]",
                            border_style="green",
                        )
                    )
                    console.print(f"\n[bold green]图像已保存：[/bold green] {file_path}")
                    console.print(f"[dim]大小：{len(image_bytes):,} 字节[/dim]")
                else:
                    console.print(
                        Panel(
                            Markdown(text_response),
                            title="[bold blue]Gemini 响应[/bold blue]",
                            border_style="yellow",
                        )
                    )
                    console.print("[yellow]响应中未生成图像。[/yellow]")

                # Token summary
                table = Table(show_header=False, box=None)
                table.add_column(style="dim")
                table.add_column(style="dim")
                table.add_row("输入令牌", f"{token_tracker.get_input_tokens():,}")
                table.add_row("输出令牌", f"{token_tracker.get_output_tokens():,}")
                table.add_row("令牌总数", f"{token_tracker.get_total_tokens():,}")
                console.print(table)

            except Exception as e:
                logger.error("Generation error: %s", e)
                console.print(f"\n[red]Error: {e}[/red]")

            console.print("\n[dim]按 Enter 键继续……[/dim]")
            input()

    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")

    # Final token report
    console.print()
    token_tracker.report()


if __name__ == "__main__":
    main()
