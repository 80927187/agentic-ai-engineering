"""
使用 Claude 进行视觉分析

演示如何将图像发送给 Claude 进行视觉理解，这是最基础的多模态能力。
支持基于 URL 的图像、本地文件分析以及多图像比较。
"""

import base64
import mimetypes
from pathlib import Path
from typing import Any

import anthropic
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from common import AnthropicTokenTracker, interactive_menu, setup_logging

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

MODEL = "claude-sonnet-4-6"

# 来自 Wikimedia Commons 的公开示例图像
SAMPLE_IMAGES = {
    "建筑——罗马斗兽场": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "d/de/Colosseo_2020.jpg/1280px-Colosseo_2020.jpg"
    ),
    "图表——世界人口": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "b/b7/Population_curve.svg/1280px-Population_curve.svg.png"
    ),
    "自然——北极光": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "a/aa/Polarlicht_2.jpg/1280px-Polarlicht_2.jpg"
    ),
}

ANALYSIS_TYPES = {
    "描述": "详细描述这张图像。你看到了什么？",
    "OCR / 文字提取": (
        "提取图像中可见的全部文字，并尽可能保持原有布局。"
    ),
    "详细分析": (
        "请详细分析这张图像，包括构图、色彩、主体、氛围和显著细节。"
        "如果是图表或文档，请解释其中的数据或内容。"
    ),
}


class VisionAnalyst:
    """使用 Claude 的视觉能力分析图像。"""

    def __init__(self, model: str, token_tracker: AnthropicTokenTracker) -> None:
        self.client = anthropic.Anthropic()
        self.model = model
        self.token_tracker = token_tracker

    def analyze_url(self, image_url: str, prompt: str) -> str:
        """分析 URL 中的图像。"""
        logger.info("Analyzing image URL: %s", image_url[:80])

        # 使用 URL 来源的图像内容块——Claude 会直接获取图像
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "url", "url": image_url},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        self.token_tracker.track(response.usage)
        logger.info(
            "Tokens — input: %d, output: %d",
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        result: str = response.content[0].text
        return result

    def analyze_file(self, image_path: str, prompt: str) -> str:
        """使用 Base64 编码分析本地图像文件。"""
        logger.info("Analyzing local file: %s", image_path)

        image_data, media_type = self._encode_image(image_path)

        # 使用 Base64 来源的图像内容块——内联发送图像数据
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        self.token_tracker.track(response.usage)
        logger.info(
            "Tokens — input: %d, output: %d",
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        result: str = response.content[0].text
        return result

    def compare_images(self, image_urls: list[str], prompt: str) -> str:
        """在单次请求中比较多张图像。"""
        logger.info("Comparing %d images", len(image_urls))

        # 构建内容块：在图像块之间交错插入文字标签
        content: list[dict[str, Any]] = []
        for i, url in enumerate(image_urls, 1):
            content.append({"type": "text", "text": f"图像 {i}："})
            content.append(
                {
                    "type": "image",
                    "source": {"type": "url", "url": url},
                }
            )

        content.append({"type": "text", "text": prompt})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": content}],
        )

        self.token_tracker.track(response.usage)
        logger.info(
            "Tokens — input: %d, output: %d",
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        result: str = response.content[0].text
        return result

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        """读取本地图像文件，返回（Base64 数据、媒体类型）。"""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 根据文件扩展名检测 MIME 类型
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            raise ValueError(f"不支持的图像类型：{mime_type}。请使用 JPEG、PNG、GIF 或 WebP。")

        data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
        logger.info("Encoded %s (%s, %d bytes)", path.name, mime_type, path.stat().st_size)
        return data, mime_type


def main() -> None:
    """交互式视觉分析演示。"""
    console = Console()
    token_tracker = AnthropicTokenTracker()
    analyst = VisionAnalyst(MODEL, token_tracker)

    welcome = Panel(
        "[bold cyan]使用 Claude 进行视觉分析[/bold cyan]\n\n"
        "发送图像给 Claude 进行视觉理解：\n"
        "  [green]•[/green] 分析 URL 中的示例图像\n"
        "  [green]•[/green] 分析本地图像文件（Base64）\n"
        "  [green]•[/green] 并排比较多张图像\n\n"
        "[dim]每张 1568×1568 像素的图像约消耗 1,600 个令牌[/dim]",
        title="多模态——视觉",
        border_style="blue",
    )

    image_menu_items = [
        *list(SAMPLE_IMAGES.keys()),
        "比较全部示例",
        "本地文件……",
    ]

    analysis_menu_items = list(ANALYSIS_TYPES.keys())

    try:
        while True:
            # Step 1: Select image source
            image_choice = interactive_menu(
                console,
                image_menu_items,
                title="选择图像",
                header=welcome,
                allow_custom=True,
                custom_label="自定义 URL……",
                custom_prompt="输入图像 URL",
            )

            if image_choice is None:
                break

            # Step 2: Select analysis type
            analysis_choice = interactive_menu(
                console,
                analysis_menu_items,
                title="选择分析类型",
                allow_custom=True,
                custom_label="自定义提示词……",
                custom_prompt="输入分析提示词",
            )

            if analysis_choice is None:
                continue

            prompt = ANALYSIS_TYPES.get(analysis_choice, analysis_choice)

            # Step 3: Execute analysis
            console.clear()
            console.print("\n[yellow]Analyzing...[/yellow]\n")

            try:
                if image_choice == "比较全部示例":
                    urls = list(SAMPLE_IMAGES.values())
                    result = analyst.compare_images(urls, prompt)
                elif image_choice == "本地文件……":
                    console.print("[bold green]输入文件路径：[/bold green] ", end="")
                    file_path = input().strip()
                    if not file_path:
                        continue
                    result = analyst.analyze_file(file_path, prompt)
                elif image_choice in SAMPLE_IMAGES:
                    url = SAMPLE_IMAGES[image_choice]
                    result = analyst.analyze_url(url, prompt)
                else:
                    # Custom URL entered by user
                    result = analyst.analyze_url(image_choice, prompt)

                # Display result
                console.print(
                    Panel(
                        Markdown(result),
                        title=f"[bold blue]分析结果：{analysis_choice}[/bold blue]",
                        border_style="green",
                    )
                )

                # Token summary for this call
                table = Table(show_header=False, box=None)
                table.add_column(style="dim")
                table.add_column(style="dim")
                table.add_row("输入令牌", f"{token_tracker.get_input_tokens():,}")
                table.add_row("输出令牌", f"{token_tracker.get_output_tokens():,}")
                table.add_row("令牌总数", f"{token_tracker.get_total_tokens():,}")
                console.print(table)

            except FileNotFoundError as e:
                console.print(f"\n[red]Error: {e}[/red]")
            except ValueError as e:
                console.print(f"\n[red]Error: {e}[/red]")
            except anthropic.APIError as e:
                logger.error("API error: %s", e)
                console.print(f"\n[red]API Error: {e}[/red]")

            console.print("\n[dim]按 Enter 键继续……[/dim]")
            input()

    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")

    # Final token report
    console.print()
    token_tracker.report()


if __name__ == "__main__":
    main()
