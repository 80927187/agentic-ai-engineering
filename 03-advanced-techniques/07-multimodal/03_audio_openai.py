"""
使用 OpenAI 音频的语音助手

演示如何使用 OpenAI 音频 API 进行文本转语音和语音转文本。
包含 6 种声音的 TTS、Whisper 转录，以及将文本转换为语音再转录以进行验证的往返演示。
"""

from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from common import interactive_menu, setup_logging

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
VOICE_DESCRIPTIONS = {
    "alloy": "中性、平衡",
    "echo": "温暖、健谈",
    "fable": "富有表现力、叙事感",
    "onyx": "深沉、权威",
    "nova": "充满活力、友好",
    "shimmer": "清晰、柔和",
}

TTS_MODEL = "tts-1"
STT_MODEL = "whisper-1"

SAMPLE_TEXTS = {
    "问候": "你好！我是你的 AI 语音助手。我可以使用六种不同的声音说话。",
    "故事": (
        "很久以前，在一个充满无限可能的地方，"
        "一个小机器人学会了说话。它说的第一句话是：“我思故我在。”"
    ),
    "技术": (
        "Transformer 架构使用自注意力机制并行处理序列，"
        "在自然语言处理领域取得了最先进的成果。"
    ),
    "诗歌": (
        "一片黄树林里分出两条路，可惜我无法同时踏上，"
        "我选择了人迹较少的一条，这带来了一切不同。"
    ),
}

OUTPUT_DIR = Path("output")


class VoiceAssistant:
    """通过 OpenAI 处理文本转语音和语音转文本。"""

    def __init__(self) -> None:
        self.client = OpenAI()
        # OpenAI 音频 API 不返回令牌用量，因此改为跟踪 API 调用次数
        # TTS 按字符计费，STT（Whisper）按音频分钟计费
        self.api_call_count = 0

    def speak(self, text: str, voice: str = "alloy") -> str:
        """将文本转换为语音并保存为 MP3。"""
        logger.info("TTS: voice=%s, text=%s", voice, text[:50])

        OUTPUT_DIR.mkdir(exist_ok=True)
        file_path = OUTPUT_DIR / f"tts_{voice}_{self.api_call_count}.mp3"

        response = self.client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=text,
        )

        response.write_to_file(str(file_path))
        self.api_call_count += 1

        file_size = file_path.stat().st_size
        logger.info("Saved audio: %s (%d bytes)", file_path, file_size)
        return str(file_path)

    def transcribe(self, audio_path: str) -> str:
        """使用 Whisper 将音频文件转录为文本。"""
        logger.info("STT: transcribing %s", audio_path)

        with open(audio_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model=STT_MODEL,
                file=audio_file,
            )

        self.api_call_count += 1
        logger.info("Transcription: %s", transcription.text[:80])
        result: str = transcription.text
        return result

    def round_trip(self, text: str, voice: str = "alloy") -> tuple[str, str]:
        """文本 → 语音 → 转录。返回（音频路径、转录文本）。"""
        logger.info("Round-trip: voice=%s", voice)

        audio_path = self.speak(text, voice)
        transcription = self.transcribe(audio_path)

        return audio_path, transcription

    def voice_comparison(self, text: str) -> list[tuple[str, str]]:
        """使用全部 6 种声音生成同一段文本。返回（声音、路径）列表。"""
        logger.info("Voice comparison: generating %d voices", len(VOICES))

        results: list[tuple[str, str]] = []
        for voice in VOICES:
            path = self.speak(text, voice)
            results.append((voice, path))

        return results


def main() -> None:
    """交互式语音助手演示。"""
    console = Console()
    assistant = VoiceAssistant()

    welcome = Panel(
        "[bold cyan]使用 OpenAI 音频的语音助手[/bold cyan]\n\n"
        "文本转语音和语音转文本功能：\n"
        "  [green]•[/green] TTS——使用 6 种声音将文本转为语音\n"
        "  [green]•[/green] STT——使用 Whisper 转录音频文件\n"
        "  [green]•[/green] 往返验证——文本 → 语音 → 转录 → 比较\n"
        "  [green]•[/green] 声音比较——试听全部 6 种声音\n\n"
        "[dim]音频文件保存到 output/ 目录[/dim]",
        title="多模态——音频",
        border_style="blue",
    )

    menu_items = [
        "TTS 演示",
        "声音比较",
        "往返验证",
        "转录文件",
    ]

    try:
        while True:
            choice = interactive_menu(
                console,
                menu_items,
                title="选择模式",
                header=welcome,
            )

            if choice is None:
                break

            console.clear()

            try:
                if choice == "TTS 演示":
                    _handle_tts_demo(console, assistant)

                elif choice == "声音比较":
                    _handle_voice_comparison(console, assistant)

                elif choice == "往返验证":
                    _handle_round_trip(console, assistant)

                elif choice == "转录文件":
                    _handle_transcription(console, assistant)

            except Exception as e:
                logger.error("Error: %s", e)
                console.print(f"\n[red]Error: {e}[/red]")

            console.print("\n[dim]按 Enter 键继续……[/dim]")
            input()

    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")

    # Report API call count (audio APIs don't use tokens)
    console.print(f"\n[dim]API 调用总数：{assistant.api_call_count}[/dim]")


def _handle_tts_demo(console: Console, assistant: VoiceAssistant) -> None:
    """选择声音和文本进行文本转语音。"""
    # Select text
    text_choice = interactive_menu(
        console,
        list(SAMPLE_TEXTS.keys()),
        title="选择文本",
        allow_custom=True,
        custom_label="自定义文本……",
        custom_prompt="输入要朗读的文本",
    )
    if text_choice is None:
        return

    text = SAMPLE_TEXTS.get(text_choice, text_choice)

    # Select voice
    voice_items = [f"{v} — {VOICE_DESCRIPTIONS[v]}" for v in VOICES]
    voice_choice = interactive_menu(console, voice_items, title="选择声音")
    if voice_choice is None:
        return

    voice = voice_choice.split(" — ")[0]

    console.clear()
    console.print(f"\n[yellow]正在使用“{voice}”声音生成语音……[/yellow]\n")

    file_path = assistant.speak(text, voice)

    console.print(
        Panel(
            f"[bold]声音：[/bold] {voice}（{VOICE_DESCRIPTIONS[voice]}）\n"
            f"[bold]文本：[/bold] {text}\n"
            f"[bold]文件：[/bold] {file_path}",
            title="[bold green]音频已生成[/bold green]",
            border_style="green",
        )
    )


def _handle_voice_comparison(console: Console, assistant: VoiceAssistant) -> None:
    """使用全部 6 种声音生成同一段文本。"""
    text_choice = interactive_menu(
        console,
        list(SAMPLE_TEXTS.keys()),
        title="选择用于比较的文本",
        allow_custom=True,
        custom_label="自定义文本……",
        custom_prompt="输入要用不同声音比较的文本",
    )
    if text_choice is None:
        return

    text = SAMPLE_TEXTS.get(text_choice, text_choice)

    console.clear()
    console.print(f"\n[yellow]正在生成 {len(VOICES)} 个声音样本……[/yellow]\n")

    results = assistant.voice_comparison(text)

    table = Table(title="声音比较结果")
    table.add_column("声音", style="cyan")
    table.add_column("描述", style="dim")
    table.add_column("文件", style="green")

    for voice, path in results:
        table.add_row(voice, VOICE_DESCRIPTIONS[voice], path)

    console.print(table)
    console.print(f"\n[dim]文本：{text}[/dim]")


def _handle_round_trip(console: Console, assistant: VoiceAssistant) -> None:
    """文本 → 语音 → 转录 → 比较。"""
    text_choice = interactive_menu(
        console,
        list(SAMPLE_TEXTS.keys()),
        title="选择用于往返测试的文本",
        allow_custom=True,
        custom_label="自定义文本……",
        custom_prompt="输入用于往返测试的文本",
    )
    if text_choice is None:
        return

    text = SAMPLE_TEXTS.get(text_choice, text_choice)

    # Select voice
    voice_items = [f"{v} — {VOICE_DESCRIPTIONS[v]}" for v in VOICES]
    voice_choice = interactive_menu(console, voice_items, title="选择声音")
    if voice_choice is None:
        return

    voice = voice_choice.split(" — ")[0]

    console.clear()
    console.print("\n[yellow]正在运行往返流程：文本 → 语音 → 转录……[/yellow]\n")

    audio_path, transcription = assistant.round_trip(text, voice)

    # Compare original and transcribed text
    original_lower = text.lower().strip()
    transcribed_lower = transcription.lower().strip()
    match = original_lower == transcribed_lower

    console.print(
        Panel(
            Markdown(
                f"**原文：** {text}\n\n"
                f"**转录文本：** {transcription}\n\n"
                f"**音频文件：** {audio_path}\n\n"
                f"**匹配：** {'完全匹配' if match else '检测到差异（正常现象——Whisper 可能会调整标点）'}"
            ),
            title="[bold blue]往返结果[/bold blue]",
            border_style="green" if match else "yellow",
        )
    )


def _handle_transcription(console: Console, assistant: VoiceAssistant) -> None:
    """转录现有音频文件。"""
    console.print("\n[bold green]输入音频文件路径：[/bold green] ", end="")
    audio_path = input().strip()

    if not audio_path:
        return

    if not Path(audio_path).exists():
        console.print(f"[red]找不到文件：{audio_path}[/red]")
        return

    console.print(f"\n[yellow]正在转录 {audio_path}……[/yellow]\n")

    transcription = assistant.transcribe(audio_path)

    console.print(
        Panel(
            Markdown(transcription),
            title="[bold blue]转录结果[/bold blue]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
