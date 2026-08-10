"""
工具测试

隔离测试工具函数，验证输入、输出格式、错误处理和边界情况，
整个过程完全不涉及大语言模型。

核心测试概念：
- 纯函数单元测试：工具具有确定性，可以直接测试
- 基于夹具的准备工作：使用 pytest 夹具创建临时文件和共享状态
- 边界情况覆盖：除数为零、文件缺失、命令被禁止和超时
"""

from pathlib import Path

import pytest

from shared.tools import BLOCKED_COMMANDS, calculator, execute_tool, read_file, run_bash


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    """创建内容已知的临时文件用于测试。"""
    filepath = tmp_path / "sample.txt"
    filepath.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n", encoding="utf-8")
    return filepath


@pytest.fixture()
def long_file(tmp_path: Path) -> Path:
    """创建包含大量行的临时文件，用于测试截断功能。"""
    filepath = tmp_path / "long.txt"
    content = "\n".join(f"line {i}" for i in range(1, 201))
    filepath.write_text(content, encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# 计算器测试
# ---------------------------------------------------------------------------


class TestCalculator:
    """计算器工具测试。"""

    def test_calculator_add(self) -> None:
        """验证加法返回正确结果。"""
        result = calculator("add", 3, 5)
        assert result["result"] == 8
        assert result["operation"] == "add"
        assert result["operands"] == [3, 5]

    def test_calculator_subtract(self) -> None:
        """验证减法返回正确结果。"""
        result = calculator("subtract", 10, 4)
        assert result["result"] == 6

    def test_calculator_multiply(self) -> None:
        """验证乘法返回正确结果。"""
        result = calculator("multiply", 7, 8)
        assert result["result"] == 56

    def test_calculator_divide(self) -> None:
        """验证除法返回正确结果。"""
        result = calculator("divide", 20, 4)
        assert result["result"] == 5.0

    def test_calculator_division_by_zero(self) -> None:
        """验证除数为零时返回错误字符串，而不是引发异常。"""
        result = calculator("divide", 10, 0)
        assert result["result"] == "错误：除数不能为零"

    def test_calculator_float_precision(self) -> None:
        """验证计算器能够处理浮点数。"""
        result = calculator("add", 0.1, 0.2)
        assert abs(result["result"] - 0.3) < 1e-9


# ---------------------------------------------------------------------------
# 文件读取测试
# ---------------------------------------------------------------------------


class TestReadFile:
    """read_file 工具测试。"""

    def test_read_file_success(self, sample_file: Path) -> None:
        """验证成功读取文件时会返回内容和元数据。"""
        result = read_file(str(sample_file))
        assert "line 1" in result["content"]
        assert result["total_lines"] == 5
        assert result["truncated"] is False

    def test_read_file_not_found(self) -> None:
        """验证文件缺失时会返回错误字典，而不是引发异常。"""
        result = read_file("/nonexistent/path/file.txt")
        assert "error" in result
        assert "找不到文件" in result["error"]

    def test_read_file_truncation(self, long_file: Path) -> None:
        """验证 max_lines 参数能够正确截断输出。"""
        result = read_file(str(long_file), max_lines=3)
        # 内容中应当只有前 3 行
        assert result["content"].count("\n") <= 3
        assert result["truncated"] is True
        assert result["total_lines"] == 200


# ---------------------------------------------------------------------------
# Bash 命令测试
# ---------------------------------------------------------------------------


class TestRunBash:
    """run_bash 工具测试。"""

    def test_run_bash_success(self) -> None:
        """验证简单命令能够执行并返回输出。"""
        result = run_bash("echo hello")
        assert result["stdout"].strip() == "hello"
        assert result["exit_code"] == 0

    def test_run_bash_blocked_commands(self) -> None:
        """验证每个禁止命令都会在执行前被拒绝。"""
        for cmd in BLOCKED_COMMANDS:
            result = run_bash(f"{cmd} something")
            assert "error" in result, f"命令“{cmd}”应被阻止"
            assert "阻止" in result["error"]

    def test_run_bash_rm_rf_blocked(self) -> None:
        """验证典型的危险命令会被阻止。"""
        result = run_bash("rm -rf /")
        assert "error" in result
        assert "阻止" in result["error"]

    def test_run_bash_timeout(self) -> None:
        """验证长时间运行的命令会受到超时限制。"""
        result = run_bash("sleep 10", timeout=1)
        assert "error" in result
        assert "超时" in result["error"]

    def test_run_bash_nonexistent_command(self) -> None:
        """验证无效命令会返回非零退出码。"""
        result = run_bash("nonexistent_command_xyz_123")
        assert result["exit_code"] != 0


# ---------------------------------------------------------------------------
# 工具分发器测试
# ---------------------------------------------------------------------------


class TestExecuteTool:
    """execute_tool 分发器测试。"""

    def test_execute_tool_unknown_tool(self) -> None:
        """验证未知工具名会返回错误。"""
        result = execute_tool("nonexistent_tool", {})
        assert "error" in result
        assert "未知工具" in result["error"]

    def test_execute_tool_invalid_args(self) -> None:
        """验证错误参数会返回错误，而不是导致程序崩溃。"""
        # calculator 需要 operation、a、b，这里传入错误的键
        result = execute_tool("calculator", {"wrong_key": "value"})
        assert "error" in result
        assert "参数无效" in result["error"]

    def test_execute_tool_dispatches_correctly(self) -> None:
        """验证分发器会路由到正确的工具函数。"""
        result = execute_tool("calculator", {"operation": "add", "a": 1, "b": 2})
        assert result["result"] == 3

    def test_execute_tool_read_file_dispatch(self, sample_file: Path) -> None:
        """验证分发器能够正确路由到 read_file。"""
        result = execute_tool("read_file", {"path": str(sample_file)})
        assert "content" in result
        assert "line 1" in result["content"]
