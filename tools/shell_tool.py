"""命令执行工具 — run_command

安全的 Shell 命令执行，带超时、输出截断和危险命令检测。

使用方式:
    from tools.shell_tool import run_command
"""

import os
import subprocess
import re
from langchain_core.tools import tool

# ── 安全配置 ──────────────────────────────────────────────────────────────────

# 危险命令黑名单（正则匹配）
_DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bdel\s+/[sS]\s+/[qQ]\b",
    r"\brd\s+/[sS]\s+/[qQ]\b",
    r"\bformat\s+[a-zA-Z]:",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\btaskkill\s+/[fF]",
    r"\breg\s+delete\b",
    r"\bnet\s+user\b",
    r"\bpowershell\s+-enc\b",
    r"\bcmd\s+/c\b.*\bdel\b",
    r"\bgit\s+push\s+.*--force\b",
    r"\bgit\s+reset\s+--hard\b",
]

# 最大输出长度（字符）
MAX_OUTPUT = 10000
# 默认超时（秒）
DEFAULT_TIMEOUT = 30


@tool
def run_command(command: str, cwd: str = "", timeout: int = DEFAULT_TIMEOUT) -> str:
    """在终端执行命令并返回输出结果。

    Args:
        command: 要执行的命令（如 "dir", "python test.py", "git status"）
        cwd:     工作目录（默认当前目录）
        timeout: 超时秒数（默认 30，最大 120）
    """
    try:
        # 危险命令检测
        for pattern in _DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return f"⛔ 拒绝执行危险命令: 匹配规则 '{pattern}'\n命令: {command}"

        # 参数校验
        timeout = min(max(timeout, 1), 120)

        # 工作目录
        work_dir = os.path.abspath(cwd) if cwd else os.getcwd()
        if not os.path.isdir(work_dir):
            return f"工作目录不存在: {work_dir}"

        # 执行命令
        result = subprocess.run(
            command,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )

        # 拼接输出
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")

        output = "\n".join(output_parts) if output_parts else "(无输出)"

        # 截断过长输出
        truncated = False
        if len(output) > MAX_OUTPUT:
            output = output[:MAX_OUTPUT]
            truncated = True

        # 格式化返回
        header = f"📦 命令: {command}\n"
        header += f"   目录: {work_dir} | 耗时: ≤{timeout}s | 退出码: {result.returncode}\n"
        header += "─" * 50 + "\n"

        if truncated:
            footer = f"\n... 输出过长，已截断（共 {len(output)} 字符）"
        else:
            footer = ""

        return header + output + footer

    except subprocess.TimeoutExpired:
        return f"⏰ 命令超时（{timeout}s）: {command}\n进程已终止。"
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"执行失败: {e}"
