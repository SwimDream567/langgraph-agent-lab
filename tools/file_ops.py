"""文件操作工具集 — read_file / list_dir / search_file / search_content

提供安全的文件读取和搜索能力，不包含任何写入/删除操作。
用于 Agent 理解用户项目代码、查找文件、搜索内容。

使用方式:
    from tools.file_ops import read_file, list_dir, search_file, search_content
"""

import os
import re
import fnmatch
from pathlib import Path
from langchain_core.tools import tool

# ── 安全边界 ──────────────────────────────────────────────────────────────────
# 禁止访问的目录（系统关键路径）
_BLOCKED = {
    r"C:\Windows", r"C:\Program Files", r"C:\Program Files (x86)",
    r"C:\ProgramData",
}
# 单次读取最大行数（防止撑爆上下文）
MAX_READ_LINES = 500
# 搜索结果最大条数
MAX_SEARCH_RESULTS = 30


def _safe_path(path: str) -> str:
    """规范化路径并做基础安全检查，返回绝对路径"""
    p = os.path.abspath(os.path.expanduser(path))
    # 简单阻断系统目录
    for b in _BLOCKED:
        if p.lower().startswith(b.lower()):
            raise ValueError(f"禁止访问系统目录: {b}")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# ① read_file — 读取文件内容
# ══════════════════════════════════════════════════════════════════════════════

@tool
def read_file(path: str, offset: int = 1, limit: int = 200) -> str:
    """读取指定文件的内容（只读，不会修改文件）。

    Args:
        path:   文件的绝对路径或相对路径
        offset: 从第几行开始读（默认第 1 行）
        limit:  最多读多少行（默认 200，最大 500）
    """
    try:
        p = _safe_path(path)
        if not os.path.isfile(p):
            return f"文件不存在: {p}"

        limit = min(limit, MAX_READ_LINES)
        ext = os.path.splitext(p)[1].lower()
        # 跳过二进制文件
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
                    ".mp4", ".mp3", ".wav", ".avi", ".mkv", ".mov",
                    ".zip", ".rar", ".7z", ".tar", ".gz",
                    ".exe", ".dll", ".so", ".dylib", ".pyc", ".pyd",
                    ".pdf", ".doc", ".docx", ".pptx", ".xlsx"}:
            size = os.path.getsize(p)
            return f"二进制文件({ext}, {size/1024:.1f}KB): {p}\n（不支持直接读取，请用其他工具处理）"

        # 尝试 UTF-8，失败回退 GBK（Windows 常见）
        for enc in ("utf-8", "gbk", "latin-1"):
            try:
                with open(p, "r", encoding=enc) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        else:
            return "无法解码文件（非 UTF-8/GBK/Latin-1 编码）"

        total = len(lines)
        start = max(0, offset - 1)
        end = min(total, start + limit)
        sliced = lines[start:end]

        # 带行号输出
        header = f"📄 {os.path.basename(p)} ({total} 行, 显示 {start+1}-{end})\n"
        header += "─" * 50 + "\n"
        body = ""
        for i, line in enumerate(sliced, start + 1):
            body += f"{i:>5}│{line.rstrip()}\n"

        if end < total:
            body += f"\n... 还有 {total - end} 行未显示（用 offset={end+1} 继续读）"

        result = header + body

        # 字符数截断（固定上限 + 感知提示）
        from tools.output_budget import truncate_output
        result = truncate_output("read_file", result)

        return result

    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"读取失败: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# ② list_dir — 列出目录内容
# ══════════════════════════════════════════════════════════════════════════════

@tool
def list_dir(path: str, pattern: str = "*", recursive: bool = False) -> str:
    """列出指定目录下的文件和子目录（只读）。

    Args:
        path:      目录的绝对路径或相对路径
        pattern:   文件名匹配模式，如 '*.py'、'*.java'（默认 '*'）
        recursive: 是否递归子目录（默认 False）
    """
    try:
        p = _safe_path(path)
        if not os.path.isdir(p):
            return f"目录不存在: {p}"

        entries = []
        if recursive:
            for root, dirs, files in os.walk(p):
                # 跳过隐藏目录和常见大目录
                dirs[:] = [d for d in dirs
                           if not d.startswith(".")
                           and d not in {"node_modules", "__pycache__", ".git", "venv", ".idea"}]
                for f in files:
                    if fnmatch.fnmatch(f, pattern):
                        full = os.path.join(root, f)
                        rel = os.path.relpath(full, p)
                        size = os.path.getsize(full)
                        entries.append((rel, size, "file"))
                for d in dirs:
                    if fnmatch.fnmatch(d, pattern):
                        rel = os.path.relpath(os.path.join(root, d), p)
                        entries.append((rel, 0, "dir"))
        else:
            for name in sorted(os.listdir(p)):
                full = os.path.join(p, name)
                if not fnmatch.fnmatch(name, pattern):
                    continue
                if os.path.isfile(full):
                    entries.append((name, os.path.getsize(full), "file"))
                elif os.path.isdir(full):
                    entries.append((name, 0, "dir"))

        if not entries:
            return f"目录为空或无匹配项: {p} (pattern={pattern})"

        # 格式化输出
        header = f"📂 {os.path.basename(p) or p}（{len(entries)} 项"
        if recursive:
            header += "，递归"
        header += "）\n"
        header += "─" * 50 + "\n"

        lines = []
        files_count = 0
        for name, size, kind in sorted(entries, key=lambda x: (x[2] == "file", x[0])):
            if kind == "dir":
                lines.append(f"  📁 {name}/")
            else:
                files_count += 1
                if size < 1024:
                    sz = f"{size}B"
                elif size < 1024 * 1024:
                    sz = f"{size/1024:.1f}KB"
                else:
                    sz = f"{size/1024/1024:.1f}MB"
                lines.append(f"  📄 {name}  ({sz})")

        # 截断过长输出
        if len(lines) > MAX_SEARCH_RESULTS:
            lines = lines[:MAX_SEARCH_RESULTS]
            lines.append(f"\n  ... 共 {len(entries)} 项，只显示前 {MAX_SEARCH_RESULTS} 项")

        result = header + "\n".join(lines)

        # 字符级截断
        from tools.output_budget import truncate_output
        return truncate_output("list_dir", result)

    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"列出目录失败: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# ③ search_file — 按文件名搜索
# ══════════════════════════════════════════════════════════════════════════════

@tool
def search_file(path: str, pattern: str) -> str:
    """在指定目录下递归搜索匹配文件名的文件（类似 find 命令）。

    Args:
        path:    搜索的起始目录
        pattern: 文件名模式，如 '*.py'、'*test*'、'config.*'
    """
    try:
        p = _safe_path(path)
        if not os.path.isdir(p):
            return f"目录不存在: {p}"

        matches = []
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs
                       if not d.startswith(".")
                       and d not in {"node_modules", "__pycache__", ".git", "venv", ".idea"}]
            for f in files:
                if fnmatch.fnmatch(f.lower(), pattern.lower()):
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, p)
                    size = os.path.getsize(full)
                    matches.append((rel, size))

        if not matches:
            return f"未找到匹配 '{pattern}' 的文件（在 {p} 下）"

        lines = [f"🔍 搜索 '{pattern}'（在 {os.path.basename(p)} 下，共 {len(matches)} 个结果）\n"]
        lines.append("─" * 50)

        for rel, size in sorted(matches)[:MAX_SEARCH_RESULTS]:
            if size < 1024:
                sz = f"{size}B"
            elif size < 1024 * 1024:
                sz = f"{size/1024:.1f}KB"
            else:
                sz = f"{size/1024/1024:.1f}MB"
            lines.append(f"  {rel}  ({sz})")

        if len(matches) > MAX_SEARCH_RESULTS:
            lines.append(f"\n  ... 共 {len(matches)} 个，只显示前 {MAX_SEARCH_RESULTS} 个")

        result = "\n".join(lines)

        from tools.output_budget import truncate_output
        return truncate_output("search_file", result)

    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"搜索文件失败: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# ④ search_content — 搜索文件内容（grep）
# ══════════════════════════════════════════════════════════════════════════════

@tool
def search_content(path: str, pattern: str, file_glob: str = "*", context: int = 2) -> str:
    """在文件内容中搜索匹配的文本（类似 grep 命令）。

    Args:
        path:       搜索的起始目录
        pattern:    正则表达式或普通文本
        file_glob:  只搜索匹配的文件类型，如 '*.py'、'*.java'（默认 '*'）
        context:    每个匹配结果显示上下各几行（默认 2）
    """
    try:
        p = _safe_path(path)
        if not os.path.isdir(p):
            return f"目录不存在: {p}"

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"正则表达式错误: {e}"

        # 二进制扩展名跳过
        _binary = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
                    ".mp4", ".mp3", ".wav", ".avi", ".mkv", ".mov",
                    ".zip", ".rar", ".7z", ".tar", ".gz",
                    ".exe", ".dll", ".so", ".dylib", ".pyc", ".pyd",
                    ".pdf", ".doc", ".docx", ".pptx", ".xlsx"}

        results = []  # (filepath, [(lineno, line_text)])
        file_count = 0

        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs
                       if not d.startswith(".")
                       and d not in {"node_modules", "__pycache__", ".git", "venv", ".idea"}]
            for fname in files:
                if not fnmatch.fnmatch(fname.lower(), file_glob.lower()):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext in _binary:
                    continue

                full = os.path.join(root, fname)
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        all_lines = f.readlines()
                except (OSError, PermissionError):
                    continue

                hits = []
                for i, line in enumerate(all_lines):
                    if regex.search(line):
                        # 收集上下文
                        start = max(0, i - context)
                        end = min(len(all_lines), i + context + 1)
                        for j in range(start, end):
                            marker = ">>>" if j == i else "   "
                            hits.append((j + 1, marker, all_lines[j].rstrip()))

                if hits:
                    rel = os.path.relpath(full, p)
                    results.append((rel, hits))
                    file_count += 1

                if file_count >= 20:  # 最多搜 20 个文件
                    break
            if file_count >= 20:
                break

        if not results:
            return f"在 {p} 下未找到匹配 '{pattern}' 的内容"

        lines = [f"🔍 grep '{pattern}'（在 {os.path.basename(p)} 下，{file_count} 个文件命中）\n"]
        lines.append("─" * 50)

        total_matches = 0
        for filepath, hits in results:
            lines.append(f"\n📄 {filepath}")
            seen = set()
            for lineno, marker, text in hits:
                if lineno not in seen:
                    seen.add(lineno)
                    total_matches += 1
                    lines.append(f"  {lineno:>5}{marker}│{text}")

        lines.append(f"\n共 {total_matches} 处匹配")

        output = "\n".join(lines)
        # 固定上限截断 + 感知提示
        from tools.output_budget import truncate_output
        output = truncate_output("search_content", output)

        return output

    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"搜索内容失败: {e}"
