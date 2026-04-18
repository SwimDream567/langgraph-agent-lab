"""文件编辑工具 — write_file / edit_file

提供安全的文件写入和编辑能力。
edit_file 采用精确字符串替换（和 Claude Code 的 Edit 一样），避免整文件覆写。

使用方式:
    from tools.file_edit import write_file, edit_file
"""

import os
from pathlib import Path
from langchain_core.tools import tool

# ── 安全边界（和 file_ops.py 一致）──────────────────────────────────────────────
_BLOCKED = {
    r"C:\Windows", r"C:\Program Files", r"C:\Program Files (x86)",
    r"C:\ProgramData",
}


def _safe_path(path: str) -> str:
    """规范化路径并做基础安全检查"""
    p = os.path.abspath(os.path.expanduser(path))
    for b in _BLOCKED:
        if p.lower().startswith(b.lower()):
            raise ValueError(f"禁止操作系统目录: {b}")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# ① write_file — 创建或覆盖文件
# ══════════════════════════════════════════════════════════════════════════════

@tool
def write_file(path: str, content: str) -> str:
    """创建或覆盖文件。会自动创建父目录。

    Args:
        path:    文件的绝对路径或相对路径
        content: 要写入的完整内容
    """
    try:
        p = _safe_path(path)

        # 自动创建父目录
        os.makedirs(os.path.dirname(p), exist_ok=True)

        # 写入（UTF-8）
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)

        size = len(content.encode("utf-8"))
        lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return f"✅ 已写入: {p}（{lines} 行, {size/1024:.1f}KB）"

    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"写入失败: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# ② edit_file — 精确字符串替换（Claude Code Edit 风格）
# ══════════════════════════════════════════════════════════════════════════════

@tool
def edit_file(path: str, old_str: str, new_str: str) -> str:
    """精确替换文件中的某段文本。old_str 必须在文件中唯一匹配。

    Args:
        path:    文件的绝对路径或相对路径
        old_str: 要被替换的原文本（必须精确匹配，包括缩进和空行）
        new_str: 替换后的新文本（传空字符串表示删除）
    """
    try:
        p = _safe_path(path)
        if not os.path.isfile(p):
            return f"文件不存在: {p}"

        # 读取（尝试多种编码）
        for enc in ("utf-8", "gbk", "latin-1"):
            try:
                with open(p, "r", encoding=enc) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            return "无法解码文件"

        # 检查 old_str 是否存在
        count = content.count(old_str)
        if count == 0:
            return f"❌ 未找到匹配文本。请检查 old_str 是否精确（包括缩进、空格、换行）"
        if count > 1:
            return f"❌ 找到 {count} 处匹配，old_str 必须在文件中唯一。请扩大上下文使其唯一。"

        # 执行替换
        new_content = content.replace(old_str, new_str, 1)

        with open(p, "w", encoding=enc) as f:
            f.write(new_content)

        # 计算变更
        old_lines = old_str.count("\n") + 1
        new_lines = new_str.count("\n") + 1 if new_str else 0
        action = "替换" if new_str else "删除"
        return f"✅ 已{action}: {p}（{old_lines} 行 → {new_lines} 行）"

    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"编辑失败: {e}"
