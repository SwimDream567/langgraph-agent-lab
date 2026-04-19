"""会话记忆文件 — Claude Code Layer 2 风格（零成本自动追踪）

核心功能：
  1. 自动追踪：每轮对话后提取文件路径、工具调用、用户话题（纯规则，零 API 调用）
  2. 上下文注入：每轮对话前将会话记忆注入到消息中

设计原则：
  - 零成本：纯规则提取，不调用 LLM
  - 增量式：只追加新信息，去重
  - 轻量：注入内容控制在 800 字符以内
  - 自动瘦身：超过容量限制时自动淘汰旧条目

存储位置：chat_history_memory/<thread_id>.json

配置（.env）：
  MEMORY_UPDATE_INTERVAL=1    ← 每 N 轮保存一次文件（默认每轮都存）
"""

import os
import json
import time
import re
from pathlib import Path

MEMORY_DIR = Path(__file__).parent.parent / "chat_history_memory"
UPDATE_INTERVAL = int(os.environ.get("MEMORY_UPDATE_INTERVAL", "1"))

# 注入内容的最大字符数
MAX_MEMORY_CHARS = 800

# 识别文件路径的扩展名
_FILE_EXTS = {
    '.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c', '.h', '.cs',
    '.rb', '.php', '.swift', '.kt', '.scala', '.sh', '.bash', '.ps1',
    '.sql', '.r', '.m', '.lua', '.pl',
    '.html', '.htm', '.css', '.scss', '.xml', '.yaml', '.yml', '.json',
    '.toml', '.ini', '.cfg', '.conf', '.env',
    '.md', '.txt', '.markdown', '.rst', '.log',
    '.csv', '.tsv',
}


def _looks_like_file_path(s: str) -> bool:
    """判断字符串是否像文件路径"""
    if not isinstance(s, str) or len(s) < 5:
        return False
    # 过滤 URL
    if s.startswith(("http://", "https://", "ftp://", "www.")):
        return False
    # 过滤纯域名（含 / 但以 .com/.org 等结尾）
    if re.match(r'^[a-zA-Z0-9][\w.-]*\.(com|org|net|io|cn|dev|app|me|co|ly|ai|us|uk|de|fr|jp|ru|info|biz|xyz|top)\b', s):
        return False
    ext = os.path.splitext(s)[1].lower()
    if ext in _FILE_EXTS:
        return True
    # 路径分隔符 + 文件扩展名特征
    if ('\\' in s or '/' in s) and '.' in os.path.basename(s):
        base = os.path.basename(s)
        _, e = os.path.splitext(base)
        if e and len(e) <= 10 and e[1:].isalnum():
            return True
    return False


class SessionMemory:
    """管理每个会话的记忆文件"""

    def __init__(self):
        MEMORY_DIR.mkdir(exist_ok=True)
        self._cache = {}  # thread_id → memory dict

    def _path(self, thread_id: str) -> Path:
        return MEMORY_DIR / f"{thread_id}.json"

    def _load(self, thread_id: str) -> dict:
        if thread_id in self._cache:
            return self._cache[thread_id]

        p = self._path(thread_id)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                self._cache[thread_id] = data
                return data
            except (json.JSONDecodeError, IOError):
                pass

        default = {
            "thread_id": thread_id,
            "turn_count": 0,
            "last_updated": 0,
            "key_files": [],       # [{"path": str}]
            "tools_used": [],      # [str] 去重，最近 20 个
            "recent_topics": [],   # [str] 最近 8 条用户话题
        }
        self._cache[thread_id] = default
        return default

    def _save(self, thread_id: str):
        if thread_id not in self._cache:
            return
        self._cache[thread_id]["last_updated"] = time.time()
        p = self._path(thread_id)
        p.parent.mkdir(exist_ok=True)
        p.write_text(
            json.dumps(self._cache[thread_id], ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def track_from_messages(self, thread_id: str, messages: list):
        """从消息中提取信息并更新记忆文件

        在每轮对话结束后调用，传入最近几条消息即可（不需要全部历史）。

        Args:
            thread_id: 会话 ID
            messages: 最近几条消息（建议传最后 8 条）
        """
        mem = self._load(thread_id)
        mem["turn_count"] += 1

        for msg in messages:
            # ── 提取文件路径 ──
            # 从工具调用参数中提取（最可靠）
            for tc in (getattr(msg, 'tool_calls', None) or []):
                # 记录工具名
                tool_name = tc.get("name", "")
                if tool_name and tool_name not in mem["tools_used"]:
                    mem["tools_used"].append(tool_name)

                # 从参数中提取文件路径
                args = tc.get("args", {})
                for val in args.values():
                    if isinstance(val, str) and _looks_like_file_path(val):
                        existing = {f["path"] for f in mem["key_files"]}
                        if val not in existing and len(mem["key_files"]) < 15:
                            mem["key_files"].append({"path": val})

            # 从消息内容中也提取路径（兜底）
            content = getattr(msg, 'content', '') or ''
            for match in re.finditer(r'[A-Za-z]:[\\\/][\w\\\/\.\-]+\.\w{1,10}', content):
                fp = match.group(0)
                existing = {f["path"] for f in mem["key_files"]}
                if fp not in existing and len(mem["key_files"]) < 15:
                    mem["key_files"].append({"path": fp})

            # ── 提取用户话题 ──
            # 使用 type checking 避免 import 循环
            msg_type = type(msg).__name__
            if msg_type == "HumanMessage":
                topic_text = content
                if "[用户消息]" in content:
                    topic_text = content.split("[用户消息]")[-1].strip()
                topic = topic_text[:80].replace("\n", " ").strip()
                if topic and len(topic) > 2 and topic not in mem["recent_topics"]:
                    mem["recent_topics"].append(topic)

        # ── 自动瘦身 ──
        if len(mem["tools_used"]) > 20:
            mem["tools_used"] = mem["tools_used"][-20:]
        if len(mem["recent_topics"]) > 8:
            mem["recent_topics"] = mem["recent_topics"][-8:]

        # 保存
        if mem["turn_count"] % UPDATE_INTERVAL == 0 or mem["turn_count"] <= 3:
            self._save(thread_id)

    def get_memory_block(self, thread_id: str) -> str:
        """获取格式化的会话记忆（用于注入到消息中）

        Returns:
            格式化的记忆文本，约 200-800 字符。
            空字符串表示没有记忆（前几轮）。
        """
        mem = self._load(thread_id)

        if mem["turn_count"] == 0:
            return ""

        lines = ["## 会话记忆"]

        # 话题
        if mem["recent_topics"]:
            topics = mem["recent_topics"][-5:]
            lines.append(f"- 最近话题：{'；'.join(topics)}")

        # 关键文件
        if mem["key_files"]:
            files = [f["path"] for f in mem["key_files"][-8:]]
            short = []
            for f in files:
                parts = f.replace("\\", "/").split("/")
                short.append("/".join(parts[-2:]) if len(parts) > 2 else f)
            lines.append(f"- 涉及文件：{', '.join(short)}")

        # 工具使用
        if mem["tools_used"]:
            lines.append(f"- 用过的工具：{', '.join(mem['tools_used'][-10:])}")

        lines.append(f"- 对话轮次：{mem['turn_count']}")

        result = "\n".join(lines)

        if len(result) > MAX_MEMORY_CHARS:
            result = result[:MAX_MEMORY_CHARS] + "..."

        return result

    def get_recent_file_context(self, thread_id: str, max_files: int = 3, max_chars_per_file: int = 1500) -> str:
        """压缩后重建：读取最近操作的文件内容，注入回上下文

        在 Layer 3 压缩后调用。压缩可能丢失了工具输出中的文件内容，
        这个方法重新读取最近 2-3 个文件，让 agent 不会"失忆"。

        Args:
            thread_id: 会话 ID
            max_files: 最多读取几个文件（默认 3）
            max_chars_per_file: 每个文件最多读取多少字符（默认 1500）

        Returns:
            格式化的文件内容文本，用于注入到消息中。
            空字符串表示没有可读取的文件。
        """
        mem = self._load(thread_id)
        if not mem["key_files"]:
            return ""

        # 取最近的几个文件
        recent_files = mem["key_files"][-max_files:]
        blocks = []

        for entry in reversed(recent_files):  # 最近的排前面
            fp = entry["path"]
            p = Path(fp)
            if not p.exists() or not p.is_file():
                continue
            try:
                # 只读前 N 个字符
                content = p.read_text(encoding="utf-8", errors="replace")
                if len(content) > max_chars_per_file:
                    content = content[:max_chars_per_file] + "\n... (截断)"
                if content.strip():
                    short_name = "/".join(fp.replace("\\", "/").split("/")[-2:])
                    blocks.append(f"### {short_name}\n```\n{content.strip()}\n```")
            except (OSError, PermissionError):
                continue

        if not blocks:
            return ""

        header = "## 压缩恢复 — 近期文件内容"
        result = header + "\n\n" + "\n\n".join(blocks)

        # 总大小控制在 5000 字符内
        if len(result) > 5000:
            result = result[:5000] + "\n... (截断)"

        return result

    def get_stats(self, thread_id: str) -> dict:
        """获取记忆文件统计信息"""
        mem = self._load(thread_id)
        return {
            "turn_count": mem["turn_count"],
            "key_files": len(mem["key_files"]),
            "tools_used": len(mem["tools_used"]),
            "recent_topics": len(mem["recent_topics"]),
            "memory_chars": len(self.get_memory_block(thread_id)),
        }


# 全局单例
session_memory = SessionMemory()
