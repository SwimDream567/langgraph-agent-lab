"""工具输出预算管理 — 固定上限 + 截断感知提示

对标 Claude Code 的工具输出限制策略：
  CC 对每个工具设固定上限（Bash 30K tokens、FileRead 25K tokens 等）。
  我们同样用固定上限，保证行为可预测。

截断感知提示（CC 没有但我们有的）：
  当输出被截断时，返回结果末尾明确告知 agent：
  "内容被截断到 X 字符，如需更多细节，请用 offset/limit 分段读取"。
  agent 能感知到截断，主动补读。

使用方式：
  from tools.output_budget import truncate_output
  result = truncate_output("read_file", raw_output)

配置（.env）：
  OUTPUT_MAX_CHARS_READ_FILE=30000
  OUTPUT_MAX_CHARS_RUN_COMMAND=10000
  ...
"""

import os

# ── 各工具的固定字符上限 ──
# 参考 Claude Code：Bash 30K tokens, FileRead 25K tokens
# 中文 1 token ≈ 1-2 字符，所以字符上限约等于 token 上限
TOOL_LIMITS = {
    "read_file":      int(os.environ.get("OUTPUT_MAX_CHARS_READ_FILE",      "30000")),
    "run_command":    int(os.environ.get("OUTPUT_MAX_CHARS_RUN_COMMAND",    "10000")),
    "search_content": int(os.environ.get("OUTPUT_MAX_CHARS_SEARCH_CONTENT", "8000")),
    "search_file":    int(os.environ.get("OUTPUT_MAX_CHARS_SEARCH_FILE",    "5000")),
    "web_fetch":      int(os.environ.get("OUTPUT_MAX_CHARS_WEB_FETCH",      "9000")),
    "web_search":     int(os.environ.get("OUTPUT_MAX_CHARS_WEB_SEARCH",     "3000")),
    "rag_search":     int(os.environ.get("OUTPUT_MAX_CHARS_RAG_SEARCH",     "5000")),
    "list_dir":       int(os.environ.get("OUTPUT_MAX_CHARS_LIST_DIR",       "5000")),
    "get_weather":    int(os.environ.get("OUTPUT_MAX_CHARS_GET_WEATHER",    "1000")),
    "get_current_time": int(os.environ.get("OUTPUT_MAX_CHARS_GET_TIME",     "200")),
    "write_file":     int(os.environ.get("OUTPUT_MAX_CHARS_WRITE_FILE",     "500")),
    "edit_file":      int(os.environ.get("OUTPUT_MAX_CHARS_EDIT_FILE",      "500")),
    "add_knowledge":  int(os.environ.get("OUTPUT_MAX_CHARS_ADD_KNOWLEDGE",  "200")),
}

# 默认上限（工具名不在列表中时使用）
DEFAULT_LIMIT = 10000

# 最低保障（无论如何至少保留这么多）
MIN_GUARANTEE = 500


def truncate_output(tool_name: str, output: str, limit: int = None) -> str:
    """截断工具输出到固定上限内

    智能截断策略：保留头部 70% + 尾部 30%，中间插入截断感知提示。
    agent 看到提示后知道内容不完整，可以主动用分段读取等方式补全。

    Args:
        tool_name: 工具名称，用于查找对应的上限
        output: 原始输出文本
        limit: 手动指定上限（覆盖工具默认值）

    Returns:
        截断后的文本，末尾带截断感知提示（如果被截断的话）
    """
    if limit is None:
        limit = TOOL_LIMITS.get(tool_name, DEFAULT_LIMIT)

    # 没超过上限，原样返回
    if len(output) <= limit:
        return output

    limit = max(limit, MIN_GUARANTEE)

    # 智能截断：头 70% + 尾 30%
    head_size = int(limit * 0.70)
    tail_size = int(limit * 0.25)  # 留 5% 给提示信息
    original_len = len(output)

    truncation_hint = (
        f"\n\n⚠️ [输出被截断] 原始 {original_len:,} 字符 → 截断到 {limit:,} 字符。"
        f"如需查看完整内容，请用 offset/limit 参数分段读取。"
    )

    truncated = (
        output[:head_size]
        + truncation_hint
        + output[-tail_size:]
    )
    return truncated


def get_tool_limit(tool_name: str) -> int:
    """获取指定工具的输出字符上限"""
    return TOOL_LIMITS.get(tool_name, DEFAULT_LIMIT)


def get_budget_info() -> str:
    """获取预算配置摘要（供 /context 命令显示）"""
    top_tools = ["read_file", "run_command", "search_content", "web_fetch", "web_search"]
    parts = [f"{t.split('_')[0]}: {TOOL_LIMITS.get(t, DEFAULT_LIMIT):,}" for t in top_tools if t in TOOL_LIMITS]
    return "固定上限 | " + " / ".join(parts) + " 字符"
