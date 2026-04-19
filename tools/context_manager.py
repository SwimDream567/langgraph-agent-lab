"""上下文管理器 — 多层压缩策略（Claude Code 风格）

架构：
  Layer 1: MicroCompact  — 清除旧工具输出（0 成本，回收 60-70% token）
  Layer 2: 会话记忆摘要  — 零成本自动追踪文件/工具/话题（session_memory.py）
  Layer 3: 完整摘要压缩  — LLM 一次性摘要 + 消息修剪

原理参考 Claude Code 的三层压缩系统：
  - Layer 1（MicroCompact）：只清除白名单工具的旧输出，不动对话内容
  - 工具输出在产生时最关键，几轮之后价值近乎为零
  - 替换为占位符，不删除消息结构（保持 tool_use/tool_result 配对完整）

配置（.env）：
  CONTEXT_KEEP_RECENT=10         ← Layer 1 保留最近 N 条 ToolMessage
  CONTEXT_COMPACT_THRESHOLD=120000 ← Layer 3 自动压缩阈值（估算 token 数，MiniMax 256K 模型建议 120K）
  CONTEXT_COMPACT_KEEP=6          ← Layer 3 压缩时保留最近 N 轮完整消息
"""

import os
from langchain_core.messages import ToolMessage

# ── 配置 ──
KEEP_RECENT = int(os.environ.get("CONTEXT_KEEP_RECENT", "10"))

# 可清除的工具名白名单（只有这些工具的输出可以被替换）
# 和 Claude Code 的 COMPACTABLE_TOOLS 设计一致：
# 管理状态的结构性工具永远不会被误删
COMPACTABLE_TOOLS = {
    "read_file", "list_dir", "search_file", "search_content",
    "web_search", "web_fetch", "rag_search",
    "run_command",
}

# 占位符（保持简短，Claude Code 用 "[Old tool result content cleared]"）
CLEARED_PLACEHOLDER = "[已清除]"

# Layer 3 配置
COMPACT_THRESHOLD = int(os.environ.get("CONTEXT_COMPACT_THRESHOLD", "120000"))
COMPACT_KEEP_TURNS = int(os.environ.get("CONTEXT_COMPACT_KEEP", "6"))

# ── 熔断器 ──
MAX_CONSECUTIVE_FAILURES = 3
_consecutive_compact_failures = 0

# ── Layer 3 摘要提示词（6 段结构化，对标 Claude Code 9 段模板） ──
SUMMARIZE_PROMPT = """你是一个会话摘要助手。请根据以下对话历史，生成一个精确的结构化摘要。

## 摘要格式（严格遵守）

### 当前任务
（用户正在做什么？最终目标是什么？）

### 关键文件和代码
（提到了哪些文件？每个文件的关键信息——函数名、变量名、行号）
格式：- 文件路径：关键信息
（保留关键代码片段，不超过 3 行）

### 已完成的工作
（已经解决了哪些问题？做了哪些修改？）

### 待解决的问题
（还有什么没解决的？如果有错误，保留关键错误信息）

### 重要决策
（做出了哪些重要的技术选择或设计决策？）

### 所有用户消息（原文完整保留，不可丢失）
（逐条列出用户说过的每一句话，防止多轮压缩后意图漂移）

## 要求
- 精确到文件名、函数名、行号、错误消息
- 保留错误日志的关键部分
- **用户消息必须原文保留，一条不漏**
- 总长度不超过 1000 字"""


def micro_compact(messages: list, keep_recent: int = KEEP_RECENT) -> list:
    """Layer 1: 微压缩 — 清除旧工具输出

    保留最近 N 条 ToolMessage 完整内容，更早的替换为占位符。
    不删除消息本身，保持 tool_use/tool_result 配对完整。

    Args:
        messages: 当前消息列表
        keep_recent: 保留最近多少条 ToolMessage 不清除

    Returns:
        压缩后的消息列表（可能是原列表的修改版）
    """
    if not messages or keep_recent <= 0:
        return messages

    # 1. 收集所有 ToolMessage 的索引位置
    tool_msg_indices = []
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            tool_msg_indices.append(i)

    # 没有工具输出，直接返回
    if not tool_msg_indices:
        return messages

    # 2. 如果工具输出总数 <= keep_recent，不需要清理
    if len(tool_msg_indices) <= keep_recent:
        return messages

    # 3. 找到分界线：哪些要清除，哪些保留
    # 从后往前数，保留最近 keep_recent 条
    clear_from_idx = len(tool_msg_indices) - keep_recent
    indices_to_clear = set(tool_msg_indices[:clear_from_idx])

    # 4. 执行替换（需要创建新对象，不能修改原始消息）
    result = list(messages)
    cleared_count = 0
    for i in indices_to_clear:
        msg = result[i]
        # 只清除白名单工具的输出
        # ToolMessage.name 可能在某些 LangChain 版本中不可用
        # 改为检查 tool_call_id 对应的 AIMessage.tool_calls 中的工具名
        tool_name = _get_tool_name(messages, i)
        if tool_name is None or tool_name in COMPACTABLE_TOOLS:
            old_content = getattr(msg, 'content', '')
            if old_content and len(old_content) > len(CLEARED_PLACEHOLDER):
                # 创建新的 ToolMessage 替换（保持 tool_call_id 不变）
                result[i] = ToolMessage(
                    content=CLEARED_PLACEHOLDER,
                    tool_call_id=msg.tool_call_id,
                )
                cleared_count += 1

    return result


def _get_tool_name(messages: list, tool_msg_idx: int) -> str | None:
    """根据 ToolMessage 的 tool_call_id，找到对应 AIMessage 中的工具名

    向前搜索找到包含该 tool_call_id 的 AIMessage.tool_calls
    """
    tool_call_id = getattr(messages[tool_msg_idx], 'tool_call_id', None)
    if not tool_call_id:
        return None

    # 从当前位置往前找
    for i in range(tool_msg_idx - 1, -1, -1):
        msg = messages[i]
        tool_calls = getattr(msg, 'tool_calls', None) or []
        for tc in tool_calls:
            if tc.get('id') == tool_call_id:
                return tc.get('name')
        # 找到了对应的 AIMessage 就停（不继续往前找）
        if tool_calls:
            break

    return None


def get_context_stats(messages: list) -> dict:
    """获取当前上下文统计信息（供 /context 命令使用）

    Returns:
        {
            "total_messages": 总消息数,
            "tool_messages": 工具输出消息数,
            "cleared_messages": 已被清除的消息数,
            "human_messages": 用户消息数,
            "ai_messages": AI消息数,
            "est_chars": 估算总字符数,
        }
    """
    from langchain_core.messages import HumanMessage, AIMessage

    stats = {
        "total_messages": len(messages),
        "tool_messages": 0,
        "cleared_messages": 0,
        "human_messages": 0,
        "ai_messages": 0,
        "est_chars": 0,
    }

    for msg in messages:
        content = getattr(msg, 'content', '') or ''
        stats["est_chars"] += len(content)

        if isinstance(msg, ToolMessage):
            stats["tool_messages"] += 1
            if content == CLEARED_PLACEHOLDER:
                stats["cleared_messages"] += 1
        elif isinstance(msg, HumanMessage):
            stats["human_messages"] += 1
        elif isinstance(msg, AIMessage):
            stats["ai_messages"] += 1

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3: LLM 摘要压缩
# ══════════════════════════════════════════════════════════════════════════════

def _format_messages_for_summary(messages: list) -> str:
    """将消息列表格式化为文本，供 LLM 生成摘要

    过滤掉已清除的工具输出，只保留有价值的内容。
    """
    from langchain_core.messages import HumanMessage, AIMessage

    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = getattr(msg, 'content', '') or ''
            # 去掉环境注入块，只保留用户消息部分
            if "[用户消息]" in content:
                content = content.split("[用户消息]")[-1].strip()
            if content:
                lines.append(f"[用户] {content[:200]}")
        elif isinstance(msg, AIMessage):
            content = getattr(msg, 'content', '') or ''
            tool_calls = getattr(msg, 'tool_calls', None) or []
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name", "?")
                    args = tc.get("args", {})
                    args_str = ", ".join(f"{k}={v}" for k, v in args.items())
                    lines.append(f"[AI→工具] {name}({args_str})")
            if content and content != CLEARED_PLACEHOLDER:
                lines.append(f"[AI] {content[:200]}")
        elif isinstance(msg, ToolMessage):
            content = getattr(msg, 'content', '') or ''
            if content and content != CLEARED_PLACEHOLDER:
                lines.append(f"[工具结果] {content[:300]}")

    return "\n".join(lines)


def _find_turn_boundary(messages: list, keep_turns: int) -> int:
    """找到最近 keep_turns 轮对话的分界索引

    从后往前找 keep_turns 个 HumanMessage，返回第一个的位置。
    """
    from langchain_core.messages import HumanMessage

    human_count = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            human_count += 1
            if human_count >= keep_turns:
                return i
    return 0


async def summarize_and_compact(
    messages: list,
    llm,
    keep_turns: int = COMPACT_KEEP_TURNS,
) -> tuple[list, str | None]:
    """Layer 3: LLM 摘要压缩

    将旧消息（前半部分）发送给 LLM 生成结构化摘要，
    然后用一条 HumanMessage 替换所有旧消息，保留最近 N 轮。

    Args:
        messages: 当前完整消息列表
        llm: ChatOpenAI 实例（用于生成摘要）
        keep_turns: 保留最近几轮对话不压缩

    Returns:
        (compressed_messages, summary_text)
        summary_text 为 None 表示不需要压缩
    """
    from langchain_core.messages import HumanMessage

    # 1. 找到分界线
    boundary = _find_turn_boundary(messages, keep_turns)

    # 如果旧消息太少（<3 条），不值得压缩
    if boundary < 3:
        return messages, None

    old_messages = messages[:boundary]
    recent_messages = messages[boundary:]

    # 2. 检查旧消息是否足够"有内容"
    old_text = _format_messages_for_summary(old_messages)
    if len(old_text) < 100:
        return messages, None

    # 3. 调用 LLM 生成摘要
    try:
        from langchain_core.messages import HumanMessage as HM
        # 限制输入长度，避免摘要本身超出上下文
        summary_input = old_text[:6000]
        resp = await llm.ainvoke([
            HM(content=f"{SUMMARIZE_PROMPT}\n\n---\n## 对话历史\n\n{summary_input}")
        ])
        summary = getattr(resp, 'content', '') or ''

        if not summary or len(summary) < 20:
            return messages, None

    except Exception as e:
        # 摘要失败不影响主流程
        return messages, None

    # 4. 构造压缩后的消息列表
    # 用一条 HumanMessage 携带摘要，接上最近的消息
    summary_msg = HumanMessage(
        content=f"[会话摘要 — 之前对话的压缩总结]\n\n{summary}"
    )

    compressed = [summary_msg] + list(recent_messages)
    return compressed, summary


def is_compact_circuit_open() -> bool:
    """检查熔断器是否已断开（连续失败次数达到上限）"""
    return _consecutive_compact_failures >= MAX_CONSECUTIVE_FAILURES


def record_compact_failure():
    """记录一次压缩失败"""
    global _consecutive_compact_failures
    _consecutive_compact_failures += 1


def record_compact_success():
    """记录一次压缩成功（重置计数器）"""
    global _consecutive_compact_failures
    _consecutive_compact_failures = 0


def get_circuit_status() -> dict:
    """获取熔断器状态"""
    return {
        "consecutive_failures": _consecutive_compact_failures,
        "max_failures": MAX_CONSECUTIVE_FAILURES,
        "is_open": is_compact_circuit_open(),
    }
