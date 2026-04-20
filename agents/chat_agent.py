#!/usr/bin/env python3
# chat_agent.py — My Agent 2.0.0 主入口（Textual TUI 版本）
# 架构：配置层 → 工具层 → Agent核心(Supervisor+主Agent+2子Agent) → 命令层 → 对话处理 → TUI入口
"""
技术栈：LangGraph Supervisor + Textual 8.x TUI + Rich Markdown
启动：python agents/chat_agent.py
"""
import os, sys, re, json, threading, time, asyncio, signal as _signal
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Windows 终端 ANSI 支持 ──
if sys.platform == "win32":
    import ctypes
    k = ctypes.windll.kernel32
    h = k.GetStdHandle(-11)
    m = ctypes.c_ulong()
    if k.GetConsoleMode(h, ctypes.byref(m)):
        k.SetConsoleMode(h, m.value | 0x0004)

_interrupt = threading.Event()
_signal.signal(_signal.SIGINT, lambda *_: _interrupt.set())

# ══════════════════════════════════════════════════════════════════════════════
# ① 配置层：环境变量 & 平台适配
# ══════════════════════════════════════════════════════════════════════════════
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_HTTP_TIMEOUT"] = "300"

_runtime = {
    "graph": None,
    "HM": None,
    "TM": None,
    "ck": None,
    "ck_ctx": None,
    "llm": None,
    "bind_chat": None,
    "bind_coder": None,
    "bind_planner": None,
    "ready": threading.Event(),
    "error": None,
    "model_name": None,
}

# ══════════════════════════════════════════════════════════════════════════════
# ② 轻量导入层
# ══════════════════════════════════════════════════════════════════════════════
from agents.ui.stream_parser import StreamParser
from agents.session_manager import SessionManager
from config.settings import ModelRegistry

_model_registry = ModelRegistry()
_compact_cooldown = [0]

# 流式模式下 token 计数器（MiniMax 等流式响应不返回 token_usage 时的兜底）
_stream_chars = [0]  # 当前轮次流式输出的字符总数

# ── 按会话独立的 Token 统计 ──
# 结构: {thread_id: {"tokens": int, "ctx_pct": int}}
_SESSION_STATS_FILE = Path(__file__).parent.parent / ".session_stats.json"
_session_stats: dict[str, dict] = {}  # 内存缓存，thread_id → {tokens, ctx_pct}
_active_tid: str = ""  # 当前活跃会话 ID（用于快速索引）
_mgr_ref = None  # SessionManager 引用（main() 中赋值）


def _load_session_stats():
    """从文件恢复所有会话的统计"""
    global _session_stats
    try:
        if _SESSION_STATS_FILE.exists():
            import json
            data = json.loads(_SESSION_STATS_FILE.read_text("utf-8"))
            _session_stats = {k: v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        _session_stats = {}


def _save_session_stats():
    """持久化所有会话统计到文件"""
    try:
        import json
        _SESSION_STATS_FILE.write_text(json.dumps(_session_stats, ensure_ascii=False), "utf-8")
    except Exception:
        pass


def _get_token_count(tid: str) -> int:
    """获取某会话的累计 token"""
    s = _session_stats.get(tid, {})
    return s.get("tokens", 0)


def _add_tokens(tid: str, n: int):
    """累加某会话的 token 并持久化"""
    if tid not in _session_stats:
        _session_stats[tid] = {"tokens": 0, "ctx_pct": 0}
    _session_stats[tid]["tokens"] += n
    _save_session_stats()


def _set_ctx_pct(tid: str, pct: int):
    """更新某会话的上下文百分比"""
    if tid not in _session_stats:
        _session_stats[tid] = {"tokens": 0, "ctx_pct": 0}
    _session_stats[tid]["ctx_pct"] = pct
    _save_session_stats()


def _get_ctx_pct(tid: str) -> int:
    """获取某会话的上下文百分比"""
    return _session_stats.get(tid, {}).get("ctx_pct", 0)

def _get_session_name(tid: str) -> str:
    """获取会话名称"""
    try:
        for s in _mgr_ref.list_sessions():
            if s.thread_id == tid:
                return s.name
    except Exception:
        pass
    return "default"

_AGENT_NAMES = {"chat": "Agent", "coder": "Coder", "planner": "Planner"}


def _get_env_block():
    home = str(Path.home())
    _sys = {"win32": "Windows", "darwin": "macOS", "linux": "Linux"}.get(sys.platform, sys.platform)
    return f"""## Runtime Environment
- OS: {_sys}
- Home: {home}
- CWD: {{cwd}}
- Current Time: {{current_time}}"""

ENV_BLOCK = _get_env_block()


# ══════════════════════════════════════════════════════════════════════════════
# ③ 工具层
# ══════════════════════════════════════════════════════════════════════════════
ALL_TOOLS = []
MAIN_TOOLS = []
CODER_TOOLS = []
PLANNER_TOOLS = []
_ToolMessage = None
_HumanMessage = None


def _build_tools():
    global ALL_TOOLS, MAIN_TOOLS, CODER_TOOLS, PLANNER_TOOLS
    global _ToolMessage, _HumanMessage

    from langchain_core.messages import HumanMessage, ToolMessage

    from tools.weather_tool import get_weather
    from tools.rag_tool import rag_search, add_knowledge
    from tools.time_tool import get_current_time
    from tools.search.web_search_tool import web_search
    from tools.search.web_fetch_tool import web_fetch
    from tools.file_ops import read_file, list_dir, search_file, search_content
    from tools.file_edit import write_file, edit_file
    from tools.shell_tool import run_command
    from tools.hitl_tool import ask_user, confirm, ask_questions

    _ToolMessage = ToolMessage
    _HumanMessage = HumanMessage

    try:
        from core.mcp_loader import mcp_manager
        mcp_tools = mcp_manager.tools
    except Exception:
        mcp_tools = []

    MAIN_TOOLS[:] = [get_weather, get_current_time, web_search, web_fetch,
                     rag_search, add_knowledge, read_file, list_dir,
                     search_file, search_content, write_file, edit_file, run_command,
                     ask_user, confirm, ask_questions]
    CODER_TOOLS[:] = [read_file, list_dir, search_file, search_content,
                      write_file, edit_file, run_command, ask_user]
    PLANNER_TOOLS[:] = [web_search, web_fetch, rag_search, add_knowledge, ask_user, ask_questions]
    if mcp_tools:
        for tool_list in (MAIN_TOOLS, CODER_TOOLS, PLANNER_TOOLS):
            tool_list.extend(mcp_tools)
    ALL_TOOLS[:] = MAIN_TOOLS + CODER_TOOLS + PLANNER_TOOLS


# ══════════════════════════════════════════════════════════════════════════════
# ④ Agent 核心层
# ══════════════════════════════════════════════════════════════════════════════
def _sanitize_messages(msgs):
    tc_ids = set()
    tm_ids = set()
    for m in msgs:
        for tc in (getattr(m, 'tool_calls', None) or []):
            tc_ids.add(tc.get('id'))
        tmid = getattr(m, 'tool_call_id', None)
        if tmid:
            tm_ids.add(tmid)

    result = []
    for m in msgs:
        tcs = getattr(m, 'tool_calls', None) or []
        tmid = getattr(m, 'tool_call_id', None)

        if tcs:
            valid = [tc for tc in tcs if tc.get('id') in tm_ids]
            if valid and len(valid) < len(tcs):
                try:
                    m = m.model_copy(update={'tool_calls': valid})
                except AttributeError:
                    m = m.copy(update={'tool_calls': valid})
            elif not valid:
                try:
                    m = m.model_copy(update={'tool_calls': []})
                except AttributeError:
                    m = m.copy(update={'tool_calls': []})

        if tmid and tmid not in tc_ids:
            continue
        result.append(m)
    return result


def _build_graph():
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import AIMessage, SystemMessage

    _build_tools()
    HumanMessage = _HumanMessage
    ToolMessage = _ToolMessage

    cfg = _model_registry.active_config
    llm = ChatOpenAI(
        model_name=cfg["id"],
        openai_api_key=cfg["key"],
        openai_api_base=cfg.get("base", ""),
        request_timeout=120,
    )

    def _make_sub_agent(bind_key, tools, prompt_text):
        tool_map = {t.name: t for t in tools}
        builtin_desc = []
        mcp_desc = []
        for t in tools:
            first_line = t.description.split('\n')[0] if t.description else ""
            if first_line.startswith("[MCP:"):
                mcp_desc.append(f"- {t.name}: {first_line}")
            else:
                builtin_desc.append(f"- {t.name}: {first_line}")
        tool_desc_parts = []
        if builtin_desc:
            tool_desc_parts.append("[Builtin Tools]\n" + "\n".join(builtin_desc))
        if mcp_desc:
            tool_desc_parts.append("[MCP External Tools]\n" + "\n".join(mcp_desc))
        tool_desc = "\n\n".join(tool_desc_parts)
        _runtime[bind_key] = llm.bind_tools(tools)

        class SubState(TypedDict):
            messages: Annotated[list, add_messages]

        def think(state):
            bound_llm = _runtime[bind_key]
            msgs = list(state["messages"])
            sys_block = f"{prompt_text}\n\n## Available Tools\n{tool_desc}"
            llm_input = [HumanMessage(content=sys_block)] + msgs
            resp = bound_llm.invoke(llm_input)
            return {"messages": [resp]}

        def route(state):
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            return END

        async def exec_tools(state):
            import logging
            last = state["messages"][-1]
            results = []
            _root_logger = logging.getLogger()
            _tools_logger = logging.getLogger("tools")
            _old_root = _root_logger.level
            _old_tools = _tools_logger.level
            _root_logger.setLevel(logging.CRITICAL)
            _tools_logger.setLevel(logging.CRITICAL)
            try:
                for tc in last.tool_calls:
                    t = tool_map.get(tc["name"])
                    if t:
                        try:
                            # HITL 工具不设超时——等用户操作可能很久
                            if tc["name"] in ("ask_user", "ask_questions", "confirm"):
                                r = await t.ainvoke(tc["args"])
                            else:
                                r = await asyncio.wait_for(t.ainvoke(tc["args"]), timeout=60)
                            if isinstance(r, tuple) and len(r) == 2:
                                content, _artifact = r
                                r = content
                            results.append(ToolMessage(content=str(r), tool_call_id=tc["id"]))
                        except asyncio.TimeoutError:
                            results.append(ToolMessage(content=f"Tool execution timeout (60s): {tc['name']}", tool_call_id=tc["id"]))
                        except Exception as e:
                            results.append(ToolMessage(content=f"Tool execution error: {e}", tool_call_id=tc["id"]))
                    else:
                        results.append(ToolMessage(content=f"Unknown tool: {tc['name']}", tool_call_id=tc["id"]))
            finally:
                _root_logger.setLevel(_old_root)
                _tools_logger.setLevel(_old_tools)
            return {"messages": results}

        g = StateGraph(SubState)
        g.add_node("think", think)
        g.add_node("tools", exec_tools)
        g.add_edge(START, "think")
        g.add_conditional_edges("think", route, {"tools": "tools", END: END})
        g.add_edge("tools", "think")
        return g.compile()

    MAIN_PROMPT = """You are a cheerful, witty, and versatile AI assistant. Answer responsibly — never guess unknown information.

You can search the web, fetch webpage content, check weather, get time, search knowledge base, read/write files, execute commands, and ask the user questions.
Answer most questions directly; proactively call tools when needed.

## Principles
- Answer directly when you can; don't search unnecessarily
- When user provides a URL/link, use web_fetch to read it first
- Proactively verify uncertain information via search
- Keep answers concise and useful; no fluff

## When to use ask_user / confirm / ask_questions
You MUST proactively use these tools when:
- User's request is ambiguous and has multiple possible interpretations
- You need to choose between approaches (e.g., "refactor vs rewrite", "Python vs JS")
- About to perform irreversible actions (delete files, overwrite data, run dangerous commands)
- User asks "which one", "what do you think", or any question requiring their preference
- You're unsure what the user wants — ask rather than guess wrong

Example: User says "build me a website" → call ask_user with tech stack options before starting.
Example: User says "delete that file" → call confirm before deleting.

Do NOT use ask_user for simple yes/no questions you can answer yourself."""

    CODER_PROMPT = """You are a professional coding assistant. You can read, edit, create files, and execute commands.

## Workflow
1. Use read_file/search_content to understand existing code first
2. Use edit_file for precise edits (preferred), or write_file to create new files
3. Use run_command to test and verify your changes
4. If errors occur, re-read the file and investigate

## Notes
- For large files, read the first 50 lines to understand structure before diving deeper
- edit_file's old_str must match exactly (including indentation, blank lines), or it will fail
- run_command has 30s timeout and dangerous command interception

## When to use ask_user
- Before making risky changes (delete files, rewrite large sections, deploy) → call confirm()
- When requirements are unclear → call ask_user() with 2-4 options
- When multiple implementation approaches exist → call ask_user() to let the user choose"""

    PLANNER_PROMPT = """You are a professional research & planning assistant. You can search the web, fetch webpage content, and search knowledge base.

## Principles
- Provide accurate, comprehensive information with source attribution
- When search results are insufficient, try different keywords and re-search
- Break down complex problems before tackling them step by step
- Synthesize information from multiple sources; never rely on a single source
- When you need clarification or direction from the user, use **ask_user** tool"""

    chat_agent = _make_sub_agent("bind_chat", MAIN_TOOLS, MAIN_PROMPT)
    coder_agent = _make_sub_agent("bind_coder", CODER_TOOLS, CODER_PROMPT)
    planner_agent = _make_sub_agent("bind_planner", PLANNER_TOOLS, PLANNER_PROMPT)

    from core.context_manager import micro_compact

    def _safe_add_messages(existing, new):
        combined = add_messages(existing, new)
        combined = _sanitize_messages(combined)
        combined = micro_compact(combined)
        return combined

    class AgentState(TypedDict):
        messages: Annotated[list, _safe_add_messages]
        next_agent: str

    ROUTER_PROMPT = """You are a task router. Route user messages to the appropriate expert.

Routing rules (priority: high → low):
- coder: **deep code development** — write complete features, refactor code, fix complex bugs, create multi-file projects
- planner: **deep research & analysis** — multi-round search, competitive analysis, technical research, complex information synthesis
- chat: everything else (chit-chat, Q&A, check weather, browse URLs, simple searches, ask time, single-search answers)

Important: default to chat when unsure. chat handles most cases.
Reply with a single English word only: chat or coder or planner
No explanation, no punctuation."""

    def _supervisor(state):
        current_llm = _runtime["llm"]
        last = state["messages"][-1]
        content = getattr(last, "content", "") or str(last)
        if "[user_message]" in content:
            content = content.split("[user_message]")[-1].strip()[:500]
        try:
            resp = current_llm.invoke([
                HumanMessage(content=f"{ROUTER_PROMPT}\n\nUser message: {content}"),
            ])
            agent = resp.content.strip().lower()
            for word in agent.split():
                if word in ("chat", "coder", "planner"):
                    agent = word
                    break
            else:
                agent = "chat"
        except Exception:
            agent = "chat"
        return {"next_agent": agent}

    g = StateGraph(AgentState)
    g.add_node("supervisor", _supervisor)
    g.add_node("chat", chat_agent)
    g.add_node("coder", coder_agent)
    g.add_node("planner", planner_agent)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s.get("next_agent", "chat"), {
        "chat": "chat",
        "coder": "coder",
        "planner": "planner",
    })
    g.add_edge("chat", END)
    g.add_edge("coder", END)
    g.add_edge("planner", END)

    return g, HumanMessage, ToolMessage, llm


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ 辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def _rebuild_graph_sync():
    try:
        g, HM, TM, llm = _build_graph()
        _runtime["graph"] = g
        _runtime["HM"] = HM
        _runtime["TM"] = TM
        _runtime["llm"] = llm
        _runtime["model_name"] = _model_registry.active_name
        _runtime["bind_chat"] = llm.bind_tools(MAIN_TOOLS)
        _runtime["bind_coder"] = llm.bind_tools(CODER_TOOLS)
        _runtime["bind_planner"] = llm.bind_tools(PLANNER_TOOLS)
        _runtime["error"] = None
    except Exception as e:
        _runtime["error"] = e
    finally:
        _runtime["ready"].set()


def _is_ollama() -> bool:
    cfg = _model_registry.active_config
    base = cfg.get("base", "")
    return "localhost:11434" in base or "127.0.0.1:11434" in base


_warmup_lock = threading.Lock()


def _warmup_ollama(quiet: bool = False):
    import httpx
    got = _warmup_lock.acquire(timeout=200)
    if not got:
        return
    try:
        cfg = _model_registry.active_config
        model_id = cfg.get("id", "")
        base = cfg.get("base", "")
        ollama_base = base.replace("/v1", "").rstrip("/")
        t0 = time.time()
        try:
            resp = httpx.post(
                f"{ollama_base}/api/chat",
                json={"model": model_id, "messages": [{"role": "user", "content": "hi"}],
                      "stream": False, "options": {"num_predict": 1}},
                timeout=180,
            )
        except httpx.TimeoutException:
            raise RuntimeError(f"Ollama 模型 {model_id} 加载超时")
        except Exception:
            pass
    finally:
        _warmup_lock.release()


def _hot_swap_llm():
    from langchain_openai import ChatOpenAI
    cfg = _model_registry.active_config
    new_llm = ChatOpenAI(
        model_name=cfg["id"],
        openai_api_key=cfg["key"],
        openai_api_base=cfg.get("base", ""),
        request_timeout=120,
    )
    _runtime["llm"] = new_llm
    _runtime["bind_chat"] = new_llm.bind_tools(MAIN_TOOLS)
    _runtime["bind_coder"] = new_llm.bind_tools(CODER_TOOLS)
    _runtime["bind_planner"] = new_llm.bind_tools(PLANNER_TOOLS)
    _runtime["model_name"] = _model_registry.active_name


async def _auto_compact(thread_id, *, force=False, app=None):
    """自动压缩上下文（app 参数用于 TUI 显示）"""
    from core.context_manager import (
        get_context_stats, COMPACT_THRESHOLD,
        is_compact_circuit_open, record_compact_failure, record_compact_success,
    )
    from langgraph.graph.message import add_messages

    if not force and is_compact_circuit_open():
        return False
    if not force:
        _compact_cooldown[0] -= 1
        if _compact_cooldown[0] > 0:
            return False

    try:
        graph = _runtime["graph"]
        if not hasattr(graph, 'aget_state'):
            return False
        cfg = {"configurable": {"thread_id": thread_id}}
        state = await graph.aget_state(cfg)
        msgs = state.values.get("messages", [])
        if not msgs:
            return False

        stats = get_context_stats(msgs)
        est_tokens = stats["est_chars"] // 2

        if not force and est_tokens < COMPACT_THRESHOLD:
            return False

        if app:
            app.add_system_message(f"⏳ 上下文 ~{est_tokens:,} tokens，正在压缩...", "info")
            app.start_spinner("Compacting")

        from core.context_manager import summarize_and_compact
        llm = _runtime.get("llm")
        if not llm:
            if app: app.stop_spinner()
            return False

        compressed, summary = await summarize_and_compact(msgs, llm)

        if app: app.stop_spinner()

        if summary is None:
            record_compact_failure()
            return False

        from langgraph.graph.message import RemoveMessage
        remove_ops = [RemoveMessage(id=m.id) for m in msgs if getattr(m, 'id', None)]
        try:
            await graph.aupdate_state(cfg, {"messages": remove_ops})
        except Exception:
            for op in remove_ops:
                try:
                    await graph.aupdate_state(cfg, {"messages": [op]})
                except Exception:
                    pass
        await graph.aupdate_state(cfg, {"messages": compressed})

        try:
            from core.session_memory import session_memory as _sm
            file_context = _sm.get_recent_file_context(thread_id)
            if file_context:
                from langchain_core.messages import HumanMessage as _HM
                ctx_msg = _HM(content=f"[压缩恢复 — 近期文件上下文]\n\n{file_context}")
                await graph.aupdate_state(cfg, {"messages": [ctx_msg]})
        except Exception:
            pass

        new_stats = get_context_stats(compressed)
        new_tokens = new_stats["est_chars"] // 2
        saved_pct = (1 - new_stats["est_chars"] / max(stats["est_chars"], 1)) * 100

        record_compact_success()
        _compact_cooldown[0] = 3
        if app:
            app.add_system_message(
                f"✅ 压缩完成：{stats['total_messages']} → {new_stats['total_messages']} 条消息 "
                f"| ~{est_tokens:,} → ~{new_tokens:,} tokens（节省 {saved_pct:.0f}%）",
                "success"
            )
        return True

    except Exception as e:
        if app: app.stop_spinner()
        record_compact_failure()
        if app:
            app.add_system_message(f"⚠ 压缩失败: {e}", "warning")
        return False


async def _track_memory(thread_id: str):
    try:
        from core.session_memory import session_memory
        cfg = {"configurable": {"thread_id": thread_id}}
        state = await _runtime["graph"].aget_state(cfg)
        msgs = state.values.get("messages", [])
        session_memory.track_from_messages(thread_id, msgs[-8:])
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# ⑥ TUI 版 — 命令处理（输出到 Textual 消息区）
# ══════════════════════════════════════════════════════════════════════════════

async def _tui_cmd(cmd: str, mgr: SessionManager, app) -> bool:
    """TUI 版命令处理——输出到 app 消息区而非 print"""
    p = cmd.strip().split(maxsplit=1)
    a, arg = p[0].lower(), (p[1].strip() if len(p) > 1 else "")

    if a in ("/quit", "/q"):
        return True

    if a in ("/sessions", "/ls"):
        sessions = mgr.list_sessions()
        if len(sessions) <= 1:
            app.add_system_message("Only one session, nothing to switch.", "info")
            return False

        # Build options for SelectorModal
        from agents.ui.tui_app import SelectorModal, Text
        options = []
        for i, s in enumerate(sessions):
            is_active = s.thread_id == mgr.active_id
            prefix = "→ " if is_active else "  "
            color = "green" if is_active else "cyan"
            label = Text.from_markup(
                f"[{color}]{prefix}{i+1}. {s.name}[/]  [dim]({s.thread_id[:12]})[/dim]"
            )
            options.append({
                "id": str(i + 1),  # 1-based, matches mgr.switch()
                "label": label,
                "disabled": False,
            })

        modal = SelectorModal(
            title="Select Session",
            options=options,
            option_id_prefix="sess-",
        )

        def on_session_choice(value: str | None):
            if value is None:
                return  # Esc cancelled
            idx = int(value)
            s = mgr.switch(idx)
            global _active_tid
            _active_tid = s.thread_id
            app.update_status(
                tokens=_get_token_count(s.thread_id),
                ctx_pct=_get_ctx_pct(s.thread_id),
                session=s.name,
            )

        app.push_screen(modal, on_session_choice)
        return False

    if a == "/rename":
        if not arg:
            app.add_system_message("用法: /rename <新名称>", "warning")
            return False
        mgr.rename(arg)
        app.update_status(session=_get_session_name(mgr.active_id))
        return False

    if a == "/delete":
        if not arg:
            app.add_system_message("用法: /delete <序号>", "warning")
            return False
        try:
            tid = mgr.delete(int(arg.strip()))
            # 清理该会话的统计
            _session_stats.pop(tid, None)
            _save_session_stats()
            app.add_system_message(f"🗑️ 已删除会话: {tid}", "success")
        except ValueError as e:
            app.add_system_message(f"✗ {e}", "error")
        return False

    if a == "/models":
        models = _model_registry.list_all()
        active = _model_registry.active_name
        if not arg:
            # 显示交互式模型选择器
            app.show_model_selector(models, active)
            return False
        # 切换模型
        try:
            old, new = _model_registry.switch(arg)
            cfg = _model_registry.active_config
            _hot_swap_llm()
            app.add_system_message(
                f"✅ 模型已切换: [yellow]{old}[/] → [green]{new}[/] ({cfg['id']})", "success"
            )
            app.update_status(model=new)
            if _is_ollama():
                threading.Thread(target=_warmup_ollama, kwargs={"quiet": True}, daemon=True).start()
        except ValueError as e:
            app.add_system_message(f"✗ {e}", "error")
        return False

    if a == "/mcp":
        if not arg or arg == "status":
            try:
                from core.mcp_loader import mcp_manager
                app.add_system_message(mcp_manager.get_status(), "info")
            except Exception as e:
                app.add_system_message(f"MCP 未配置: {e}", "warning")
            return False
        if arg == "reload":
            app.start_spinner("MCP Reloading")
            try:
                from core.mcp_loader import mcp_manager
                await mcp_manager.reload()
                app.stop_spinner()
                if mcp_manager.is_active:
                    app.add_system_message(f"✅ MCP 已重载: {len(mcp_manager.tools)} 个工具", "success")
                    # 重建图
                    _rebuild_graph_sync()
                    if _runtime["ck_ctx"] is not None:
                        _runtime["graph"] = _runtime["graph"].compile(checkpointer=_runtime["ck"])
                else:
                    app.add_system_message("⚠ MCP 无活跃连接", "warning")
            except Exception as e:
                app.stop_spinner()
                app.add_system_message(f"✗ MCP 重载失败: {e}", "error")
            return False
        app.add_system_message("用法: /mcp [status|reload]", "info")
        return False

    if a == "/context":
        try:
            from core.context_manager import get_context_stats
            ck = _runtime.get("ck")
            if ck is None:
                app.add_system_message("上下文尚未初始化（发一条消息后可用）", "info")
                return False
            tid = mgr.active_id
            cfg_ctx = {"configurable": {"thread_id": tid}}
            state = await _runtime["graph"].aget_state(cfg_ctx)
            msgs = state.values.get("messages", [])
            stats = get_context_stats(msgs)
            est_tokens = stats['est_chars'] // 2
            lines = [
                "[bold cyan]上下文状态:[/bold cyan]",
                f"  消息总数:    {stats['total_messages']}",
                f"  ├─ User messages: {stats['human_messages']}",
                f"  ├─ AI 消息:  {stats['ai_messages']}",
                f"  └─ 工具输出: {stats['tool_messages']}（已压缩 {stats['cleared_messages']}）",
                f"  估算 token:  ~{est_tokens:,}",
            ]
            app.add_system_message("\n".join(lines), "info")
        except Exception as e:
            app.add_system_message(f"获取上下文失败: {e}", "warning")
        return False

    if a == "/compact":
        compressed = await _auto_compact(mgr.active_id, force=True, app=app)
        if not compressed:
            app.add_system_message("压缩失败或无需压缩", "warning")
        return False

    app.add_system_message(f"未知命令: {a}", "error")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# ⑦ TUI 版 — 核心对话处理（流式输出到 Textual Markdown）
# ══════════════════════════════════════════════════════════════════════════════

async def _tui_turn(user_input: str, thread_id: str, app):
    """TUI 版单轮对话：流式输出到 Textual Markdown 组件"""
    parser = StreamParser()
    reply_buffer = []  # 收集非思考内容的完整回复
    thinking_buffer = []

    # 注入实时时间和工作目录
    from datetime import datetime
    _now = datetime.now().strftime('%Y-%m-%d %A %H:%M:%S')
    _cwd = os.environ.get("LAUNCH_DIR", "") or str(Path.cwd())
    _env = ENV_BLOCK.replace("{current_time}", _now).replace("{cwd}", _cwd)

    # 注入会话记忆
    from core.session_memory import session_memory
    memory_block = session_memory.get_memory_block(thread_id)
    parts = [_env]
    if memory_block:
        parts.append(memory_block)
    parts.append(f"[user_message]\n{user_input}")

    cfg = {"configurable": {"thread_id": thread_id}}

    # 标记当前活跃会话（用于 token/ctx 按会话独立统计）
    global _active_tid
    _active_tid = thread_id
    # 切换到该会话时，立即刷新状态栏为该会话的统计数据
    app.update_status(
        tokens=_get_token_count(thread_id),
        ctx_pct=_get_ctx_pct(thread_id),
        session=_get_session_name(thread_id),
    )

    try:
        # 显示 Agent 标签 + 启动 Thinking Spinner
        app.add_agent_label()
        app.start_spinner("Thinking")

        # 等待后台加载完成
        if not _runtime["ready"].is_set():
            await asyncio.to_thread(_runtime["ready"].wait)
            if _runtime["error"]:
                app.stop_spinner()
                app.add_system_message(f"启动失败: {_runtime['error']}", "error")
                return

        # 首次编译 checkpointer
        if _runtime["ck_ctx"] is None:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            db = Path(__file__).parent.parent / "chat_history.db"
            ck_ctx = AsyncSqliteSaver.from_conn_string(str(db))
            _runtime["ck"] = await ck_ctx.__aenter__()
            _runtime["ck_ctx"] = ck_ctx
            _runtime["graph"] = _runtime["graph"].compile(checkpointer=_runtime["ck"])

        graph = _runtime["graph"]
        HumanMessage = _runtime["HM"]

        # MCP 懒加载
        if not _runtime.get("mcp_loaded"):
            try:
                from core.mcp_loader import mcp_manager
                if not mcp_manager.is_active:
                    await mcp_manager.load()
                if mcp_manager.is_active:
                    _build_tools()
                    _rebuild_graph_sync()
                    if _runtime["ck_ctx"] is not None:
                        _runtime["graph"] = _runtime["graph"].compile(checkpointer=_runtime["ck"])
                    graph = _runtime["graph"]
            except Exception:
                pass
            finally:
                _runtime["mcp_loaded"] = True

        # Ollama 预热
        if _is_ollama():
            try:
                await asyncio.to_thread(_warmup_ollama, quiet=True)
            except RuntimeError:
                app.stop_spinner()
                app.add_system_message("✗ 模型加载失败，请检查 Ollama 状态", "error")
                return

        # 构造输入
        inp = {
            "messages": [HumanMessage(content="\n\n".join(parts))],
            "next_agent": "",
        }

        _phase = "supervisor"
        _stream_started = False  # 是否已开始 Markdown 流式输出
        _in_thinking = False

        async for ev in graph.astream_events(inp, config=cfg, version="v2"):
            if _interrupt.is_set():
                raise KeyboardInterrupt("打断")

            et = ev.get("event", "")
            node = ev.get("metadata", {}).get("langgraph_node", "")

            # LLM 开始
            if et == "on_chat_model_start":
                _in_thinking = False  # 每次新 LLM 调用重置，确保思考内容不跨轮次累积
                if node == "supervisor":
                    _phase = "supervisor"
                else:
                    _phase = "agent"
                    ckpt_ns = ev.get("metadata", {}).get("checkpoint_ns", "")
                    sub_name = ckpt_ns.split(":")[0] if ckpt_ns else node
                    agent_label = _AGENT_NAMES.get(sub_name, sub_name.title())
                    if sub_name != "chat":
                        app.stop_spinner()
                        app.add_agent_label(sub_label=agent_label)
                        app.start_spinner("Thinking")

            # LLM 结束（提取 token）
            elif et == "on_chat_model_end":
                if _phase == "supervisor":
                    continue
                try:
                    out = ev.get("data", {}).get("output", {})

                    total = 0
                    meta = {}
                    if hasattr(out, "response_metadata"):
                        meta = out.response_metadata or {}

                    # ── 策略 1: 从 response_metadata.token_usage 提取（非流式/部分模型）──
                    usage = meta.get("token_usage", meta.get("usage", {}))
                    if isinstance(usage, dict):
                        total = usage.get("total_tokens", 0)
                        if not total:
                            total = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                            if not total and "input_tokens" in usage:
                                total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

                    # ── 策略 2: 从 usage_metadata 提取（LangChain 标准字段）──
                    if not total:
                        am_meta = getattr(out, "usage_metadata", None) or {}
                        if isinstance(am_meta, dict) and "total_tokens" in am_meta:
                            total = am_meta.get("total_tokens", 0)
                        elif isinstance(am_meta, dict) and "input_tokens" in am_meta:
                            total = am_meta.get("input_tokens", 0) + am_meta.get("output_tokens", 0)

                    # ── 策略 3: 流式字符估算兜底（MiniMax/Gemini 等流式不返回用量）──
                    if not total and _stream_chars[0] > 0:
                        # 基于 MiniMax M2.7 实测数据（2026-04-20）：
                        #   纯中文短回复:   ~2.1-2.3 chars/token
                        #   中长中文回复:   ~2.4-2.8 chars/token（含思考标签时更高）
                        #   长回复(>1000字):~3.5-4.0 chars/token
                        # 取 2.5 作为通用折中值，覆盖大多数场景
                        total = max(int(_stream_chars[0] / 2.5), 1)

                    if total:
                        _add_tokens(_active_tid, total)
                        app.update_status(tokens=_get_token_count(_active_tid))

                    # 重置本轮流式计数器
                    _stream_chars[0] = 0
                except Exception:
                    pass

            # LLM 流式输出
            elif et == "on_chat_model_stream":
                if _phase == "supervisor":
                    continue
                chunk = ev["data"]["chunk"]
                content = getattr(chunk, "content", "") or ""
                if not content:
                    continue

                # 累计流式输出字符数（用于 token 估算兜底）
                _stream_chars[0] += len(content)

                _t0 = time.time()
                _parsed_count = 0  # 本 chunk 被 parser 分成几段输出
                _total_text_len = 0
                for is_t, text in parser.feed(content):
                    _parsed_count += 1
                    _total_text_len += len(text)
                    if is_t:
                        # 思考内容 — 累积到 ThinkingWidget
                        if not _in_thinking:
                            _in_thinking = True
                            app.stop_spinner()
                            app.start_thinking()
                        await app.stream_thinking(text)
                    else:
                        # 正式回复内容
                        if _in_thinking:
                            _in_thinking = False
                        if not _stream_started:
                            # 首次收到正式回复 → 停 spinner，启动 Markdown 流式
                            app.stop_spinner()
                            app.start_agent_reply()
                            _stream_started = True
                        reply_buffer.append(text)
                        await app.stream_text(text)

                _dt = time.time() - _t0

            # 工具调用开始
            elif et == "on_tool_start":
                nm = ev.get("name", "unknown")
                inp2 = ev.get("data", {}).get("input", {})

                # ── 参数过滤：隐藏列表型/冗余参数，长值截断 ──
                _HIDE_PARAMS = {"options", "allow_custom_input", "tools"}
                parts = []
                for k, v in inp2.items():
                    if k in _HIDE_PARAMS:
                        continue
                    if isinstance(v, list):
                        continue  # 跳过所有列表型参数（options/files 等）
                    if isinstance(v, str) and len(v) > 60:
                        v = v[:57] + "..."
                    parts.append(f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}')
                params = "，".join(parts) if parts else ""
                
                info = f"{nm}（{params}）" if params else nm
                app.stop_spinner()
                app.start_spinner("ToolCalling", info)

            # 工具调用结束
            elif et == "on_tool_end":
                app.stop_spinner()
                # 关闭旧的 Markdown 流，下一轮正文会创建新的
                if _stream_started:
                    await app.finish_stream()
                    _stream_started = False
                _in_thinking = False
                parser = StreamParser()

    except (KeyboardInterrupt, asyncio.CancelledError):
        app.stop_spinner()
        _interrupt.clear()
        return
    except Exception as e:
        app.stop_spinner()
        app.add_system_message(f"错误: {e}", "error")

    # 刷出剩余内容
    for is_t, text in parser.done():
        if is_t:
            thinking_buffer.append(text)
        else:
            if not _stream_started:
                app.stop_spinner()
                app.start_agent_reply()
                _stream_started = True
            reply_buffer.append(text)
            await app.stream_text(text)

    # 结束流式写入
    if _stream_started:
        await app.finish_stream()
    else:
        app.stop_spinner()
        if not reply_buffer and not thinking_buffer:
            app.add_system_message("(无回复)", "info")


# ══════════════════════════════════════════════════════════════════════════════
# ⑧ TUI 入口：启动 Textual App
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """启动 TUI 版 My Agent"""
    from agents.ui.tui_app import AgentTUI, InteractiveSelector

    # 创建会话管理器
    global _mgr_ref
    mgr = SessionManager()
    _mgr_ref = mgr
    if not mgr.list_sessions():
        mgr.create("default")

    # 后台加载重型模块
    _runtime["ready"].clear()

    def _background_load():
        try:
            g, HM, TM, llm = _build_graph()
            _runtime["graph"] = g
            _runtime["HM"] = HM
            _runtime["TM"] = TM
            _runtime["llm"] = llm
            _runtime["model_name"] = _model_registry.active_name
        except Exception as e:
            _runtime["error"] = e
        finally:
            _runtime["ready"].set()

    threading.Thread(target=_background_load, daemon=True).start()

    # ── 恢复所有会话的统计（按会话独立）──
    _load_session_stats()

    # ── 创建 TUI App ──
    class MyAgentTUI(AgentTUI):
        """注入业务逻辑的 TUI App"""

        def on_mount(self):
            super().on_mount()
            mn = _model_registry.active_name
            mid = _model_registry.active_config.get("id", "?")
            cur = mgr.active_session
            if cur:
                session_label = cur.name
            else:
                session_label = "default"

            self.update_status(
                model=f"{mn} ({mid})",
                tokens=0,
                ctx_pct=0,
                session=session_label,
            )
            # 不显示欢迎信息

            # ── 初始化 HITL Bridge 的事件循环 ──
            import asyncio
            try:
                from tools.hitl_tool import get_hitl_bridge
                _hb = get_hitl_bridge()
                _hb.set_event_loop(asyncio.get_running_loop())
            except RuntimeError as e:
                self.add_system_message(f"[HITL] 警告: 无法获取事件循环 — {e}", "warning")

        async def on_user_input(self, text: str):
            """User message callback"""
            # 重置本轮流式字符计数
            _stream_chars[0] = 0
            tid = mgr.active_id
            await _auto_compact(tid, app=self)
            await _tui_turn(text, tid, self)
            mgr.touch()
            await _track_memory(tid)

            # 更新上下文进度
            try:
                from core.context_manager import get_context_stats
                if _runtime["graph"] and hasattr(_runtime["graph"], 'aget_state'):
                    cfg = {"configurable": {"thread_id": tid}}
                    state = await _runtime["graph"].aget_state(cfg)
                    msgs = state.values.get("messages", [])
                    stats = get_context_stats(msgs)
                    est_tokens = stats["est_chars"] // 2
                    # 假设模型上限 128K，计算百分比
                    ctx_pct = min(int(est_tokens / 128000 * 100), 100)
                    _set_ctx_pct(tid, ctx_pct)
                    self.update_status(tokens=_get_token_count(tid), ctx_pct=ctx_pct)
            except Exception:
                pass

        async def on_command(self, cmd: str) -> bool:
            """命令回调"""
            return await _tui_cmd(cmd, mgr, self)

        async def on_model_switch(self, name: str):
            """模型切换回调"""
            try:
                old, new = _model_registry.switch(name)
                cfg = _model_registry.active_config
                self.update_status(model=f"{new} ({cfg['id']})")
                _hot_swap_llm()
                if _is_ollama():
                    threading.Thread(target=_warmup_ollama, kwargs={"quiet": True}, daemon=True).start()
            except ValueError as e:
                self.add_system_message(f"✗ {e}", "error")
            except Exception as e:
                self.add_system_message(f"✗ 切换后处理异常: {e}", "error")

    app = MyAgentTUI()

    # ─── Initialize Human-in-the-Loop Bridge ───
    from tools.hitl_tool import get_hitl_bridge
    hitl_bridge = get_hitl_bridge()

    def _show_hitl_selector(request: dict):
        """Callback: HITL tool asks a question → show InteractiveSelector."""
        question = request["question"]
        options = request["options"]
        allow_input = request.get("allow_input", True)
        placeholder = request.get("placeholder", "Type your choice...")

        from agents.ui.tui_app import InteractiveSelector
        from rich.text import Text as RText
        opt_items = []
        for i, opt_text in enumerate(options):
            label = RText.from_markup(f"[cyan]  {i+1}. {opt_text}[/]")
            opt_items.append({"id": str(i), "label": label})

        # 没有选项且不允许自定义输入 → 无意义的调用，直接返回错误
        if not opt_items and not allow_input:
            bridge = get_hitl_bridge()
            bridge.resolve("[Error] Empty selector: no options and allow_input=False. Please provide options or allow custom input.")
            return

        sel = InteractiveSelector(
            title=question,
            options=opt_items,
            allow_input=allow_input,
            placeholder=placeholder,
            option_id_prefix="hitl-",
        )
        app._current_selector_ctx = "hitl"

        def _hitl_choice(value: str):
            idx = int(value)
            chosen = options[idx] if idx < len(options) else value
            hitl_bridge.resolve(chosen)

        def _hitl_cancelled():
            hitl_bridge.cancel()

        def _hitl_custom(text: str):
            hitl_bridge.resolve(text)

        app.on_selector_choice = _hitl_choice
        app.on_selector_cancelled = _hitl_cancelled
        app.on_selector_custom_input = _hitl_custom
        app.show_selector(sel)

    hitl_bridge.set_callbacks(_show_hitl_selector)

    # ── Survey callback (multi-question HITL) ─
    def _show_hitl_survey(request: dict):
        """Callback: ask_questions tool → show MultiQuestionSurvey."""
        title = request["title"]
        questions = request["questions"]

        from agents.ui.tui_app import MultiQuestionSurvey

        survey = MultiQuestionSurvey(
            title=title,
            questions=questions,
            option_id_prefix="hitl-survey-",
        )
        app._current_selector_ctx = "survey"

        def _survey_submitted(results: dict):
            hitl_bridge.resolve_survey(results)

        def _survey_cancelled():
            hitl_bridge.cancel()

        app.on_survey_submitted = _survey_submitted
        app.on_survey_cancelled = _survey_cancelled
        app.show_survey(survey)

    hitl_bridge.set_survey_callback(_show_hitl_survey)

    app.run()


if __name__ == "__main__":
    main()
