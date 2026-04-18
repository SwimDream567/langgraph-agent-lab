# chat_agent.py — My Agent 1.0.0：入口文件
# 架构：配置层 → 工具层 → Agent核心 → 命令层 → 对话循环 → 入口
"""
技术栈：LangGraph + LangChain + ChromaDB + BM25 + ChatOpenAI + asyncio
设计：SessionManager(JSON索引) + AsyncSqliteSaver(历史持久化) + 斜杠命令
模块：agents/session_manager.py | agents/ui/{spinner,stream_parser,display}.py
启动优化：重型导入(langchain/langgraph)延迟到 main() 内，Banner 先行显示
"""
import os, sys, re, threading, time, asyncio, signal as _signal
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))

# ══════════════════════════════════════════════════════════════════════════════
# ① 配置层：环境变量 & 平台适配
# ══════════════════════════════════════════════════════════════════════════════
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_HTTP_TIMEOUT"] = "300"

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
# ② 轻量导入层：UI & 会话管理（秒级）
# ══════════════════════════════════════════════════════════════════════════════
from agents.ui.colors import A as _A
from agents.ui.spinner import start_spinner, stop_spinner
from agents.ui.stream_parser import StreamParser
from agents.ui.display import type_text, render_session_list, select_session
from agents.session_manager import SessionManager
from config.settings import ModelRegistry

_model_registry = ModelRegistry()
_llm_with_tools_holder = [None]  # 可变引用，支持运行时切换模型
_total_tokens = [0]              # 累计 token 消耗

def _get_system_prompt():
    """生成系统提示词（cwd/time 在 _turn 里动态替换）"""
    home = str(Path.home())
    desktop = str(Path.home() / "Desktop")
    downloads = str(Path.home() / "Downloads")
    documents = str(Path.home() / "Documents")
    user_name = os.environ.get("USERNAME", os.environ.get("USER", "unknown"))
    return f"""你是活泼开朗、风趣幽默的AI助手。回答认真负责，不猜测未知内容。

## 运行环境

- 用户：{user_name}
- 系统：Windows
- 主目录：{home}
- 桌面：{desktop}
- 下载：{downloads}
- 文档：{documents}
- 当前工作目录：{{cwd}}
- 当前时间：{{current_time}}

当用户说"桌面上的文件"时，你知道指的是 {desktop}。
当用户说"看看当前目录"时，你知道指的是 {{cwd}}。
当用户给相对路径时，基于 {{cwd}} 解析。

## 工具使用指南

### 文件读取（安全，随时可用）
- `read_file(path, offset, limit)` — 读取文件内容
- `list_dir(path, pattern, recursive)` — 列出目录内容
- `search_file(path, pattern)` — 按文件名搜索
- `search_content(path, pattern, file_glob, context)` — 搜索文件内容

### 文件编辑（需谨慎）
- `write_file(path, content)` — 创建或覆盖文件（自动创建父目录）
- `edit_file(path, old_str, new_str)` — 精确替换文件中的文本（old_str 必须唯一匹配）

### 命令执行（需谨慎）
- `run_command(command, cwd, timeout)` — 执行 Shell 命令，返回输出

### 工作流程
1. 先用 read_file/search_content 了解现有代码
2. 用 edit_file 做精确修改（优先），或 write_file 创建新文件
3. 用 run_command 运行测试验证修改效果
4. 如果出错了，再次读取文件排查

### 注意事项
- 读取大文件时先读前 50 行了解结构，再按需读取关键部分
- edit_file 的 old_str 必须精确匹配（包括缩进、空行），否则会失败
- run_command 有 30 秒超时和危险命令拦截
"""

SYSTEM_PROMPT = _get_system_prompt()

# ══════════════════════════════════════════════════════════════════════════════
# ③ 工具层：LangChain @tool 封装（重型导入延迟到 main）
# ══════════════════════════════════════════════════════════════════════════════
# 预声明：tools 在 _build_graph() 里被真正赋值
ALL_TOOLS = []
_ToolMessage = None
_HumanMessage = None


def _build_tools():
    """延迟构建工具列表（langchain_core/langchain_openai 很慢）"""
    global ALL_TOOLS, _ToolMessage, _HumanMessage

    from langchain_core.messages import HumanMessage, ToolMessage
    from langchain_openai import ChatOpenAI

    # 全部从各工具模块导入（@tool 已在模块内定义）
    from tools.weather_tool import get_weather
    from tools.rag_tool import rag_search, add_knowledge
    from tools.time_tool import get_current_time
    from tools.search.web_search_tool import web_search
    from tools.search.web_fetch_tool import web_fetch
    from tools.file_ops import read_file, list_dir, search_file, search_content
    from tools.file_edit import write_file, edit_file
    from tools.shell_tool import run_command

    _ToolMessage = ToolMessage
    _HumanMessage = HumanMessage

    ALL_TOOLS[:] = [get_weather, rag_search, add_knowledge, get_current_time,
                     web_search, web_fetch,
                     read_file, list_dir, search_file, search_content,
                     write_file, edit_file, run_command]

    # 用注册表创建 LLM（支持运行时切换）
    cfg = _model_registry.active_config
    llm = ChatOpenAI(model_name=cfg["id"], openai_api_key=cfg["key"],
                     openai_api_base=cfg.get("base", ""))
    _llm_with_tools_holder[0] = llm.bind_tools(ALL_TOOLS)
    return llm


# ══════════════════════════════════════════════════════════════════════════════
# ④ Agent 核心层：LangGraph 状态图（延迟构建）
# ══════════════════════════════════════════════════════════════════════════════
def _build_graph():
    """延迟构建 LangGraph 状态图（含所有重型导入）"""
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    # 确保工具层已加载
    llm = _build_tools()

    ToolMessage = _ToolMessage
    HumanMessage = _HumanMessage

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    def _sanitize_messages(msgs: list) -> list:
        from langchain_core.messages import SystemMessage as _SM
        clean = [m for m in msgs if not isinstance(m, _SM)]
        result = []
        for m in clean:
            if result and hasattr(result[-1], "tool_calls") and result[-1].tool_calls:
                if isinstance(m, ToolMessage):
                    result.append(m)
                else:
                    for tc in result[-1].tool_calls:
                        result.append(ToolMessage(content="[工具调用被中断，无结果]", tool_call_id=tc["id"]))
                    result.append(m)
            else:
                if isinstance(m, ToolMessage):
                    has_match = False
                    for prev in result:
                        if hasattr(prev, "tool_calls") and prev.tool_calls:
                            if any(tc["id"] == m.tool_call_id for tc in prev.tool_calls):
                                has_match = True
                                break
                    if not has_match:
                        continue
                result.append(m)
        return result

    def _parse_retry_wait(err_msg: str) -> float:
        for pat in [
            r'(?:retry\s*after|wait|等待)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|second|秒)',
            r'rate\s*limit.*?(\d+(?:\.\d+)?)\s*(?:s|sec|second|秒)',
            r'(\d+(?:\.\d+)?)\s*(?:s|sec|second|秒).*(?:retry|重试)',
            r'retry.*?(\d+(?:\.\d+)?)\s*(?:ms|msec|millisecond|毫秒)',
        ]:
            match = re.search(pat, err_msg, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                if 'ms' in match.group(0).lower() or '毫秒' in match.group(0):
                    return val / 1000
                return val
        return None

    MAX_LLM_RETRIES = 10

    def _think(state: AgentState) -> dict:
        safe_msgs = _sanitize_messages(state["messages"])
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            try:
                return {"messages": [_llm_with_tools_holder[0].invoke(safe_msgs)]}
            except Exception as e:
                is_last = attempt == MAX_LLM_RETRIES
                err_msg = str(e)
                if is_last:
                    raise
                wait = _parse_retry_wait(err_msg) or min(2 ** attempt, 60)
                print(f"\n  {_A['y']}⚠ LLM 第{attempt}次调用失败，{wait:.1f}s 后重试: {_A['d']}{err_msg[:120]}{_A['0']}")
                time.sleep(wait)

    def _exec_tool(state: AgentState) -> dict:
        last = state["messages"][-1]
        tmap = {t.name: t for t in ALL_TOOLS}
        results = []
        for tc in last.tool_calls:
            t = tmap.get(tc["name"])
            if not t:
                results.append(ToolMessage(content=f"[工具错误] 未知工具: {tc['name']}", tool_call_id=tc["id"]))
                continue
            try:
                results.append(ToolMessage(content=str(t.invoke(tc["args"])), tool_call_id=tc["id"]))
            except Exception as e:
                results.append(ToolMessage(content=f"[工具错误] {tc['name']} 执行失败: {e}", tool_call_id=tc["id"]))
        return {"messages": results}

    def _route(state: AgentState) -> str:
        return "use_tool" if getattr(state["messages"][-1], "tool_calls", None) else "done"

    g = StateGraph(AgentState)
    g.add_node("think", _think)
    g.add_node("exec_tool", _exec_tool)
    g.add_edge(START, "think")
    g.add_conditional_edges("think", _route, {"use_tool": "exec_tool", "done": END})
    g.add_edge("exec_tool", "think")

    return g, HumanMessage, ToolMessage


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ 命令层：斜杠命令处理器
# ══════════════════════════════════════════════════════════════════════════════
async def _cmd(cmd: str, graph, mgr: SessionManager) -> bool:
    """斜杠命令分发：/help /sessions /new /switch /rename /delete /quit"""
    p = cmd.strip().split(maxsplit=1)
    a, arg = p[0].lower(), (p[1].strip() if len(p) > 1 else "")

    if a in ("/help", "/?"):
        print(f"\n{_A['b']}{_A['c']}可用命令:{_A['0']}")
        print(f"  {_A['g']}/sessions{_A['0']} / {_A['g']}/ls{_A['0']}          列出所有会话")
        print(f"  {_A['g']}/new [名称]{_A['0']}              新建并切换到新会话")
        print(f"  {_A['g']}/switch <序号|名称>{_A['0']}       切换会话（支持 {_A['g']}/s{_A['0']}）")
        print(f"  {_A['g']}/rename <名称>{_A['0']}             重命名当前会话")
        print(f"  {_A['g']}/delete <序号>{_A['0']}           删除指定会话")
        print(f"  {_A['g']}/models{_A['0']}                  查看已配置模型列表")
        print(f"  {_A['g']}/models <名称|序号>{_A['0']}       切换模型（支持前缀匹配）")
        print(f"  {_A['g']}/quit{_A['0']} / {_A['g']}/q{_A['0']}             退出程序\n")
        return False

    if a in ("/quit", "/q"): return True
    if a in ("/sessions", "/ls"): render_session_list(mgr); return False

    if a == "/new":
        s = mgr.create(arg or None)
        print(f"\n  ✅ 新建会话: {s.name} ({s.thread_id})")
        return False

    if a in ("/switch", "/s"):
        if not arg:
            sel = await select_session(mgr)
            if sel: mgr.switch_by_name(sel.name); print(f"\n  ✅ 已切换到: {sel.name} ({sel.thread_id})")
            return False
        try: s = mgr.switch(int(arg.strip())); print(f"\n  ✅ 已切换到: {s.name} ({s.thread_id})")
        except ValueError:
            try: s = mgr.switch_by_name(arg); print(f"\n  ✅ 已切换到: {s.name} ({s.thread_id})")
            except ValueError as e: print(f"\n  ✗ {e}")
        return False

    if a == "/rename":
        if not arg: print(f"\n  {_A['y']}用法: /rename <新名称>{_A['0']}"); return False
        old = mgr.active_session.name if mgr.active_session else "默认"
        mgr.rename(arg); print(f"\n  ✅ 重命名: '{old}' → '{arg}'"); return False

    if a == "/delete":
        if not arg: print(f"\n  {_A['y']}用法: /delete <序号>{_A['0']}"); render_session_list(mgr); return False
        try: tid = mgr.delete(int(arg.strip())); print(f"\n  🗑️ 已删除会话: {tid}")
        except ValueError as e: print(f"\n  ✗ {e}")
        return False

    if a == "/models":
        if not arg:
            # 列出所有模型
            models = _model_registry.list_all()
            active = _model_registry.active_name
            print(f"\n  {_A['b']}{_A['c']}已配置的模型:{_A['0']}")
            for i, (name, cfg) in enumerate(models, 1):
                marker = f"{_A['g']} ●{_A['0']}" if name == active else "  "
                base_short = cfg.get("base", "").replace("https://", "").replace("http://", "").split("/")[0]
                print(f"  {marker} {i}. {_A['c']}{name}{_A['0']} — {cfg['id']}")
                print(f"       {base_short}")
            print(f"\n  当前: {_A['g']}{active}{_A['0']} | 切换: {_A['g']}/models <名称|序号>{_A['0']}\n")
            return False
        # 切换模型
        try:
            old, new = _model_registry.switch(arg)
            cfg = _model_registry.active_config
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model_name=cfg["id"], openai_api_key=cfg["key"],
                             openai_api_base=cfg.get("base", ""))
            _llm_with_tools_holder[0] = llm.bind_tools(ALL_TOOLS)
            print(f"\n  ✅ 模型已切换: {_A['y']}{old}{_A['0']} → {_A['g']}{new}{_A['0']} ({cfg['id']})\n")
        except ValueError as e:
            print(f"\n  ✗ {e}\n")
        return False

    print(f"  {_A['r']}未知命令: {a}（输入 /help 查看帮助）{_A['0']}")
    return False

# ══════════════════════════════════════════════════════════════════════════════
# ⑥ 对话循环层：核心消息处理
# ══════════════════════════════════════════════════════════════════════════════
def _clear_input(text: str):
    """清除终端上的 '> 用户输入' 行（智能处理自动换行的多行情况）"""
    import shutil
    from agents.ui.spinner import _cw
    try:
        cols = shutil.get_terminal_size().columns
    except OSError:
        cols = 80
    # "> " 前缀占 2 列 + 用户文本
    display_w = 2 + _cw(text)
    # 计算占了几行（向上取整）
    lines = max(1, (display_w + cols - 1) // cols)
    # 上移 lines 行，每行清除
    for _ in range(lines):
        sys.stdout.write("\033[1A\033[2K")
    sys.stdout.flush()


def _echo_user(text: str):
    """在滚动区显示 User + 缩进内容 + Agent 标签"""
    print(f"{_A['g']}User{_A['0']}")
    for line in text.split("\n"):
        print(f"    {line}")
    print()
    print(f"{_A['c']}Agent{_A['0']}")


async def _turn(graph, user_input: str, thread_id: str, HumanMessage, ToolMessage):
    """单轮对话：Thinking动画 → LangGraph流式 → 解析正文/思考 → 工具动画 → 打印"""
    tp = rp = False

    _clear_input(user_input)
    _echo_user(user_input)
    start_spinner("Thinking")
    parser = StreamParser()

    # 注入实时时间和工作目录到系统提示词
    from datetime import datetime
    _now = datetime.now().strftime('%Y-%m-%d %A %H:%M:%S')
    _cwd = os.environ.get("LAUNCH_DIR", str(Path.cwd()))
    _sys = SYSTEM_PROMPT.replace("{current_time}", _now).replace("{cwd}", _cwd)

    inp = {"messages": [
        HumanMessage(content=f"[系统指令]\n{_sys}\n\n[用户消息]\n{user_input}"),
    ]}
    cfg = {"configurable": {"thread_id": thread_id}}

    try:
        rnd = 0
        async for ev in graph.astream_events(inp, config=cfg, version="v2"):
            if _interrupt.is_set(): raise KeyboardInterrupt("打断")
            et = ev.get("event", "")

            if et == "on_chat_model_start":
                if rnd > 0: start_spinner("Thinking")
                rnd += 1

            elif et == "on_chat_model_end":
                # 提取 token 用量
                try:
                    out = ev.get("data", {}).get("output", {})
                    meta = {}
                    if hasattr(out, "response_metadata"):
                        meta = out.response_metadata or {}
                    usage = meta.get("token_usage", meta.get("usage", {}))
                    if isinstance(usage, dict):
                        _total_tokens[0] += usage.get("total_tokens", 0)
                except Exception:
                    pass

            elif et == "on_chat_model_stream":
                chunk = ev["data"]["chunk"]
                content = getattr(chunk, "content", "") or ""
                if not content: continue
                for is_t, text in parser.feed(content):
                    if is_t:
                        if not tp:
                            stop_spinner(); tp = True
                            sys.stdout.write(_A["gr"]); sys.stdout.flush()
                        type_text(text.replace("\n", "\n    "), 0.03)
                    else:
                        if not rp:
                            stop_spinner()
                            if tp: sys.stdout.write(_A["0"]); print()
                            rp = True
                        type_text(text.replace("\n", "\n    "), 0.025)

            elif et == "on_tool_start":
                nm = ev.get("name", "unknown")
                inp2 = ev.get("data", {}).get("input", {})
                params = "，".join(f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}' for k, v in inp2.items()) if inp2 else ""
                info = f"{nm}（{params}）" if params else nm
                stop_spinner(); start_spinner("ToolCalling", info)

            elif et == "on_tool_end":
                stop_spinner()
                tp = rp = False
                parser = StreamParser()

    except (KeyboardInterrupt, asyncio.CancelledError):
        stop_spinner(); print(f"\n{_A['y']}⏹ 已打断{_A['0']}"); _interrupt.clear(); return
    except Exception as e:
        stop_spinner(); print(f"\n  [{_A['r']}错误: {e}{_A['0']}]")

    # 刷出剩余内容
    for is_t, text in parser.done():
        if is_t:
            if not tp: stop_spinner(); tp = True; sys.stdout.write(f"{_A['d']}{_A['gr']}"); sys.stdout.flush()
            type_text(text.replace("\n", "\n    "), 0.03)
        else:
            if not rp:
                stop_spinner()
                if tp: sys.stdout.write(_A["0"]); print()
                rp = True
            type_text(text.replace("\n", "\n    "), 0.025)

    if not tp and not rp: stop_spinner(); print("  (无回复)")
    elif tp and not rp: sys.stdout.write(_A["0"])
    print()

# ══════════════════════════════════════════════════════════════════════════════
# ⑦ 入口层：主程序（Banner 先行，重型导入后台加载）
# ══════════════════════════════════════════════════════════════════════════════
async def run_chat_loop(graph, mgr: SessionManager, HumanMessage, ToolMessage):
    while True:
        try:
            ui = input(f"\r{_A['g']}> {_A['0']}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_A['c']}再见！👋{_A['0']}")
            break
        if not ui:
            continue
        if ui.startswith("/"):
            if await _cmd(ui, graph, mgr): print(f"{_A['c']}再见！👋{_A['0']}"); break
            continue
        await _turn(graph, ui, mgr.active_id, HumanMessage, ToolMessage)
        mgr.touch()

async def main():
    # Banner
    mn = _model_registry.active_name
    mid = _model_registry.active_config.get("id", "?")
    _launch_dir = os.environ.get("LAUNCH_DIR", None)
    _actual_cwd = str(Path.cwd())
    _display_cwd = _launch_dir or _actual_cwd
    print(f"\n╔{'═' * 48}╗\n║  🤖 {_A['b']}My Agent — 1.0.0{_A['0']}{' ' * 27}║\n╚{'═' * 48}╝")
    print(f"  📡 {_A['c']}{mn}{_A['0']} · {mid}")
    print(f"  📂 {_A['d']}{_display_cwd}{_A['0']}\n")

    # 后台静默加载重型模块（无加载提示）
    _ready = threading.Event()
    _graph_data = [None]
    _load_error = [None]

    def _background_load():
        try:
            _graph_data[0] = _build_graph()
        except Exception as e:
            _load_error[0] = e
        finally:
            _ready.set()

    threading.Thread(target=_background_load, daemon=True).start()

    # 轻量操作
    mgr = SessionManager()
    if not mgr.list_sessions():
        d = mgr.create("默认")
        print(f"  {_A['d']}已创建默认会话: {d.name} ({d.thread_id}){_A['0']}")
    cur = mgr.active_session
    if cur: print(f"  当前会话:{_A['d']} {cur.name} ({cur.thread_id}){_A['0']}")
    print(f"  共 {len(mgr.list_sessions())} 个会话 · 输入 {_A['c']}/help{_A['0']} 查看命令")
    print()

    # 主循环
    while True:
        try:
            ui = input(f"\r{_A['g']}> {_A['0']}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_A['c']}再见！👋{_A['0']}")
            break
        if not ui:
            continue

        if ui.startswith("/"):
            if await _cmd(ui, None, mgr):
                print(f"{_A['c']}再见！👋{_A['0']}")
                break
            continue

        # 首次发送：等待后台加载完成
        if not _ready.is_set():
            start_spinner("Starting")
            _ready.wait()
            stop_spinner()

        if _load_error[0]:
            print(f"\n  [{_A['r']}启动失败: {_load_error[0]}{_A['0']}]")
            return

        # 编译图 + 打开持久化
        g, HM, TM = _graph_data[0]
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        db = Path(__file__).parent.parent / "chat_history.db"

        async with AsyncSqliteSaver.from_conn_string(str(db)) as ck:
            graph = g.compile(checkpointer=ck)
            await _turn(graph, ui, mgr.active_id, HM, TM)
            mgr.touch()
            await run_chat_loop(graph, mgr, HM, TM)
        break

if __name__ == "__main__":
    asyncio.run(main())
