# chat_agent.py — My Agent：多会话 + 流式输出 + LangGraph 工具循环
# 架构：配置层 → 工具层 → Agent核心 → 会话管理 → UI层 → 入口层
"""
技术栈：LangGraph + LangChain + ChromaDB + BM25 + ChatOpenAI + asyncio
设计：SessionManager(JSON索引) + AsyncSqliteSaver(历史持久化) + 斜杠命令
"""
from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# ① 配置层：环境变量 & 标准库
# ══════════════════════════════════════════════════════════════════════════════
import os, sys, json, random, re, threading, time, uuid, signal as _signal, asyncio
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Annotated

sys.path.insert(0, str(Path(__file__).parent.parent))

# HuggingFace 镜像 + Windows VT100
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_HTTP_TIMEOUT"] = "300"
if sys.platform == "win32":
    import ctypes
    k = ctypes.windll.kernel32
    h = k.GetStdHandle(-11)
    m = ctypes.c_ulong()
    if k.GetConsoleMode(h, ctypes.byref(m)):
        k.SetConsoleMode(h, m.value | 0x0004)

# Ctrl+C 打断标志
_interrupt = threading.Event()
_signal.signal(_signal.SIGINT, lambda *_: _interrupt.set())

# ══════════════════════════════════════════════════════════════════════════════
# ② 配置层：AI 模型 & UI 常量
# ══════════════════════════════════════════════════════════════════════════════
THINK_TAGS = [
    ("<" + "think" + ">", "</" + "think" + ">"),  # MiniMax / DeepSeek（旧版确认）
    ("<thinking>", "</thinking>"),                 # Qwen
    ("<think/>", "</think/>"),                     # Gemma
]
MAX_BUFFER = 30  # 流式解析 buffer 上限

SYSTEM_PROMPT = """你是活泼开朗、风趣幽默的AI助手。回答认真负责，不猜测未知内容。"""

# ANSI 颜色（终端 UI）
_A = {"b": "\033[1m", "d": "\033[2m", "c": "\033[36m", "y": "\033[33m",
      "g": "\033[32m", "r": "\033[31m", "gr": "\033[90m", "0": "\033[0m"}

# ══════════════════════════════════════════════════════════════════════════════
# ③ 工具层：LangChain @tool 封装（转发至 tools/ 目录）
# ══════════════════════════════════════════════════════════════════════════════
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from config.settings import API_KEY, API_BASE, MODEL_FAST
from tools.rag_tool import rag_search as _rs, ingest as _ing
from tools.weather_tool import get_weather as _gw
from tools.search.web_search_tool import web_search as _wss
from tools.search.web_fetch_tool import web_fetch as _wfs

@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    return _gw(city)

@tool
def rag_search(query: str) -> str:
    """搜索现有知识库（默认是游梦的Obsidian）"""
    return _rs(query)

@tool
def add_knowledge(folder_path: str) -> str:
    """将文件夹里的文档加入知识库（强制重建索引）"""
    if not os.path.isabs(folder_path):
        return "请提供绝对路径"
    if not os.path.isdir(folder_path):
        return f"目录不存在: {folder_path}"
    _ing(folder_path, force=True)
    return "入库完成！"

@tool
def get_current_time() -> str:
    """获取当前日期、时间和星期几（系统本地时区）"""
    from datetime import datetime
    now = datetime.now()
    try:
        import time as _t
        tz = _t.tzname[0]
    except Exception:
        tz = ""
    return f"{now.strftime('%Y-%m-%d %A %H:%M:%S')} ({tz})"

ALL_TOOLS = [get_weather, rag_search, add_knowledge, get_current_time, _wss, _wfs]

# ══════════════════════════════════════════════════════════════════════════════
# ④ Agent 核心层：LangGraph 状态图
# ══════════════════════════════════════════════════════════════════════════════
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 消息历史链表

llm = ChatOpenAI(model_name=MODEL_FAST, openai_api_key=API_KEY, openai_api_base=API_BASE)
llm_with_tools = llm.bind_tools(ALL_TOOLS)

def _sanitize_messages(msgs: list) -> list:
    """确保消息序列符合 API 要求：tool_use 后必须紧跟 tool_result"""
    clean = [m for m in msgs if not isinstance(m, SystemMessage)]
    result = []
    for m in clean:
        # 检查前一条是否是带 tool_calls 的 AIMessage
        if result and hasattr(result[-1], "tool_calls") and result[-1].tool_calls:
            if isinstance(m, ToolMessage):
                result.append(m)
            else:
                # tool_use 后面不是 ToolMessage → 用空结果补上
                for tc in result[-1].tool_calls:
                    result.append(ToolMessage(content="[工具调用被中断，无结果]", tool_call_id=tc["id"]))
                result.append(m)
        else:
            # ToolMessage 没有对应的 tool_use → 跳过
            if isinstance(m, ToolMessage):
                # 检查是否有匹配的 tool_use
                has_match = False
                for prev in result:
                    if hasattr(prev, "tool_calls") and prev.tool_calls:
                        if any(tc["id"] == m.tool_call_id for tc in prev.tool_calls):
                            has_match = True
                            break
                if not has_match:
                    continue  # 丢弃孤儿 ToolMessage
            result.append(m)
    return result

def _parse_retry_wait(err_msg: str) -> float:
    """从 API 错误信息中提取建议等待秒数，找不到则返回 None"""
    # 常见格式："Please retry after 20 seconds" / "请等待30秒后重试" / "rate limit ... 20s"
    for pat in [
        r'(?:retry\s*after|wait|等待)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|second|秒)',
        r'rate\s*limit.*?(\d+(?:\.\d+)?)\s*(?:s|sec|second|秒)',
        r'(\d+(?:\.\d+)?)\s*(?:s|sec|second|秒).*(?:retry|重试)',
        r'retry.*?(\d+(?:\.\d+)?)\s*(?:ms|msec|millisecond|毫秒)',
    ]:
        m = re.search(pat, err_msg, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            # 毫秒转秒
            if 'ms' in m.group(0).lower() or '毫秒' in m.group(0):
                return val / 1000
            return val
    return None

MAX_LLM_RETRIES = 10

def _think(state: AgentState) -> dict:
    """过滤 SystemMessage + 修复消息序列 后调用 LLM（含重试）"""
    safe_msgs = _sanitize_messages(state["messages"])
    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            return {"messages": [llm_with_tools.invoke(safe_msgs)]}
        except Exception as e:
            is_last = attempt == MAX_LLM_RETRIES
            err_msg = str(e)
            # 最后一次尝试直接抛出，不再重试
            if is_last:
                raise
            # 从错误信息提取等待时间，没有则指数退避
            wait = _parse_retry_wait(err_msg) or min(2 ** attempt, 60)
            print(f"\n  {_A['y']}⚠ LLM 第{attempt}次调用失败，{wait:.1f}s 后重试: {_A['d']}{err_msg[:120]}{_A['0']}")
            time.sleep(wait)

def _exec_tool(state: AgentState) -> dict:
    """依次执行 LLM 请求的工具，结果通过 ToolMessage 回传（异常不外抛，交给 LLM 处理）"""
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

_g = StateGraph(AgentState)
_g.add_node("think", _think)
_g.add_node("exec_tool", _exec_tool)
_g.add_edge(START, "think")
_g.add_conditional_edges("think", _route, {"use_tool": "exec_tool", "done": END})
_g.add_edge("exec_tool", "think")

# ══════════════════════════════════════════════════════════════════════════════
# ⑤ 流式解析层：StreamParser（处理思考标签被分包切断）
# ══════════════════════════════════════════════════════════════════════════════
class StreamParser:
    """识别 <think/> 标签，分流模型输出为正文/思考两类"""
    def __init__(self):
        self.buf = ""
        self.in_think = False
        self.q: list[tuple[bool, str]] = []

    def feed(self, chunk: str) -> list[tuple[bool, str]]:
        self.buf += chunk
        self._drain()
        return self._flush()

    def done(self) -> list[tuple[bool, str]]:
        if self.buf:
            self.q.append((self.in_think, self.buf))
            self.buf = ""
            self.in_think = False
        return self._flush()

    def _tag(self, text: str, opening: bool) -> tuple[int, int]:
        for o, c in THINK_TAGS:
            idx = text.find(o if opening else c)
            if idx >= 0:
                return idx, len(o if opening else c)
        return -1, 0

    def _drain(self):
        while True:
            if self.in_think:
                i, l = self._tag(self.buf, opening=False)
                if i >= 0:
                    if self.buf[:i]:
                        self.q.append((True, self.buf[:i]))
                    self.buf = self.buf[i + l:]
                    self.in_think = False
                else:
                    break
            else:
                i, l = self._tag(self.buf, opening=True)
                if i >= 0:
                    if self.buf[:i]:
                        self.q.append((False, self.buf[:i]))
                    self.buf = self.buf[i + l:]
                    self.in_think = True
                else:
                    max_l = max(len(o) for o, _ in THINK_TAGS)
                    if len(self.buf) > MAX_BUFFER + max_l:
                        self.q.append((False, self.buf))
                        self.buf = ""
                    break

    def _flush(self) -> list[tuple[bool, str]]:
        out = self.q[:]
        self.q.clear()
        return out

# ══════════════════════════════════════════════════════════════════════════════
# ⑥ 会话管理层：SessionManager（JSON持久化 + LRU排序）
# ══════════════════════════════════════════════════════════════════════════════
_SF = Path(__file__).parent.parent / "sessions.json"

@dataclass
class Session:
    thread_id: str  # UUID[:8]，关联 LangGraph thread_id
    name: str
    created_at: float
    updated_at: float

class SessionManager:
    """sessions.json 持久化 + 按最后活跃时间 LRU 排序"""
    def __init__(self, f: Path = _SF):
        self._f = f
        self._data = self._load()

    def _load(self) -> dict:
        if self._f.exists():
            try: return json.loads(self._f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError): pass
        return {"sessions": [], "active_thread_id": "default"}

    def _save(self):
        self._f.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")

    @property
    def active_id(self) -> str:
        return self._data.get("active_thread_id", "default")

    @property
    def active_session(self) -> Session | None:
        for s in self.list_sessions():
            if s.thread_id == self.active_id:
                return s
        return None

    def list_sessions(self) -> list[Session]:
        ss = [Session(**d) for d in self._data.get("sessions", []) if isinstance(d, dict)]
        ss.sort(key=lambda s: s.updated_at, reverse=True)
        return ss

    def create(self, name: str = None) -> Session:
        s = Session(uuid.uuid4().hex[:8],
                    name or f"session_{len(self.list_sessions()) + 1}",
                    time.time(), time.time())
        self._data.setdefault("sessions", []).append(asdict(s))
        self._data["active_thread_id"] = s.thread_id
        self._save()
        return s

    def switch(self, index: int) -> Session:
        ss = self.list_sessions()
        if not 1 <= index <= len(ss):
            raise ValueError(f"序号超出范围 (1-{len(ss)})")
        self._data["active_thread_id"] = ss[index - 1].thread_id
        self._save()
        return ss[index - 1]

    def switch_by_name(self, name: str) -> Session:
        ms = [s for s in self.list_sessions() if name.lower() in s.name.lower()]
        if not ms: raise ValueError(f"未找到: {name}")
        if len(ms) > 1: raise ValueError(f"多个匹配 '{name}': {[s.name for s in ms]}")
        self._data["active_thread_id"] = ms[0].thread_id
        self._save()
        return ms[0]

    def rename(self, new_name: str):
        for s in self._data.get("sessions", []):
            if s.get("thread_id") == self.active_id:
                s["name"] = new_name
                break
        self._save()

    def delete(self, index: int) -> str:
        ss = self.list_sessions()
        if not 1 <= index <= len(ss): raise ValueError(f"序号超出范围 (1-{len(ss)})")
        removed = ss[index - 1]
        dl = self._data.get("sessions", [])
        dl.pop(index - 1)
        self._data["sessions"] = dl
        if self._data.get("active_thread_id") == removed.thread_id:
            self._data["active_thread_id"] = dl[0]["thread_id"] if dl else "default"
        self._save()
        return removed.thread_id

    def touch(self, tid: str = None):
        t = tid or self.active_id
        for s in self._data.get("sessions", []):
            if s.get("thread_id") == t:
                s["updated_at"] = time.time()
                break
        self._save()

# ══════════════════════════════════════════════════════════════════════════════
# ⑦ UI层：终端动画 & 打字机效果
# ══════════════════════════════════════════════════════════════════════════════
_SPIN_STOP = threading.Event()
_SPIN_LABEL = "Thinking"
_SPIN_INFO = ""
_SPIN_THREAD: threading.Thread | None = None
_SPIN_START = 0.0
_SP = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_LC = {"Thinking": _A["y"], "tool_calling": _A["c"]}

def _elapsed(s: float) -> str:
    s = int(s)
    if s < 60: return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60: return f"{m}m{s}s"
    h, m = divmod(m, 60)
    return f"{h}h{m}m{s}s"

def _cw(s: str) -> int:
    w = 0
    for c in s:
        cp = ord(c)
        w += 2 if (0x4E00 <= cp <= 0x9FFF or 0x3000 <= cp <= 0x303F or
                    0xFF01 <= cp <= 0xFF60 or 0x2018 <= cp <= 0x201F) else 1
    return w

def _trunc(text: str, avail: int) -> str:
    if _cw(text) <= avail: return text
    h2 = (avail - 3) // 2
    def _w(c): cp = ord(c); return 0x4E00 <= cp <= 0x9FFF or 0x3000 <= cp <= 0x303F or 0xFF01 <= cp <= 0xFF60 or 0x2018 <= cp <= 0x201F
    head, w = [], 0
    for c in text:
        cw = 2 if _w(c) else 1
        if w + cw > h2: break
        head.append(c); w += cw
    tail, w = [], 0
    for c in reversed(text):
        cw = 2 if _w(c) else 1
        if w + cw > h2: break
        tail.append(c); w += cw
    return ''.join(head) + '...' + ''.join(reversed(tail))

def _spin_loop():
    i = 0
    while not _SPIN_STOP.is_set():
        sp = _SP[i % len(_SP)]
        color = _LC.get(_SPIN_LABEL, _A["y"])
        e = _elapsed(time.time() - _SPIN_START)
        timer = f"（{e}）" if time.time() - _SPIN_START >= 5 else ""
        if _SPIN_INFO:
            sys.stdout.write(f"\r\033[2K  {color}{sp} {_SPIN_LABEL}...{timer}{_A['0']}\n"
                             f"\r\033[2K    {_A['d']}⎿ {_SPIN_INFO}{_A['0']}\033[1A")
        else:
            sys.stdout.write(f"\r\033[2K  {color}{sp} {_SPIN_LABEL}...{timer}{_A['0']}")
        sys.stdout.flush()
        time.sleep(0.08)
        i += 1

def start_spinner(label: str = "Thinking", info: str = ""):
    global _SPIN_THREAD, _SPIN_LABEL, _SPIN_INFO, _SPIN_START
    if _SPIN_THREAD and _SPIN_THREAD.is_alive():
        _SPIN_STOP.set(); _SPIN_THREAD.join(timeout=1); _SPIN_STOP.clear()
    _SPIN_LABEL = label
    try: cols = os.get_terminal_size().columns
    except OSError: cols = 80
    _SPIN_INFO = _trunc(info, max(cols - 8, 20))
    _SPIN_START = time.time()
    _SPIN_STOP.clear()
    # 启动前先输出换行，让 spinner 与上方内容有间距
    sys.stdout.write("\n"); sys.stdout.flush()
    _SPIN_THREAD = threading.Thread(target=_spin_loop, daemon=True)
    _SPIN_THREAD.start()

def stop_spinner(success: bool = True):
    global _SPIN_THREAD, _SPIN_INFO
    was = _SPIN_THREAD and _SPIN_THREAD.is_alive()
    if was:
        _SPIN_STOP.set(); _SPIN_THREAD.join(timeout=2)
        mk = f"{_A['g']}●{_A['0']}" if success else f"{_A['r']}●{_A['0']}"
        if _SPIN_INFO:
            sys.stdout.write(f"\r\033[2K  {mk} {_SPIN_LABEL}{_A['0']}\n"
                             f"\r\033[2K    {_A['d']}⎿ {_SPIN_INFO}{_A['0']}\n")
        else:
            sys.stdout.write(f"\r\033[2K  {mk} {_SPIN_LABEL}{_A['0']}\n")
        sys.stdout.flush()
    _SPIN_THREAD = None; _SPIN_INFO = ""

def type_text(text: str, delay: float = 0.025):
    for c in text:
        print(c, end="", flush=True)
        time.sleep(delay * (0.5 + random.random()))

def _rel(ts: float) -> str:
    e = time.time() - ts
    if e < 60: return "刚刚"
    if e < 3600: return f"{int(e // 60)}分钟前"
    if e < 86400: h = int(e // 3600); return "1小时前" if h == 1 else f"{h}小时前"
    if e < 172800: return "昨天"
    return f"{int(e // 86400)}天前"

# ══════════════════════════════════════════════════════════════════════════════
# ⑧ UI层：会话列表渲染 & 交互式选择
# ══════════════════════════════════════════════════════════════════════════════
def _render_list(mgr: SessionManager):
    ss = mgr.list_sessions()
    aid = mgr.active_id
    print(f"\n{_A['c']}{_A['b']}{'#':<4} {'名称':<20} {'ID':<12} {'更新'}{_A['0']}")
    print(f"{_A['d']}{'─' * 52}{_A['0']}")
    for i, s in enumerate(ss, 1):
        ia = s.thread_id == aid
        p = f"{_A['g']}*{_A['0']}" if ia else " "
        nc = _A["g"] if ia else _A["gr"]
        nm = s.name[:18] if len(s.name) <= 18 else s.name[:16] + ".."
        ts = _rel(s.updated_at)
        mk = f"{p}{nc}{i:<3}{_A['0']}"
        if ia:
            print(f"{mk} {_A['b']}{_A['c']}{nm:<20}{_A['0']} {s.thread_id:<12} {_A['d']}{ts}{_A['0']}")
        else:
            print(f"{mk} {nm:<20} {s.thread_id:<12} {_A['gr']}{ts}{_A['0']}")
    cur = mgr.active_session
    print(f"\n  {_A['g']}当前:{_A['0']} {cur.name if cur else '默认'} · 共 {len(ss)} 个会话")

async def _select(mgr: SessionManager) -> Session | None:
    ss = mgr.list_sessions()
    if not ss:
        print(f"\n  {_A['y']}没有可切换的会话，输入 /new 创建{_A['0']}")
        return None
    aid = mgr.active_id
    print(f"\n  {_A['c']}{_A['b']}选择会话{_A['0']}{_A['d']} (Enter确认 Esc取消){_A['0']}")
    for i, s in enumerate(ss, 1):
        ia = s.thread_id == aid
        mk = f"{_A['g']}*{_A['0']}" if ia else " "
        dn = s.name[:24] if len(s.name) <= 24 else s.name[:22] + ".."
        if ia:
            print(f"  {mk} {_A['b']}{_A['g']}{i}.{_A['0']} {_A['b']}{dn}{_A['0']}  {_A['d']}{_rel(s.updated_at)}{_A['0']}")
        else:
            print(f"  {mk} {i}. {dn}  {_A['gr']}{_rel(s.updated_at)}{_A['0']}")
    while True:
        try:
            ch = input(f"\r  {_A['c']}> {_A['0']}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_A['d']}(已取消){_A['0']}")
            return None
        if not ch: continue
        try:
            idx = int(ch)
            if 1 <= idx <= len(ss): return ss[idx - 1]
            print(f"  {_A['r']}序号超出范围 (1-{len(ss)}){_A['0']}")
        except ValueError:
            try: return mgr.switch_by_name(ch)
            except ValueError as e: print(f"  {_A['r']}{e}{_A['0']}")

# ══════════════════════════════════════════════════════════════════════════════
# ⑨ 命令层：斜杠命令处理器
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
        print(f"  {_A['g']}/quit{_A['0']} / {_A['g']}/q{_A['0']}             退出程序\n")
        return False

    if a in ("/quit", "/q"): return True
    if a in ("/sessions", "/ls"): _render_list(mgr); return False

    if a == "/new":
        s = mgr.create(arg or None)
        print(f"\n  ✅ 新建会话: {s.name} ({s.thread_id})")
        return False

    if a in ("/switch", "/s"):
        if not arg:
            sel = await _select(mgr)
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
        if not arg: print(f"\n  {_A['y']}用法: /delete <序号>{_A['0']}"); _render_list(mgr); return False
        try: tid = mgr.delete(int(arg.strip())); print(f"\n  🗑️ 已删除会话: {tid}")
        except ValueError as e: print(f"\n  ✗ {e}")
        return False

    print(f"  {_A['r']}未知命令: {a}（输入 /help 查看帮助）{_A['0']}")
    return False

# ══════════════════════════════════════════════════════════════════════════════
# ⑩ 对话循环层：核心消息处理
# ══════════════════════════════════════════════════════════════════════════════
async def _turn(graph, user_input: str, thread_id: str):
    """单轮对话：Thinking动画 → LangGraph流式 → 解析正文/思考 → 工具动画 → 打印"""
    global tp, rp
    tp = rp = False

    start_spinner("Thinking")
    parser = StreamParser()

    # API不支持system角色，系统指令通过 HumanMessage 合并传递（避免双 HumanMessage 打断 tool 链）
    inp = {"messages": [
        HumanMessage(content=f"[系统指令]\n{SYSTEM_PROMPT}\n\n[用户消息]\n{user_input}"),
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
                            print(f"{_A['c']}Agent{_A['0']} ", end="", flush=True); rp = True
                        type_text(text.replace("\n", "\n    "), 0.025)

            elif et == "on_tool_start":
                nm = ev.get("name", "unknown")
                inp2 = ev.get("data", {}).get("input", {})
                params = "，".join(f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}' for k, v in inp2.items()) if inp2 else ""
                info = f"{nm}（{params}）" if params else nm
                stop_spinner(); start_spinner("tool_calling", info)

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
                print(f"{_A['c']}Agent{_A['0']} ", end="", flush=True); rp = True
            type_text(text.replace("\n", "\n    "), 0.025)

    if not tp and not rp: stop_spinner(); print(f"{_A['c']}Agent{_A['0']} (无回复)")
    elif tp and not rp: sys.stdout.write(_A["0"])
    print()

# ══════════════════════════════════════════════════════════════════════════════
# ⑪ 入口层：主程序
# ══════════════════════════════════════════════════════════════════════════════
async def run_chat_loop(graph, mgr: SessionManager):
    while True:
        sn = mgr.active_session.name if mgr.active_session else "默认"
        try: ui = input(f"{_A['g']}> {_A['d']}[{sn}]{_A['0']} ").strip()
        except (EOFError, KeyboardInterrupt): print(f"\n{_A['c']}再见！👋{_A['0']}"); break
        if not ui: continue
        if ui.startswith("/"):
            if await _cmd(ui, graph, mgr): print(f"\n{_A['c']}再见！👋{_A['0']}"); break
            continue
        await _turn(graph, ui, mgr.active_id)
        mgr.touch()

async def main():
    print(f"\n╔{'═' * 48}╗\n║  🤖 {_A['b']}My Agent — 1.0.0{_A['0']}{' ' * 27}║\n╚{'═' * 48}╝\n")
    mgr = SessionManager()
    if not mgr.list_sessions():
        d = mgr.create("默认")
        print(f"  {_A['d']}已创建默认会话: {d.name} ({d.thread_id}){_A['0']}")
    cur = mgr.active_session
    if cur: print(f"  当前会话:{_A['d']} {cur.name} ({cur.thread_id}){_A['0']}")
    print(f"  共 {len(mgr.list_sessions())} 个会话 · 输入 {_A['c']}/help{_A['0']} 查看命令\n")

    db = Path(__file__).parent.parent / "chat_history.db"
    cm = AsyncSqliteSaver.from_conn_string(str(db))
    async with cm as ck:
        graph = _g.compile(checkpointer=ck)
        await run_chat_loop(graph, mgr)

if __name__ == "__main__":
    asyncio.run(main())
