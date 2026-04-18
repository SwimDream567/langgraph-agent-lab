# chat_agent.py — My Agent 2.0.0：Multi-Agent Supervisor 入口文件
# 架构：配置层 → 工具层 → Agent核心(Supervisor+主Agent+2子Agent) → 命令层 → 对话循环 → 入口
"""
技术栈：LangGraph Supervisor + 手动 StateGraph + ChatOpenAI + asyncio
设计：Supervisor(路由) → chat(主Agent·全能) / coder(代码专家) / planner(研究规划)
模块：agents/session_manager.py | agents/ui/{spinner,stream_parser,display}.py
启动优化：重型导入(langchain/langgraph)延迟到 main() 内，Banner 先行显示
"""
import os, sys, re, json, threading, time, asyncio, signal as _signal
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
_total_tokens = [0]  # 累计 token 消耗

# Agent 显示名
_AGENT_NAMES = {"chat": "Agent", "coder": "Coder", "planner": "Planner"}


def _get_env_block():
    """生成环境信息块（cwd/time 在 _turn 里动态替换）"""
    home = str(Path.home())
    user_name = os.environ.get("USERNAME", os.environ.get("USER", "unknown"))
    return f"""## 运行环境
- 用户：{user_name}
- 系统：Windows
- 主目录：{home}
- 当前工作目录：{{cwd}}
- 当前时间：{{current_time}}

当用户给相对路径时，基于 {{cwd}} 解析。"""


ENV_BLOCK = _get_env_block()


# ══════════════════════════════════════════════════════════════════════════════
# ③ 工具层：LangChain @tool 封装（重型导入延迟到 main）
# ══════════════════════════════════════════════════════════════════════════════
ALL_TOOLS = []
MAIN_TOOLS = []
CODER_TOOLS = []
PLANNER_TOOLS = []
_ToolMessage = None
_HumanMessage = None


def _build_tools():
    """延迟构建工具列表，按 Agent 职责分组"""
    global ALL_TOOLS, MAIN_TOOLS, CODER_TOOLS, PLANNER_TOOLS
    global _ToolMessage, _HumanMessage

    from langchain_core.messages import HumanMessage, ToolMessage

    # 导入所有工具
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

    # 主 Agent：全能（所有工具）
    MAIN_TOOLS[:] = [get_weather, get_current_time, web_search, web_fetch,
                     rag_search, add_knowledge, read_file, list_dir,
                     search_file, search_content, write_file, edit_file, run_command]
    # Coder：文件操作 + 命令执行（深度代码任务）
    CODER_TOOLS[:] = [read_file, list_dir, search_file, search_content,
                      write_file, edit_file, run_command]
    # Planner：搜索 + 知识库（深度研究/规划任务）
    PLANNER_TOOLS[:] = [web_search, web_fetch, rag_search, add_knowledge]
    ALL_TOOLS[:] = MAIN_TOOLS + CODER_TOOLS + PLANNER_TOOLS


# ══════════════════════════════════════════════════════════════════════════════
# ④ Agent 核心层：Supervisor + 主Agent(全能) + 2 子 Agent（延迟构建）
# ══════════════════════════════════════════════════════════════════════════════
def _sanitize_messages(msgs):
    """
    清理消息链中的孤儿 tool_calls：确保每个 AIMessage.tool_calls
    都有对应的 ToolMessage，反之亦然。防止旧 checkpoint 数据导致 API 报错。
    """
    # 第一遍：收集所有 tool_call ID 和 ToolMessage ID
    tc_ids = set()   # AIMessage.tool_calls 中的 id
    tm_ids = set()   # ToolMessage.tool_call_id
    for m in msgs:
        for tc in (getattr(m, 'tool_calls', None) or []):
            tc_ids.add(tc.get('id'))
        tmid = getattr(m, 'tool_call_id', None)
        if tmid:
            tm_ids.add(tmid)

    # 第二遍：过滤/修复
    result = []
    for m in msgs:
        tcs = getattr(m, 'tool_calls', None) or []
        tmid = getattr(m, 'tool_call_id', None)

        if tcs:
            # AIMessage 带 tool_calls：只保留有结果的
            valid = [tc for tc in tcs if tc.get('id') in tm_ids]
            if valid and len(valid) < len(tcs):
                # 部分孤儿：保留有效的
                try:
                    m = m.model_copy(update={'tool_calls': valid})
                except AttributeError:
                    m = m.copy(update={'tool_calls': valid})
            elif not valid:
                # 全部孤儿：清空 tool_calls，保留 content
                try:
                    m = m.model_copy(update={'tool_calls': []})
                except AttributeError:
                    m = m.copy(update={'tool_calls': []})

        if tmid and tmid not in tc_ids:
            continue  # 孤儿 ToolMessage，丢弃

        result.append(m)
    return result


def _build_graph():
    """
    构建 Supervisor 多 Agent 图：
    Supervisor(路由) → chat(主Agent·全能) / coder(代码专家) / planner(研究规划)

    主 Agent 拥有所有工具，大部分对话直接处理。
    只有明确的深度代码/研究需求才委托给子 Agent。
    兼容 MiniMax 等不支持 system 角色的模型。
    """
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
    )

    # ── 构建单个子 Agent 的 StateGraph ──
    def _make_sub_agent(tools, prompt_text):
        """
        手动构建子 Agent 图：think → exec_tool 循环。
        系统指令在 think 时临时注入到消息列表（不保存到 state），
        兼容不支持 system 角色的模型（MiniMax 等）。
        """
        # 工具名 → 工具对象映射
        tool_map = {t.name: t for t in tools}
        tool_desc = "\n".join(f"- {t.name}: {t.description.split(chr(10))[0]}" for t in tools)
        # 绑定工具到 LLM，让模型能生成结构化 tool_calls
        llm_with_tools = llm.bind_tools(tools)

        class SubState(TypedDict):
            messages: Annotated[list, add_messages]

        def think(state: SubState) -> dict:
            msgs = list(state["messages"])
            # 构造系统指令（临时，不保存到 state）
            sys_block = f"{prompt_text}\n\n## 可用工具\n{tool_desc}\n\n请根据上下文回答用户问题或调用工具。"
            llm_input = [HumanMessage(content=sys_block)] + msgs
            resp = llm_with_tools.invoke(llm_input)
            return {"messages": [resp]}

        def route(state: SubState) -> str:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            return END

        def exec_tools(state: SubState) -> dict:
            last = state["messages"][-1]
            results = []
            for tc in last.tool_calls:
                t = tool_map.get(tc["name"])
                if t:
                    try:
                        r = t.invoke(tc["args"])
                        results.append(ToolMessage(content=str(r), tool_call_id=tc["id"]))
                    except Exception as e:
                        results.append(ToolMessage(content=f"工具执行错误: {e}", tool_call_id=tc["id"]))
                else:
                    results.append(ToolMessage(content=f"未知工具: {tc['name']}", tool_call_id=tc["id"]))
            return {"messages": results}

        g = StateGraph(SubState)
        g.add_node("think", think)
        g.add_node("tools", exec_tools)
        g.add_edge(START, "think")
        g.add_conditional_edges("think", route, {"tools": "tools", END: END})
        g.add_edge("tools", "think")
        return g.compile()

    # ── 提示词 ──
    MAIN_PROMPT = """你是活泼开朗、风趣幽默的全能AI助手。回答认真负责，不猜测未知内容。

你可以搜索网页、抓取网页内容、查天气、看时间、搜索知识库、读写文件、执行命令。
大部分问题你应该直接回答，需要时主动调用工具。

## 工作原则
- 能自己回答的就直接回答，不必每次都搜索
- 用户给了 URL/链接，先调用 web_fetch 抓取内容再看
- 不确定的信息主动搜索确认
- 保持回答简洁有用，不要废话"""

    CODER_PROMPT = """你是专业的代码助手。你可以读取、编辑、创建文件，执行命令。

## 工作流程
1. 先用 read_file/search_content 了解现有代码
2. 用 edit_file 做精确修改（优先），或 write_file 创建新文件
3. 用 run_command 运行测试验证修改效果
4. 如果出错了，再次读取文件排查

## 注意事项
- 读取大文件时先读前 50 行了解结构，再按需读取关键部分
- edit_file 的 old_str 必须精确匹配（包括缩进、空行），否则会失败
- run_command 有 30 秒超时和危险命令拦截"""

    PLANNER_PROMPT = """你是专业的研究规划助手。你可以搜索网页、获取网页内容、搜索知识库。

## 工作原则
- 提供准确、全面的信息，标注信息来源
- 搜索结果不够时，尝试换个关键词再搜
- 对复杂问题，先拆解再逐步搜索
- 综合多个来源的信息给出分析，而非只引用一条"""

    # ── 创建 Agent ──
    chat_agent = _make_sub_agent(MAIN_TOOLS, MAIN_PROMPT)
    coder_agent = _make_sub_agent(CODER_TOOLS, CODER_PROMPT)
    planner_agent = _make_sub_agent(PLANNER_TOOLS, PLANNER_PROMPT)

    # ── 自定义 reducer：add_messages + 清理孤儿 tool_calls ──
    def _safe_add_messages(existing, new):
        combined = add_messages(existing, new)
        return _sanitize_messages(combined)

    # ── Supervisor 状态 ──
    class AgentState(TypedDict):
        messages: Annotated[list, _safe_add_messages]
        next_agent: str  # 路由目标：chat / coder / planner

    # ── Supervisor 路由（不用 SystemMessage，合并到 HumanMessage） ──
    ROUTER_PROMPT = """你是一个任务路由器。根据用户消息判断是否需要委托给专家处理。

分类规则（优先级从高到低）：
- coder：需要**深度代码开发**——写完整功能、重构代码、修复杂Bug、创建多文件项目
- planner：需要**深度研究分析**——多轮搜索、竞品分析、技术调研、复杂信息整理
- chat：其他所有情况（闲聊、问答、查天气、看网址、简单搜索、问时间、单次搜索即可回答的问题）

重要：不确定时选 chat。大部分情况 chat 就够了。
只回复一个英文单词：chat 或 coder 或 planner
不要解释，不要加标点。"""

    def _supervisor(state: AgentState) -> dict:
        """Supervisor 路由节点：分析用户意图，决定分配给哪个 Agent"""
        last = state["messages"][-1]
        content = getattr(last, "content", "") or str(last)
        if "[用户消息]" in content:
            content = content.split("[用户消息]")[-1].strip()[:500]

        try:
            # MiniMax 不支持 system 角色，合并到 HumanMessage
            resp = llm.invoke([
                HumanMessage(content=f"{ROUTER_PROMPT}\n\n用户消息：{content}"),
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

    # ── 组装图 ──
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
        print(f"  {_A['g']}/models <名称|序号>{_A['0']}       切换模型（需重启生效）")
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
            models = _model_registry.list_all()
            active = _model_registry.active_name
            print(f"\n  {_A['b']}{_A['c']}已配置的模型:{_A['0']}")
            for i, (name, cfg) in enumerate(models, 1):
                marker = f"{_A['g']} ●{_A['0']}" if name == active else "  "
                base_short = cfg.get("base", "").replace("https://", "").replace("http://", "").split("/")[0]
                print(f"  {marker} {i}. {_A['c']}{name}{_A['0']} — {cfg['id']}")
                print(f"       {base_short}")
            print(f"\n  当前: {_A['g']}{active}{_A['0']} | 切换: {_A['g']}/models <名称|序号>{_A['0']}")
            print(f"  {_A['y']}⚠ Multi-Agent 模式下切换模型需要重启{_A['0']}\n")
            return False
        # 切换模型（Multi-Agent 下需重启，但先更新注册表和 .env）
        try:
            old, new = _model_registry.switch(arg)
            cfg = _model_registry.active_config
            print(f"\n  ✅ 模型注册已切换: {_A['y']}{old}{_A['0']} → {_A['g']}{new}{_A['0']} ({cfg['id']})")
            print(f"  {_A['y']}⚠ 请重启 Agent 以应用新的 Multi-Agent 图{_A['0']}\n")
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
    display_w = 2 + _cw(text)
    lines = max(1, (display_w + cols - 1) // cols)
    for _ in range(lines):
        sys.stdout.write("\033[1A\033[2K")
    sys.stdout.flush()


def _echo_user(text: str):
    """在滚动区显示 User + 缩进内容"""
    print(f"{_A['g']}User{_A['0']}")
    for line in text.split("\n"):
        print(f"    {line}")
    print()


async def _turn(graph, user_input: str, thread_id: str, HumanMessage, ToolMessage):
    """单轮对话：Routing → 子Agent流式 → 工具动画 → 打印"""
    tp = rp = False

    _clear_input(user_input)
    _echo_user(user_input)
    parser = StreamParser()

    # 注入实时时间和工作目录
    from datetime import datetime
    _now = datetime.now().strftime('%Y-%m-%d %A %H:%M:%S')
    _cwd = os.environ.get("LAUNCH_DIR", str(Path.cwd()))
    _env = ENV_BLOCK.replace("{current_time}", _now).replace("{cwd}", _cwd)

    inp = {
        "messages": [
            HumanMessage(content=f"{_env}\n\n[用户消息]\n{user_input}"),
        ],
        "next_agent": "",
    }
    cfg = {"configurable": {"thread_id": thread_id}}

    try:
        _phase = "supervisor"  # supervisor | agent | tool
        async for ev in graph.astream_events(inp, config=cfg, version="v2"):
            if _interrupt.is_set(): raise KeyboardInterrupt("打断")
            et = ev.get("event", "")
            node = ev.get("metadata", {}).get("langgraph_node", "")

            # ── LLM 开始 ──
            if et == "on_chat_model_start":
                if node == "supervisor":
                    _phase = "supervisor"
                else:
                    _phase = "agent"
                    # 从 checkpoint_ns 提取子图名（格式：chat:UUID|think:UUID）
                    ckpt_ns = ev.get("metadata", {}).get("checkpoint_ns", "")
                    sub_name = ckpt_ns.split(":")[0] if ckpt_ns else node
                    agent_label = _AGENT_NAMES.get(sub_name, sub_name.title())
                    stop_spinner()
                    # 主Agent(chat) 直接显示 Agent，子Agent 显示 Agent | Coder/Planner
                    if sub_name == "chat":
                        print(f"{_A['c']}Agent{_A['0']}\n")
                    else:
                        print(f"{_A['c']}Agent{_A['0']} {_A['d']}|{_A['0']} {_A['c']}{agent_label}{_A['0']}\n")
                    start_spinner("Thinking")

            # ── LLM 结束（提取 token） ──
            elif et == "on_chat_model_end":
                if _phase == "supervisor":
                    continue  # supervisor 的 token 不计入
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

            # ── LLM 流式输出 ──
            elif et == "on_chat_model_stream":
                if _phase == "supervisor":
                    continue  # 抑制 supervisor 路由输出
                chunk = ev["data"]["chunk"]
                content = getattr(chunk, "content", "") or ""
                if not content:
                    continue
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

            # ── 工具调用开始 ──
            elif et == "on_tool_start":
                nm = ev.get("name", "unknown")
                inp2 = ev.get("data", {}).get("input", {})
                params = "，".join(
                    f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}'
                    for k, v in inp2.items()
                ) if inp2 else ""
                info = f"{nm}（{params}）" if params else nm
                stop_spinner(); start_spinner("ToolCalling", info)

            # ── 工具调用结束 ──
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
    print(f"\n╔{'═' * 50}╗")
    print(f"║  🤖 {_A['b']}My Agent — 2.0.0{_A['0']} (Multi-Agent){' ' * 19}║")
    print(f"╚{'═' * 50}╝")
    print(f"  📡 {_A['c']}{mn}{_A['0']} · {mid}")
    print(f"  📂 {_A['d']}{_display_cwd}{_A['0']}")
    print(f"  🏗️  {_A['d']}Supervisor → chat(全能) / coder / planner{_A['0']}\n")

    # 后台静默加载重型模块
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
