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

# 调试日志（写到文件，不受 stdout guard / logging level 影响）
_DEBUG_LOG_PATH = Path(__file__).parent.parent / "debug.log"
def _debug_log(msg: str):
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            import datetime
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass

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

# ── 全局图容器（支持热切换模型时重建） ──
_runtime = {
    "graph": None,          # 编译后的 CompiledGraph（带 checkpointer）
    "HM": None,             # HumanMessage 类
    "TM": None,             # ToolMessage 类
    "ck": None,             # AsyncSqliteSaver 实例
    "ck_ctx": None,         # async context manager（保持 alive）
    "llm": None,            # ChatOpenAI 实例（供 Layer 3 摘要压缩 + supervisor 用）
    "bind_chat": None,      # llm.bind_tools(MAIN_TOOLS) — 热切换时更新
    "bind_coder": None,     # llm.bind_tools(CODER_TOOLS)
    "bind_planner": None,   # llm.bind_tools(PLANNER_TOOLS)
    "ready": threading.Event(),
    "error": None,
    "model_name": None,     # 当前模型名
}

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
_compact_cooldown = [0]  # 自动压缩冷却计数（成功后设 3，每轮 -1）

# Agent 显示名
_AGENT_NAMES = {"chat": "Agent", "coder": "Coder", "planner": "Planner"}


def _get_env_block():
    """生成环境信息块（cwd/time 在 _turn 里动态替换）"""
    home = str(Path.home())
    return f"""## 运行环境
- 系统：Windows
- 主目录：{home}
- 当前工作目录：{{cwd}}
- 当前时间：{{current_time}}"""


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

    # ★ 加载 MCP 工具（从全局单例读取，由 main() 提前加载）
    try:
        from tools.mcp_loader import mcp_manager
        mcp_tools = mcp_manager.tools
    except Exception as e:
        print(f"  {_A['d']}[MCP] 加载跳过: {e}{_A['0']}")
        mcp_tools = []

    # 主 Agent：全能（所有工具）
    MAIN_TOOLS[:] = [get_weather, get_current_time, web_search, web_fetch,
                     rag_search, add_knowledge, read_file, list_dir,
                     search_file, search_content, write_file, edit_file, run_command]
    # Coder：文件操作 + 命令执行（深度代码任务）
    CODER_TOOLS[:] = [read_file, list_dir, search_file, search_content,
                      write_file, edit_file, run_command]
    # Planner：搜索 + 知识库（深度研究/规划任务）
    PLANNER_TOOLS[:] = [web_search, web_fetch, rag_search, add_knowledge]
    # MCP 工具自动分配给所有 Agent
    if mcp_tools:
        for tool_list in (MAIN_TOOLS, CODER_TOOLS, PLANNER_TOOLS):
            tool_list.extend(mcp_tools)
    ALL_TOOLS[:] = MAIN_TOOLS + CODER_TOOLS + PLANNER_TOOLS
    if mcp_tools:
        pass  # MCP 工具已加载，静默


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
        request_timeout=120,  # 本地模型首次加载可能需要 1-2 分钟
    )

    # ── 构建单个子 Agent 的 StateGraph ──
    def _make_sub_agent(bind_key: str, tools, prompt_text):
        """
        手动构建子 Agent 图：think → exec_tool 循环。
        系统指令在 think 时临时注入到消息列表（不保存到 state），
        兼容不支持 system 角色的模型（MiniMax 等）。

        bind_key: _runtime 中存储 llm.bind_tools 结果的 key
                  （"bind_chat" / "bind_coder" / "bind_planner"）
                  闭包运行时从 _runtime[bind_key] 读取当前绑定的 LLM，
                  这样换模型时不需要重建图。
        """
        # 工具名 → 工具对象映射
        tool_map = {t.name: t for t in tools}
        # 工具分类：内置 vs MCP（description 中带 [MCP:xxx] 前缀的是外部 MCP 工具）
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
            tool_desc_parts.append("【内置工具】\n" + "\n".join(builtin_desc))
        if mcp_desc:
            tool_desc_parts.append("【MCP 外部工具】（由外部 MCP 服务器提供，功能可能受限）\n" + "\n".join(mcp_desc))
        tool_desc = "\n\n".join(tool_desc_parts)
        # 首次绑定（后续热切换时由 _hot_swap_llm 更新）
        _runtime[bind_key] = llm.bind_tools(tools)

        class SubState(TypedDict):
            messages: Annotated[list, add_messages]

        def think(state: SubState) -> dict:
            # 运行时从 _runtime 读取当前绑定的 LLM（支持无感换模型）
            bound_llm = _runtime[bind_key]
            msgs = list(state["messages"])
            # 构造系统指令（临时，不保存到 state）
            sys_block = f"{prompt_text}\n\n## 可用工具\n{tool_desc}"
            llm_input = [HumanMessage(content=sys_block)] + msgs
            resp = bound_llm.invoke(llm_input)
            return {"messages": [resp]}

        def route(state: SubState) -> str:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            return END

        async def exec_tools(state: SubState) -> dict:
            import logging, asyncio
            last = state["messages"][-1]
            results = []
            # 工具执行期间抑制日志输出，避免和 spinner 动画混在一起
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
                            _debug_log(f"[TOOL] 开始执行: {tc['name']}({tc['args']})")
                            # MCP 工具可能挂住，加超时保护（60s）
                            r = await asyncio.wait_for(t.ainvoke(tc["args"]), timeout=60)
                            _debug_log(f"[TOOL] 执行完成: {tc['name']}, 返回类型: {type(r).__name__}")
                            # MCP 工具返回 (content, artifact) 元组，需要解包
                            if isinstance(r, tuple) and len(r) == 2:
                                content, _artifact = r
                                r = content
                            results.append(ToolMessage(content=str(r), tool_call_id=tc["id"]))
                        except asyncio.TimeoutError:
                            results.append(ToolMessage(content=f"工具执行超时（60s）: {tc['name']}", tool_call_id=tc["id"]))
                        except Exception as e:
                            _debug_log(f"[TOOL] 执行错误: {tc['name']}: {e}")
                            results.append(ToolMessage(content=f"工具执行错误: {e}", tool_call_id=tc["id"]))
                    else:
                        results.append(ToolMessage(content=f"未知工具: {tc['name']}", tool_call_id=tc["id"]))
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
    chat_agent = _make_sub_agent("bind_chat", MAIN_TOOLS, MAIN_PROMPT)
    coder_agent = _make_sub_agent("bind_coder", CODER_TOOLS, CODER_PROMPT)
    planner_agent = _make_sub_agent("bind_planner", PLANNER_TOOLS, PLANNER_PROMPT)

    # ── 自定义 reducer：add_messages + MicroCompact + 清理孤儿 tool_calls ──
    from tools.context_manager import micro_compact

    def _safe_add_messages(existing, new):
        combined = add_messages(existing, new)
        combined = _sanitize_messages(combined)
        combined = micro_compact(combined)      # Layer 1: 清除旧工具输出
        return combined

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
        # 运行时从 _runtime 读取当前 LLM（支持无感换模型）
        current_llm = _runtime["llm"]
        last = state["messages"][-1]
        content = getattr(last, "content", "") or str(last)
        if "[用户消息]" in content:
            content = content.split("[用户消息]")[-1].strip()[:500]

        try:
            # MiniMax 不支持 system 角色，合并到 HumanMessage
            resp = current_llm.invoke([
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

    return g, HumanMessage, ToolMessage, llm


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ 命令层：斜杠命令处理器
# ══════════════════════════════════════════════════════════════════════════════

def _rebuild_graph_sync():
    """同步重建图（在后台线程中调用）"""
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


async def _rebuild_graph():
    """热切换模型后重建 Multi-Agent 图（不中断对话）"""
    new_model = _model_registry.active_name
    if new_model == _runtime.get("model_name"):
        return  # 同一个模型，不需要重建

    cfg = _model_registry.active_config
    print(f"  {_A['d']}🔄 正在重建 Multi-Agent 图（{cfg['id']}）...{_A['0']}")
    _runtime["ready"].clear()
    t = threading.Thread(target=_rebuild_graph_sync, daemon=True)
    t.start()

    # 等待重建完成（带 spinner）
    start_spinner("Rebuilding")
    _runtime["ready"].wait()
    stop_spinner()

    if _runtime["error"]:
        print(f"  {_A['r']}✗ 重建失败: {_runtime['error']}{_A['0']}")
        return

    # 重新编译（带 checkpointer）
    if _runtime["ck_ctx"] is not None:
        g = _runtime["graph"]
        _runtime["graph"] = g.compile(checkpointer=_runtime["ck_ctx"])

    print(f"  {_A['g']}✅ Multi-Agent 图已重建，新对话立即生效{_A['0']}")


def _is_ollama() -> bool:
    """检测当前模型是否是 Ollama 本地模型"""
    cfg = _model_registry.active_config
    base = cfg.get("base", "")
    return "localhost:11434" in base or "127.0.0.1:11434" in base


# 预热防重入：后台加载 + _turn 同时触发时排队等
_warmup_lock = threading.Lock()


def _warmup_ollama(quiet: bool = False):
    """Ollama 模型预热：确保模型已加载，避免对话请求挂死

    Ollama 默认 keep_alive=5m，模型空闲后自动卸载。
    本函数通过 Ollama 原生 API /api/chat 发一个极短请求来触发加载。
    模型已在内存时 <1s 返回，成本可忽略。

    防重入：如果后台预热正在进行，后到的调用等锁释放后再发请求
    （此时模型已被前一次加载到内存，秒回）。

    Args:
        quiet: True 时不操作 spinner/打印（用于嵌入 _turn 的 Thinking 阶段）
    """
    import httpx

    got = _warmup_lock.acquire(timeout=200)  # 等前面的预热完成
    if not got:
        return
    try:
        cfg = _model_registry.active_config
        model_id = cfg.get("id", "")
        base = cfg.get("base", "")
        ollama_base = base.replace("/v1", "").rstrip("/")

        if not quiet:
            start_spinner("Loading Model")
        t0 = time.time()
        try:
            resp = httpx.post(
                f"{ollama_base}/api/chat",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=180,
            )
            elapsed = time.time() - t0
            if not quiet:
                stop_spinner()
                if resp.status_code == 200:
                    print(f"  {_A['g']}✅ 模型已就绪{_A['0']} {_A['d']}({elapsed:.1f}s){_A['0']}")
                else:
                    print(f"  {_A['y']}⚠ Ollama 返回 {resp.status_code}，继续尝试...{_A['0']}")
        except httpx.TimeoutException:
            if not quiet:
                stop_spinner()
            elapsed = time.time() - t0
            print(f"\n  {_A['r']}✗ 模型加载超时（{elapsed:.0f}s），请检查 Ollama 状态{_A['0']}")
            print(f"  {_A['d']}提示: 运行 ollama ps 查看模型状态{_A['0']}")
            raise RuntimeError(f"Ollama 模型 {model_id} 加载超时")
        except Exception as e:
            if not quiet:
                stop_spinner()
            print(f"  {_A['y']}⚠ 预热失败: {e}，继续尝试...{_A['0']}")
    finally:
        _warmup_lock.release()


def _hot_swap_llm():
    """无感换模型：只替换 LLM 引用，不重建图

    原理：图的闭包运行时从 _runtime["bind_*"] 读取当前 LLM，
    所以只需更新 _runtime 中的引用，图结构完全不变。
    耗时 < 1ms（只有对象创建 + 本地 bind_tools）。
    """
    from langchain_openai import ChatOpenAI

    cfg = _model_registry.active_config
    new_llm = ChatOpenAI(
        model_name=cfg["id"],
        openai_api_key=cfg["key"],
        openai_api_base=cfg.get("base", ""),
        request_timeout=120,  # 本地模型首次加载可能需要 1-2 分钟
    )

    # 更新 LLM 引用 + 重新绑定工具
    _runtime["llm"] = new_llm
    _runtime["bind_chat"] = new_llm.bind_tools(MAIN_TOOLS)
    _runtime["bind_coder"] = new_llm.bind_tools(CODER_TOOLS)
    _runtime["bind_planner"] = new_llm.bind_tools(PLANNER_TOOLS)
    _runtime["model_name"] = _model_registry.active_name



async def _cmd(cmd: str, mgr: SessionManager) -> bool:
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
        print(f"  {_A['g']}/models <名称|序号>{_A['0']}       切换模型（热切换，无需重启）")
        print(f"  {_A['g']}/mcp{_A['0']}                     查看 MCP 服务器状态")
        print(f"  {_A['g']}/mcp reload{_A['0']}              热重载 MCP 配置（自动重建图）")
        print(f"  {_A['g']}/mcp add <JSON>{_A['0']}          添加 MCP 服务器")
        print(f"  {_A['g']}/mcp remove <名称>{_A['0']}       删除 MCP 服务器")
        print(f"  {_A['g']}/context{_A['0']}                查看当前上下文状态（消息数/token 估算）")
        print(f"  {_A['g']}/compact{_A['0']}               手动压缩上下文（LLM 摘要旧消息）")
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
            print(f"\n  当前: {_A['g']}{active}{_A['0']} | 切换: {_A['g']}/models <名称|序号>{_A['0']}\n")
            return False
        # 切换模型 → 热切换（无感，< 1ms）
        try:
            old, new = _model_registry.switch(arg)
            cfg = _model_registry.active_config
            _hot_swap_llm()
            print(f"\n  ✅ 模型已切换: {_A['y']}{old}{_A['0']} → {_A['g']}{new}{_A['0']} ({cfg['id']})")
            if _is_ollama():
                # 本地模型后台静默加载，不阻塞用户
                threading.Thread(target=_warmup_ollama, kwargs={"quiet": True}, daemon=True).start()
                print(f"  {_A['d']}（本地模型后台加载中，首次提问会等待加载完成）{_A['0']}\n")
            else:
                print(f"  {_A['d']}（已热加载，新对话立即生效）{_A['0']}\n")
        except ValueError as e:
            print(f"\n  ✗ {e}\n")
        return False

    # ── /mcp 命令 — MCP 服务器管理 ──
    if a == "/mcp":
        from tools.mcp_loader import mcp_manager

        if not arg or arg == "status":
            # /mcp 或 /mcp status — 显示状态
            print(f"\n{mcp_manager.get_status()}\n")
            return False

        if arg == "reload":
            # /mcp reload — 热重载
            start_spinner("MCP Reloading")
            await mcp_manager.reload()
            stop_spinner()
            if mcp_manager.is_active:
                print(f"\n  {_A['g']}✅ MCP 已重载: {len(mcp_manager.tools)} 个工具{_A['0']}")
                # MCP 工具变了，需要重建图（bind_tools 需要新的工具列表）
                await _rebuild_graph()
            else:
                print(f"\n  {_A['y']}⚠ MCP 无活跃连接（检查 mcp_servers.json 配置）{_A['0']}")
            print()
            return False

        if arg.startswith("add "):
            # /mcp add <JSON> — 添加 MCP 服务器
            # 格式: /mcp add {"name": "playwright", "command": "npx", "args": ["@playwright/mcp@latest"], "transport": "stdio"}
            # 或: /mcp add playwright {"command": "npx", "args": ["@playwright/mcp@latest"], "transport": "stdio"}
            import json as _json
            add_arg = arg[4:].strip()
            try:
                # 尝试解析为: name {"config": "..."}
                parts = add_arg.split(None, 1)
                if len(parts) == 2 and parts[1].startswith("{"):
                    srv_name = parts[0]
                    srv_config = _json.loads(parts[1])
                else:
                    # 解析为完整 JSON: {"name": "...", "command": "...", ...}
                    data = _json.loads(add_arg)
                    srv_name = data.pop("name", None)
                    if not srv_name:
                        print(f"\n  {_A['r']}✗ 缺少服务器名称（添加 name 字段）{_A['0']}\n")
                        return False
                    srv_config = data
                print(f"\n  {mcp_manager.add_server(srv_name, srv_config)}")
                print(f"  {_A['d']}提示: 输入 /mcp reload 立即生效{_A['0']}\n")
            except _json.JSONDecodeError as e:
                print(f"\n  {_A['r']}✗ JSON 格式错误: {e}{_A['0']}")
                print(f"  {_A['d']}用法: /mcp add playwright {{\"command\": \"npx\", \"args\": [\"@playwright/mcp@latest\"], \"transport\": \"stdio\"}}{_A['0']}\n")
            return False

        if arg.startswith("remove "):
            # /mcp remove <名称>
            srv_name = arg[7:].strip()
            print(f"\n  {mcp_manager.remove_server(srv_name)}")
            print(f"  {_A['d']}提示: 输入 /mcp reload 立即生效{_A['0']}\n")
            return False

        print(f"\n  {_A['y']}用法: /mcp [status|reload|add|remove]{_A['0']}\n")
        return False

    # ── /context — 查看当前上下文状态 ──
    if a == "/context":
        try:
            from tools.context_manager import get_context_stats
            ck = _runtime.get("ck")
            if ck is None:
                print(f"\n  {_A['y']}上下文尚未初始化（发一条消息后可用）{_A['0']}\n")
                return False
            tid = mgr.active_id
            cfg_ctx = {"configurable": {"thread_id": tid}}
            state = await _runtime["graph"].aget_state(cfg_ctx)
            msgs = state.values.get("messages", [])
            stats = get_context_stats(msgs)
            print(f"\n  {_A['b']}{_A['c']}上下文状态:{_A['0']}")
            print(f"  {'─' * 40}")
            print(f"  消息总数:    {stats['total_messages']}")
            print(f"  ├─ 用户消息: {stats['human_messages']}")
            print(f"  ├─ AI 消息:  {stats['ai_messages']}")
            print(f"  └─ 工具输出: {stats['tool_messages']}（已压缩 {stats['cleared_messages']}）")
            est_tokens = stats['est_chars'] // 2  # 粗略估算：中文约 2 字符/token
            print(f"  估算 token:  ~{est_tokens:,}")
            # 会话记忆统计
            try:
                from tools.session_memory import session_memory as _sm
                ms = _sm.get_stats(tid)
                print(f"  会话记忆:    {ms['turn_count']} 轮 | "
                      f"{ms['key_files']} 文件 | {ms['tools_used']} 工具 | "
                      f"~{ms['memory_chars']} 字符")
            except Exception:
                pass
            # 熔断器状态
            try:
                from tools.context_manager import get_circuit_status
                cs = get_circuit_status()
                if cs["is_open"]:
                    print(f"  熔断器:      {_A['r']}⚠ 已断开（连续 {cs['consecutive_failures']} 次失败）{_A['0']}")
                else:
                    print(f"  熔断器:      {_A['g']}正常{_A['0']}（{cs['consecutive_failures']}/{cs['max_failures']}）")
            except Exception:
                pass
            # 输出预算状态
            try:
                from tools.output_budget import get_budget_info
                print(f"  输出预算:    {get_budget_info()}")
            except Exception:
                pass
            print(f"  {'─' * 40}\n")
        except Exception as e:
            print(f"\n  {_A['y']}获取上下文失败: {e}{_A['0']}\n")
        return False

    # ── /compact — 手动压缩上下文（强制执行） ──
    if a == "/compact":
        compressed = await _auto_compact(mgr.active_id, force=True)
        if not compressed:
            print(f"\n  {_A['y']}压缩失败或无需压缩{_A['0']}\n")
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


async def _turn(user_input: str, thread_id: str):
    """单轮对话：Routing → 子Agent流式 → 工具动画 → 打印"""
    tp = rp = False
    _reply_started = False

    _clear_input(user_input)
    _echo_user(user_input)
    parser = StreamParser()

    # 注入实时时间和工作目录
    from datetime import datetime
    _now = datetime.now().strftime('%Y-%m-%d %A %H:%M:%S')
    _cwd = os.environ.get("LAUNCH_DIR", "") or str(Path.cwd())
    _env = ENV_BLOCK.replace("{current_time}", _now).replace("{cwd}", _cwd)

    # 注入会话记忆（Layer 2：零成本自动追踪）
    from tools.session_memory import session_memory
    memory_block = session_memory.get_memory_block(thread_id)
    parts = [_env]
    if memory_block:
        parts.append(memory_block)
    parts.append(f"[用户消息]\n{user_input}")

    cfg = {"configurable": {"thread_id": thread_id}}

    try:
        # 立即显示 Agent 标签 + Thinking 动画
        # 后台加载 / checkpointer 编译 / 模型预热 / 推理 全部算在 Thinking 里
        print(f"{_A['c']}Agent{_A['0']}")
        start_spinner("Thinking")

        # ★ 等待后台加载完成（首次启动时 graph 还是 None）
        if not _runtime["ready"].is_set():
            await asyncio.to_thread(_runtime["ready"].wait)
            if _runtime["error"]:
                stop_spinner()
                print(f"  {_A['r']}启动失败: {_runtime['error']}{_A['0']}")
                return

        # ★ 首次编译：打开 checkpointer 并编译图
        if _runtime["ck_ctx"] is None:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            db = Path(__file__).parent.parent / "chat_history.db"
            ck_ctx = AsyncSqliteSaver.from_conn_string(str(db))
            _runtime["ck"] = await ck_ctx.__aenter__()
            _runtime["ck_ctx"] = ck_ctx
            _runtime["graph"] = _runtime["graph"].compile(checkpointer=_runtime["ck"])

        # 用最新的 graph + HM（首次编译后可能变了）
        graph = _runtime["graph"]
        HumanMessage = _runtime["HM"]

        # ★ MCP 懒加载：首次对话时同步加载（算在 Thinking 里，静默）
        if not _runtime.get("mcp_loaded"):
            try:
                from tools.mcp_loader import mcp_manager
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

        # ★ Ollama 预热：确保模型已加载（时间算在 Thinking 里，用户无感）
        if _is_ollama():
            try:
                await asyncio.to_thread(_warmup_ollama, quiet=True)
            except RuntimeError:
                stop_spinner()
                print(f"  {_A['r']}✗ 模型加载失败，请检查 Ollama 状态{_A['0']}")
                return

        # 构造输入消息（HumanMessage 在 ready 之后才有）
        inp = {
            "messages": [
                HumanMessage(content="\n\n".join(parts)),
            ],
            "next_agent": "",
        }

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
                    # 子 Agent（Coder/Planner）：上移覆盖 Agent 行，改为 Agent | Label
                    if sub_name != "chat":
                        stop_spinner()
                        # stop_spinner 输出了 "  ● Thinking（Xs）\n" 占 1 行
                        # cursor 在 Agent 下面第 2 行，需要上移 2 行才能覆盖 Agent
                        sys.stdout.write(f"\033[2A\r\033[K{_A['c']}Agent {_A['d']}| {agent_label}{_A['0']}\033[1B\n")
                        sys.stdout.flush()
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
                            sys.stdout.write(f"    {_A['d']}{_A['gr']}"); sys.stdout.flush()
                        type_text(text.replace("\n", "\n    "), 0.03)
                    else:
                        if not rp:
                            stop_spinner()
                            if tp: sys.stdout.write(_A["0"]); print()
                            rp = True
                        # 首次正式回复写缩进，后续靠 \n 替换自动缩进
                        if not _reply_started:
                            sys.stdout.write("    "); sys.stdout.flush()
                            _reply_started = True
                        type_text(text.replace("\n", "\n    "), 0.025)

            # ── 工具调用开始 ──
            elif et == "on_tool_start":
                nm = ev.get("name", "unknown")
                inp2 = ev.get("data", {}).get("input", {})
                params = "，".join(
                    f'{k}="{v[0]}"' if isinstance(v, list) and len(v) == 1
                    else f'{k}="{v}"' if isinstance(v, str)
                    else f'{k}={v}'
                    for k, v in inp2.items()
                ) if inp2 else ""
                info = f"{nm}（{params}）" if params else nm
                stop_spinner(); start_spinner("ToolCalling", info)

            # ── 工具调用结束 ──
            elif et == "on_tool_end":
                stop_spinner()
                tp = False
                parser = StreamParser()

    except (KeyboardInterrupt, asyncio.CancelledError):
        stop_spinner(); print(f"\n{_A['y']}⏹ 已打断{_A['0']}"); _interrupt.clear(); return
    except Exception as e:
        stop_spinner(); print(f"\n  [{_A['r']}错误: {e}{_A['0']}]")

    # 刷出剩余内容
    for is_t, text in parser.done():
        if is_t:
            if not tp: stop_spinner(); tp = True; sys.stdout.write(f"    {_A['d']}{_A['gr']}"); sys.stdout.flush()
            type_text(text.replace("\n", "\n    "), 0.03)
        else:
            if not rp:
                stop_spinner()
                if tp: sys.stdout.write(_A["0"]); print()
                rp = True
            sys.stdout.write("    "); sys.stdout.flush()
            type_text(text.replace("\n", "\n    "), 0.025)

    if not tp and not rp: stop_spinner(); print("  (无回复)")
    elif tp and not rp: sys.stdout.write(_A["0"])
    print()

# ══════════════════════════════════════════════════════════════════════════════
# ⑦ 入口层：主程序（Banner 先行，重型导入后台加载）
# ══════════════════════════════════════════════════════════════════════════════

async def _track_memory(thread_id: str):
    """对话结束后更新会话记忆（Layer 2：零成本自动追踪）"""
    try:
        from tools.session_memory import session_memory
        cfg = {"configurable": {"thread_id": thread_id}}
        state = await _runtime["graph"].aget_state(cfg)
        msgs = state.values.get("messages", [])
        # 只传最近 8 条消息，避免重复处理
        session_memory.track_from_messages(thread_id, msgs[-8:])
    except Exception:
        pass  # 记忆追踪失败不影响主流程


async def _auto_compact(thread_id: str, *, force: bool = False) -> bool:
    """自动检测上下文大小，超过阈值时触发 Layer 3 摘要压缩

    Args:
        force: 为 True 时跳过阈值检查，强制压缩（供 /compact 命令使用）

    Returns:
        True 表示执行了压缩
    """
    from tools.context_manager import (
        get_context_stats, COMPACT_THRESHOLD,
        is_compact_circuit_open, record_compact_failure, record_compact_success,
    )
    from langgraph.graph.message import add_messages

    # 熔断器检查（强制模式跳过）
    if not force and is_compact_circuit_open():
        return False

    # 冷却：距离上次成功压缩不足 3 轮时跳过（避免无限压缩循环）
    if not force:
        _compact_cooldown[0] -= 1
        if _compact_cooldown[0] > 0:
            return False

    try:
        graph = _runtime["graph"]
        # 确保图已编译（未编译的 StateGraph 没有 aget_state）
        if not hasattr(graph, 'aget_state'):
            _debug_log("[COMPACT] 图未编译，跳过")
            return False
        cfg = {"configurable": {"thread_id": thread_id}}
        state = await graph.aget_state(cfg)
        msgs = state.values.get("messages", [])
        if not msgs:
            return False

        stats = get_context_stats(msgs)
        est_tokens = stats["est_chars"] // 2

        # 低于阈值，不需要压缩（强制模式跳过）
        if not force and est_tokens < COMPACT_THRESHOLD:
            return False

        # 需要压缩
        print(f"  {_A['y']}⏳ 上下文 ~{est_tokens:,} tokens，正在压缩...{_A['0']}")
        start_spinner("Compacting")

        from tools.context_manager import summarize_and_compact
        llm = _runtime.get("llm")
        if not llm:
            stop_spinner()
            return False

        compressed, summary = await summarize_and_compact(msgs, llm)

        stop_spinner()

        if summary is None:
            record_compact_failure()
            return False

        # 用 LangGraph 的 state update 替换消息
        # 策略：逐条删除，跳过 checkpoint 中不存在的 ID
        from langgraph.graph.message import RemoveMessage
        remove_ops = [RemoveMessage(id=m.id) for m in msgs if getattr(m, 'id', None)]
        _debug_log(f"[COMPACT] 尝试删除 {len(remove_ops)} 条消息")
        try:
            await graph.aupdate_state(cfg, {"messages": remove_ops})
        except Exception as del_err:
            # 如果批量删除失败（某些 ID 不存在于 checkpoint），逐条尝试
            _debug_log(f"[COMPACT] 批量删除失败: {del_err}，改为逐条删除")
            for op in remove_ops:
                try:
                    await graph.aupdate_state(cfg, {"messages": [op]})
                except Exception:
                    _debug_log(f"[COMPACT] 跳过无效 ID: {op.id}")
        await graph.aupdate_state(cfg, {"messages": compressed})

        # Step 4: 压缩后重建 — 重新注入近期文件上下文
        try:
            from tools.session_memory import session_memory as _sm
            file_context = _sm.get_recent_file_context(thread_id)
            if file_context:
                from langchain_core.messages import HumanMessage as _HM
                ctx_msg = _HM(content=f"[压缩恢复 — 近期文件上下文]\n\n{file_context}")
                await graph.aupdate_state(cfg, {"messages": [ctx_msg]})
        except Exception:
            pass  # 文件重建失败不影响主流程

        new_stats = get_context_stats(compressed)
        new_tokens = new_stats["est_chars"] // 2
        saved_pct = (1 - new_stats["est_chars"] / max(stats["est_chars"], 1)) * 100

        record_compact_success()
        _compact_cooldown[0] = 3  # 成功后冷却 3 轮
        print(f"  {_A['g']}✅ 压缩完成：{stats['total_messages']} → {new_stats['total_messages']} 条消息 "
              f"| ~{est_tokens:,} → ~{new_tokens:,} tokens（节省 {saved_pct:.0f}%）{_A['0']}")
        return True

    except Exception as e:
        stop_spinner()
        record_compact_failure()
        print(f"  {_A['y']}⚠ 压缩失败: {e}{_A['0']}")
        return False


async def run_chat_loop(mgr: SessionManager):
    while True:
        try:
            ui = input(f"\r{_A['g']}> {_A['0']}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_A['c']}再见！👋{_A['0']}")
            break
        if not ui:
            continue
        if ui.startswith("/"):
            if await _cmd(ui, mgr): print(f"{_A['c']}再见！👋{_A['0']}"); break
            continue
        await _auto_compact(mgr.active_id)
        await _turn(ui, mgr.active_id)
        mgr.touch()
        await _track_memory(mgr.active_id)

async def main():
    # Banner
    mn = _model_registry.active_name
    mid = _model_registry.active_config.get("id", "?")
    _launch_dir = os.environ.get("LAUNCH_DIR", None)
    _actual_cwd = str(Path.cwd())
    _display_cwd = _launch_dir or _actual_cwd
    print(f"\n╔{'═' * 50}╗")
    print(f"║  🤖 {_A['b']}My Agent — 2.0.0{_A['0']} (Multi-Agent){' ' * 15}║")
    print(f"╚{'═' * 50}╝")
    print(f"  📡 {_A['c']}{mn}{_A['0']} · {mid}")
    print(f"  📂 {_A['d']}{_display_cwd}{_A['0']}")

    # ★ MCP 懒加载在 _turn() 内部（首次 Thinking 时同步加载）

    print()

    # 后台静默加载重型模块
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
    await run_chat_loop(mgr)

    # 清理
    if _runtime["ck_ctx"] is not None:
        await _runtime["ck_ctx"].__aexit__(None, None, None)
    # 清理 MCP 连接
    try:
        from tools.mcp_loader import mcp_manager
        await mcp_manager._close()
    except Exception:
        pass

if __name__ == "__main__":
    asyncio.run(main())
