# chat_agent.py - 钳钳 Agent (流式版 + LangGraph 循环)
"""钳钳 Agent - 流式输出 + 思考过程展示 + LangGraph 工具循环

═══════════════════════════════════════════════════════════════
 技术选型对比
═══════════════════════════════════════════════════════════════

 ┌────────────────┬──────────────────┬──────────────────────────┐
 │ 本项目使用      │ 替代方案          │ 企业级首选               │
 ├────────────────┼──────────────────┼──────────────────────────┤
 │ LangGraph      │ AutoGen, CrewAI  │ LangGraph / Dify         │
 │ LangChain      │ LlamaIndex, Haystack │ LangChain + LangSmith │
 │ ChromaDB       │ FAISS, Milvus    │ Pinecone / Weaviate      │
 │ BM25 (rank_bm25)│ Elasticsearch   │ Elasticsearch / Meilisearch│
 │ ChatOpenAI     │ litellm, instructor │ litellm / httpx 直接调用│
 │ asyncio        │ trio, curio      │ asyncio（标准库，最稳定） │
 │ threading      │ multiprocessing  │ asyncio（IO密集型优先）   │
 └────────────────┴──────────────────┴──────────────────────────┘

 为什么选 LangGraph：
   - LangChain 团队官方出品，生态集成最好
   - StateGraph 比 CrewAI 的 "Agent 对话" 模式更灵活，能精确控制流程
   - 原生支持 conditional_edges（条件路由），适合"思考→调用工具→再思考"循环
   - 企业用 Dify（可视化编排）时，底层也是类似的 DAG 思路

 为什么选 asyncio：
   - astream_events 是异步生成器，必须用 async for 遍历
   - 不能用 sync for（会报 'async_generator' object is not iterable）
   - 不能用 multiprocessing（GIL 限制 + 进程间通信复杂）

 为什么用 threading 做动画：
   - 动画是纯 IO（print），asyncio 也能做，但需要和 astream_events 共享事件循环
   - threading 更简单：独立线程跑动画，主线程处理 LLM 流式输出
   - daemon=True 确保主程序退出时动画线程自动终止
"""

# ═══════════════════════════════════════════════════════════════
# 环境配置（必须在所有 import 之前）
# ═══════════════════════════════════════════════════════════════

import os
import sys

# ★ HuggingFace 镜像 — 为什么放在最前面？
#   Python 的 import 是一次性操作，模块内的全局变量在 import 时就初始化了
#   langchain_huggingface 内部会在 import 时读取 HF_ENDPOINT
#   如果放在 import 之后设置，模块已经用默认值初始化完毕，设置不生效
#   类比：就像你要在咖啡店开门前贴"今日特惠"，开门后再贴就来不及了
#
#   企业级做法：
#     - 用 .env 文件 + python-dotenv（python-dotenv.load_dotenv()）
#     - 或 Docker 环境变量（docker-compose.yml 的 environment 字段）
#     - 或 Kubernetes ConfigMap/Secret
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像，避免连不上 huggingface.co
os.environ["HF_HUB_HTTP_TIMEOUT"] = "300"  # 5分钟超时，防止大模型下载中断

import asyncio   # Python 标准库异步框架。替代：trio（更简洁但不主流）、curio
import time      # 标准库，用于 sleep 和计时
import random    # 标准库，用于打字动画的随机延迟
import threading # 标准库，用于 Thinking 动画的独立线程

# 将项目根目录加入 sys.path，使得 `from config.settings` 等导入能正常工作
# os.path.abspath(__file__) = .../langgraph-learn/agents/chat_agent.py
# dirname 一次 = .../langgraph-learn/agents/
# dirname 两次 = .../langgraph-learn/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════════════
# 类型系统
# ═══════════════════════════════════════════════════════════════

# Annotated: Python 3.9+ 的类型注解增强，可以给类型附加元数据
#   这里用来给 LangGraph 的 StateGraph 声明 "这个 list 字段用 add_messages 函数合并"
#   替代：不用 Annotated 的话，需要手写 reducer 函数
from typing import Annotated

# TypedDict: Python 3.8+ 的类型工具，定义固定 key 的字典类型
#   用于 LangGraph 的 StateGraph 的状态定义
#   替代：dataclass（但 LangGraph 要求 TypedDict 或 Pydantic Model）
#   企业级：Pydantic BaseModel（自带数据验证、序列化）
from typing_extensions import TypedDict

# ═══════════════════════════════════════════════════════════════
# LangChain 核心组件
# ═══════════════════════════════════════════════════════════════

# LangChain 消息类型 — 为什么不用普通 dict？
#   LangChain 的消息链路（LLM → Tool → LLM）需要统一的 Message 对象
#   每种消息有不同角色：System（系统提示）、Human（用户）、AI（助手）、Tool（工具返回）
#   如果用 dict，LLM 无法区分消息来源，工具调用会失败
from langchain_core.messages import (
    ToolMessage,    # 工具执行结果，必须带 tool_call_id 关联回 AI 的调用请求
    SystemMessage,  # 系统提示词，定义 Agent 的角色和行为
    HumanMessage,   # 用户输入
    AIMessage,      # AI 回复（含 tool_calls 字段）
)

# @tool 装饰器 — 为什么用它？
#   把普通 Python 函数变成 LangChain Tool 对象
#   自动提取函数签名（参数名、类型）和 docstring 作为工具描述
#   LLM 根据描述决定调用哪个工具、传什么参数
#   替代：手动构造 Tool(name=..., description=..., func=...) — 繁琐且容易出错
#   企业级：LangSmith + Tool Trace（工具调用的可观测性）
from langchain_core.tools import tool

# ChatOpenAI — 为什么用 langchain_openai 而不是 openai 官方 SDK？
#   1. openai 官方 SDK 只支持 OpenAI 自家模型
#   2. langchain_openai 通过 openai_api_base 参数支持任何 OpenAI 兼容 API
#      （如 MiniMax、DeepSeek、Ollama、vLLM 等国产/私有模型）
#   3. 自带 .bind_tools() 方法，一行代码就能把工具注册给模型
#   4. 自带 .stream() 方法，支持逐 token 流式输出
#   替代：
#     - litellm：统一代理 100+ 模型提供商，企业级首选
#     - httpx 直接调用：最灵活但要自己处理重试、超时、错误码
from langchain_openai import ChatOpenAI

# ═══════════════════════════════════════════════════════════════
# LangGraph — Agent 编排框架
# ═══════════════════════════════════════════════════════════════

# StateGraph: LangGraph 的核心类，定义 Agent 的执行图（有向无环图 + 条件循环）
#   为什么不用纯 Python while 循环？
#     - while 循环也能实现"思考→工具→再思考"，但：
#       1. 难以可视化（LangGraph 可以生成流程图）
#       2. 难以扩展（加节点只需 add_node，不用改主循环）
#       3. 难以调试（LangGraph 每步都有状态快照）
#       4. 难以持久化（LangGraph 自带 checkpointer 支持断点续跑）
#   替代：
#     - CrewAI：多 Agent 协作框架，适合 "研究员+写手+审稿人" 场景
#     - AutoGen：微软出品，多 Agent 对话模式，适合需要多轮协商的任务
#     - 自研状态机：灵活但开发量大
#   企业级：
#     - LangGraph + LangSmith（全链路追踪 + 评估）
#     - Dify（可视化 Agent 编排，非程序员也能用）
from langgraph.graph import StateGraph, START, END

# add_messages: LangGraph 内置的 reducer 函数
#   作用：当 StateGraph 节点返回 {"messages": [新消息]} 时，
#   不是替换旧消息列表，而是追加到末尾
#   这就是为什么 ai_think 返回 {"messages": [response]} 不会覆盖之前的消息
from langgraph.graph.message import add_messages

# ═══════════════════════════════════════════════════════════════
# 项目内部模块
# ═══════════════════════════════════════════════════════════════

from config.settings import API_KEY, API_BASE, MODEL_FAST
from tools.rag_tool import rag_search as _rag_search_raw, ingest as _ingest_raw
from tools.weather_tool import get_weather as _get_weather_raw


# ═══════════════════════════════════════════════════════════════
# 思考标签配置（按模型修改）
# ═══════════════════════════════════════════════════════════════
# 为什么需要思考标签？
#   部分 LLM（如 DeepSeek、MiniMax）会在回复前先输出 <think)...(think)> 包裹的思考过程
#   这些思考内容对用户有参考价值，但不应和正式回复混在一起
#   StreamParser 会把 <think)...(think)> 内的内容标记为"思考"，外部标记为"回复"
#
# 为什么用 "<" + "think" + ">" 而不是直接写 "<think"？
#   避免某些编辑器/Linter 误认为是 HTML/XML 标签而触发警告

THINK_OPEN = "<" + "think" + ">"   # 思考开始标签
THINK_CLOSE = "</" + "think" + ">" # 思考结束标签
MAX_BUFFER = 30  # 超过 30 字符还没找到标签 → 强制当作回复内容（兼容无思考标签的模型）

SYSTEM_PROMPT = """你是"钳钳"，一个活泼开朗、风趣幽默的AI助手。性格特点：
- 活泼开朗，喜欢用emoji表达情感
- 回答问题时认真负责，但保持幽默风格
- 喜欢用比喻和类比来解释复杂概念
- 对技术话题特别感兴趣

你可以使用工具来回答问题。当用户问的问题可能需要查资料时，主动使用工具。"""


# ═══════════════════════════════════════════════════════════════
# 工具定义 — Agent 的"双手"
# ═══════════════════════════════════════════════════════════════
# @tool 装饰器做了什么？
#   1. 读取函数签名 → 生成 JSON Schema（LLM 用来知道传什么参数）
#   2. 读取 docstring → 生成工具描述（LLM 用来决定什么时候调用）
#   3. 包装成 BaseTool 对象 → 可以被 bind_tools() 绑定到 LLM
#
# 企业级工具设计原则：
#   1. docstring 要精确：说明什么时候用、什么时候不用
#   2. 参数要有类型注解：LLM 根据类型传参（str vs int vs list）
#   3. 返回值要简洁：太长的结果会浪费 token，截取关键信息
#   4. 加安全校验：路径遍历、SQL 注入等（见 add_knowledge 的校验）


@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    return _get_weather_raw(city)


@tool
def rag_search(query: str) -> str:
    """搜索游梦的个人知识库"""
    return _rag_search_raw(query)


@tool
def add_knowledge(folder_path: str) -> str:
    """将文件夹里的文档加入知识库"""
    # 安全校验：只允许绝对路径，防止 LLM 传入相对路径导致意外访问
    if not os.path.isabs(folder_path):
        return "请提供绝对路径"
    if not os.path.isdir(folder_path):
        return f"目录不存在: {folder_path}"
    _ingest_raw(folder_path, force=True)
    return f"入库完成！现在可以问我关于这些文档的问题了。"


# 工具列表 — 传给 bind_tools()，LLM 会看到所有可用工具
tools = [get_weather, rag_search, add_knowledge]


# ═══════════════════════════════════════════════════════════════
# LangGraph 状态与节点
# ═══════════════════════════════════════════════════════════════
# LangGraph 的核心概念：
#
#   1. State（状态）— 所有节点共享的数据
#      就像一条流水线上的传送带，每个工位（节点）都能读取和追加
#
#   2. Node（节点）— 处理逻辑的函数
#      接收 State，返回 State 的更新（增量）
#
#   3. Edge（边）— 节点之间的连接
#      普通边：A → B（固定走向）
#      条件边：A → B 或 A → C（根据返回值决定走向）
#
# 本项目的流程图：
#
#   START → ai_think → should_use_tool?
#                        ├── "done"     → END
#                        └── "use_tool" → execute_tool → ai_think（循环）
#
#   这就是经典的 ReAct (Reason + Act) 模式：
#     Reason: ai_think（LLM 思考下一步做什么）
#     Act:    execute_tool（执行工具）
#     Observe: 工具结果自动追加到 messages，LLM 看到后再次思考

class AgentState(TypedDict):
    """LangGraph 状态定义

    messages 字段用了 Annotated[list, add_messages]：
      - 普通的 dict 赋值是覆盖：state["messages"] = [新消息] → 旧消息没了
      - 加了 add_messages reducer 后是追加：新消息会被 append 到旧列表末尾
      - 这保证了每轮 LLM 调用都能看到完整的对话历史
    """
    messages: Annotated[list, add_messages]


# LLM 实例 — 为什么在模块级别创建而不是函数内部？
#   1. 避免重复创建（HTTP 连接池、模型配置只需要初始化一次）
#   2. LangGraph 的 compiled graph 在模块加载时就构建，需要引用 llm_model
#   企业级：用依赖注入（FastAPI 的 Depends）或配置类
llm_model = ChatOpenAI(
    model_name=MODEL_FAST,
    openai_api_key=API_KEY,
    openai_api_base=API_BASE,  # 指向 MiniMax / DeepSeek 等兼容 API
)

# bind_tools() 做了什么？
#   把工具的 JSON Schema 注册到 LLM 的请求参数中
#   LLM 每次调用时都会看到可用工具列表，根据需要选择调用
#   如果 LLM 决定调用工具，返回的 AIMessage 会包含 tool_calls 字段
llm_with_tools = llm_model.bind_tools(tools)


def ai_think(state: AgentState) -> dict:
    """AI 思考节点

    接收完整的消息历史（含之前的工具结果），调用 LLM 获取下一步动作。

    返回的 {"messages": [response]} 不是替换，而是追加（因为有 add_messages reducer）。
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def execute_tool(state: AgentState) -> dict:
    """工具执行节点

    从最后一条 AIMessage 中提取 tool_calls，逐个执行，返回 ToolMessage 列表。

    为什么需要 tool_call_id？
      LLM 可以一次请求调用多个工具（parallel tool calls）
      每个 tool_call 有唯一 id，ToolMessage 必须关联回对应的 id
      否则 LLM 无法把工具结果和自己的请求配对
    """
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    tool_messages = []
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_function = {t.name: t for t in tools}[tool_name]
        result = tool_function.invoke(tool_args)
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

    return {"messages": tool_messages}


def should_use_tool(state: AgentState) -> str:
    """条件路由：判断 LLM 是想调用工具还是直接回复

    返回值对应 add_conditional_edges 的映射表：
      "use_tool" → execute_tool 节点
      "done"     → END（对话结束）
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tool"
    return "done"


# ── 组装 LangGraph 状态图 ──
#
# add_edge(A, B):            固定边，A 执行完必定走到 B
# add_conditional_edges(A, fn, mapping): 条件边，根据 fn 返回值查 mapping 决定走向
# add_node(name, fn):        注册节点，fn 是处理函数
# compile():                 编译为可执行的 Runnable，支持 invoke / stream / astream_events
_builder = StateGraph(AgentState)
_builder.add_node("ai_think", ai_think)          # 节点1：LLM 思考
_builder.add_node("execute_tool", execute_tool)    # 节点2：工具执行
_builder.add_edge(START, "ai_think")               # 入口 → 先思考
_builder.add_conditional_edges(
    "ai_think",                                     # 从思考节点出发
    should_use_tool,                                # 路由函数
    {"use_tool": "execute_tool", "done": END},     # 映射表
)
_builder.add_edge("execute_tool", "ai_think")       # 工具执行完 → 回到思考
_graph = _builder.compile()                          # 编译成可执行的图


# ═══════════════════════════════════════════════════════════════
# 流式标签解析器 StreamParser
# ═══════════════════════════════════════════════════════════════
# 为什么需要这个？
#
# LLM 流式输出时，一个 <think...> 标签可能被切分成多个 chunk：
#   chunk1: "我需要<thi"
#   chunk2: "nk>分析一下..."
#   chunk3: "用户意图</th"
#   chunk4: "ink>你好！"
#
# 如果直接逐 chunk 检测标签，"thi" + "nk" 这种切分会导致标签丢失
# StreamParser 用 buffer 累积 + 滑动窗口的方式可靠地处理任意切分
#
# 工作原理：
#   1. chunk 到达 → 追加到 buffer
#   2. _drain() 扫描 buffer 中的标签
#   3. 找到完整标签 → 把之前的内容放入队列（标记为 thinking 或 reply）
#   4. 没找到 → 只释放"安全"部分（不会被标签切分的前缀），剩余留着等更多 chunk
#   5. _flush() 返回队列中的内容给调用方

class StreamParser:
    """可靠处理标签被切分的流式解析器"""

    def __init__(self):
        self.buffer = ""         # 累积未处理的文本
        self.in_thinking = False  # 当前是否在 <think...> 标签内
        self._queue = []          # 输出队列：[(is_thinking, text), ...]

    def feed(self, chunk: str):
        """喂入一个 chunk，返回可立即输出的文本列表"""
        self.buffer += chunk
        self._drain()
        return self._flush()

    def done(self):
        """流结束时，把 buffer 中剩余内容全部输出"""
        if self.buffer:
            self._queue.append((self.in_thinking, self.buffer))
            self.buffer = ""
        return self._flush()

    def _drain(self):
        """扫描 buffer，提取可安全输出的文本"""
        while True:
            if self.in_thinking:
                # 在思考模式内：找 </think...> 结束标签
                idx = self.buffer.find(THINK_CLOSE)
                if idx >= 0:
                    # 找到结束标签 → 标签前的内容是思考内容
                    before = self.buffer[:idx]
                    if before:
                        self._queue.append((True, before))
                    self.buffer = self.buffer[idx + len(THINK_CLOSE):]
                    self.in_thinking = False
                else:
                    # 没找到 → 释放"安全"部分（保留结束标签长度的余量）
                    safe = max(0, len(self.buffer) - len(THINK_CLOSE))
                    if safe > 0:
                        self._queue.append((True, self.buffer[:safe]))
                        self.buffer = self.buffer[safe:]
                    break
            else:
                # 在回复模式内：找 <think...> 开始标签
                idx = self.buffer.find(THINK_OPEN)
                if idx >= 0:
                    before = self.buffer[:idx]
                    if before:
                        self._queue.append((False, before))
                    self.buffer = self.buffer[idx + len(THINK_OPEN):]
                    self.in_thinking = True
                else:
                    # 没找到开始标签
                    if len(self.buffer) > MAX_BUFFER:
                        # 超过 MAX_BUFFER 还没出现标签 → 大概率不是思考标签
                        # （兼容 GLM 等不带思考标签的模型）
                        self._queue.append((False, self.buffer))
                        self.buffer = ""
                        break
                    # 释放安全部分，保留开始标签长度的余量
                    safe = max(0, len(self.buffer) - len(THINK_OPEN))
                    if safe > 0:
                        self._queue.append((False, self.buffer[:safe]))
                        self.buffer = self.buffer[safe:]
                    break

    def _flush(self):
        """返回并清空输出队列"""
        out = self._queue[:]
        self._queue.clear()
        return out


# ═══════════════════════════════════════════════════════════════
# UI 辅助 — Thinking 动画 + 逐字打印
# ═══════════════════════════════════════════════════════════════

# threading.Event: 线程安全的"信号灯"
#   set() → 绿灯（停止等待）
#   clear() → 红灯（继续等待）
#   is_set() → 查看当前状态
#   替代：asyncio.Event（但动画线程是独立线程，不用 asyncio）
thinking_stop = threading.Event()

# 动画帧：所有帧等宽（11字符），用空格补齐
# 为什么必须等宽？
#   \b（backspace）回退的字符数是固定的（回退 11 个）
#   如果帧宽度不同：
#     "Thinking." (9字符) → 回退 11 → 多回退 2 个 → 残留上一帧的 2 个字符
#     导致 "Thinking." 变成 "Thinking.." 或 "TThinking."
#   等宽后每帧都是 11 字符，回退 11 个刚好覆盖干净
_THINKING_FRAMES = ["Thinking   ", "Thinking.  ", "Thinking.. ", "Thinking..."]
_THINKING_MAX_LEN = len(_THINKING_FRAMES[-1])  # 11


def thinking_loop():
    """后台线程：Thinking 动画（固定宽度 + \\b 回退）

    为什么用 \\b 而不是 \\r（回车）？
      \\r 在 Windows 终端的行为不一致：
        - cmd.exe：光标回到行首，可以覆盖
        - PowerShell：光标不动或行为怪异
        - Windows Terminal：有时有效有时无效
      \\b（退格）是标准 ASCII 控制字符（0x08），所有终端都支持

    为什么不用 \\x1b[2K（ANSI 清行）？
      虽然更优雅，但 Windows 旧版 cmd 不支持 ANSI 转义序列
      （Windows 10+ 需要 EnableVirtualTerminalProcessing）
    """
    i = 0
    # 先打印初始帧（否则会有短暂空白）
    sys.stdout.write(_THINKING_FRAMES[0])
    sys.stdout.flush()
    while not thinking_stop.is_set():
        time.sleep(0.4)
        i += 1
        frame = _THINKING_FRAMES[i % len(_THINKING_FRAMES)]
        # 回退最大宽度 + 写新帧 = 覆盖式动画
        sys.stdout.write("\b" * _THINKING_MAX_LEN + frame)
        sys.stdout.flush()


def stop_thinking(thinking_th):
    """安全停止动画线程

    为什么不擦行？
      之前的版本在停止时用空格覆盖 Thinking 文字 → 用户反馈"思考完 Thinking 消失了"
      现在保留 Thinking 文字，由后续的 print() 或换行自然覆盖
      用户体验：看到 Thinking... → 思考内容 → Agent > 回复
    """
    if not thinking_stop.is_set():
        thinking_stop.set()
        thinking_th.join(timeout=2)  # 等线程退出，最多 2 秒（防止死锁）
        print()  # Thinking 后换行，让后续内容在新行显示


def type_text(text: str, base_delay: float = 0.025):
    """逐字打印（打字机效果）

    为什么加随机延迟？
      固定延迟看起来像机器人，加 random.random() 的微小抖动更自然
      base_delay * (0.5 + random) → 实际延迟在 0.5x ~ 1.5x 之间波动
    """
    for char in text:
        print(char, end="", flush=True)  # flush=True：立即显示，不等缓冲区满
        time.sleep(base_delay * (0.5 + random.random()))


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 50)
    print("  Agent (流式版 + LangGraph 循环, quit 退出)")
    print("=" * 50)

    messages_history = []  # 多轮对话历史（保留之前的 Human + AI 消息）

    while True:
        try:
            user_input = input("\nUser > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input in ("quit", "exit", "q"):
            print("再见!")
            break
        if user_input == "clear":
            os.system("cls")
            continue

        print()

        # ── 启动 Thinking 动画 ──
        # 每轮对话开始前重置状态，启动新的动画线程
        thinking_stop.clear()
        thinking_th = threading.Thread(target=thinking_loop, daemon=True)
        thinking_th.start()
        # daemon=True：主程序退出时动画线程自动终止，不会卡住

        # 每轮对话独立的解析器和状态
        parser = StreamParser()
        thinking_printed = False  # 是否已经打印过思考内容
        response_started = False  # 是否已经开始打印正式回复

        # 构建消息列表（含历史）
        # LangGraph 每次调用是独立的推理过程，不会记住上一轮
        # 所以要把历史消息都传进去，LLM 才有上下文
        full_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]
        for msg in messages_history:
            full_messages.append(msg)

        # ── LangGraph 流式事件循环 ──
        # astream_events 返回的事件类型：
        #   on_chat_model_start:  LLM 开始新一轮调用
        #   on_chat_model_stream: LLM 逐 token 输出
        #   on_tool_start:        工具开始执行
        #   on_tool_end:          工具执行完毕
        #
        # 为什么用 asyncio.run() 包装？
        #   astream_events 是 async generator（异步生成器）
        #   必须用 async for 遍历，不能直接 for
        #   asyncio.run() 创建一个临时事件循环来运行异步函数
        #   替代：把 main() 改成 async def main()，用 asyncio.run(main())
        #         但这样 input() 也需要用 asyncio.to_thread 包装，更复杂
        try:
            async def consume_events():
                nonlocal thinking_printed, response_started, parser, thinking_th
                # nonlocal 的作用：
                #   consume_events 是嵌套函数，要修改外层的局部变量
                #   没有 nonlocal → Python 认为赋值操作创建的是局部变量
                #   读取时局部变量还没赋值 → UnboundLocalError
                llm_round = 0  # 追踪 LLM 调用轮次（第1轮 / 工具后的第2轮...）

                async for event in _graph.astream_events(
                    {"messages": full_messages},
                    version="v2",  # v2 是 LangGraph 推荐的事件格式
                ):
                    event_type = event.get("event", "")

                    # ── LLM 新一轮思考开始 ──
                    if event_type == "on_chat_model_start":
                        # 第2轮及以后：工具执行完毕后，LLM 会再次思考
                        # 需要重新启动动画线程（之前的已经 stop 了）
                        if llm_round > 0:
                            thinking_stop.clear()
                            thinking_th = threading.Thread(
                                target=thinking_loop, daemon=True
                            )
                            thinking_th.start()
                        llm_round += 1

                    # ── LLM 流式输出 token ──
                    elif event_type == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        content = getattr(chunk, "content", "") or ""
                        if not content:
                            continue

                        # 用 StreamParser 解析思考/回复边界
                        for is_thinking, text in parser.feed(content):
                            if is_thinking:
                                # 思考内容：停动画 + 打印
                                if not thinking_printed:
                                    stop_thinking(thinking_th)
                                    thinking_printed = True
                                type_text(text, 0.03)

                            else:
                                # 正式回复：每轮都独立输出 "Agent > "
                                if not response_started:
                                    stop_thinking(thinking_th)
                                    if thinking_printed:
                                        print()
                                    print("Agent > ", end="", flush=True)
                                    response_started = True
                                type_text(text, 0.025)

                    # ── 工具执行开始 ──
                    elif event_type == "on_tool_start":
                        inp = event["data"].get("input", {})
                        tool_name = (
                            inp.get("name", "")
                            if isinstance(inp, dict) else str(inp)
                        )
                        print(f"\n\n  \U0001f527 调用工具: {tool_name}")

                    # ── 工具执行结束 ──
                    # ★ 关键：重置所有状态，让下一轮 LLM 输出完全独立
                    elif event_type == "on_tool_end":
                        result = event["data"].get("output", "")
                        print(f"\n  \u2705 结果: {str(result)[:200]}")
                        response_started = False       # 下轮重新输出 "Agent > "
                        thinking_printed = False       # 下轮重新处理思考标签
                        parser = StreamParser()        # 清空解析器，不残留旧 token

            asyncio.run(consume_events())

        except Exception as e:
            stop_thinking(thinking_th)
            print(f"\n  [错误: {e}]")

        # 处理解析器缓冲区中的剩余内容（流结束后的兜底）
        for is_thinking, text in parser.done():
            if is_thinking:
                if not thinking_printed:
                    stop_thinking(thinking_th)
                    thinking_printed = True
                type_text(text, 0.03)
            else:
                if not response_started:
                    stop_thinking(thinking_th)
                    if thinking_printed:
                        print()
                    print("Agent > ", end="", flush=True)
                    response_started = True
                type_text(text, 0.025)

        # 兜底：如果 LLM 一个 token 都没返回，也要停动画
        if not thinking_printed and not response_started:
            stop_thinking(thinking_th)
            print("Agent > (无回复)")

        print()


if __name__ == "__main__":
    main()
