"""第三个练习：交互式 Agent + 真实天气 API

新增功能：
1. 用 input() 让你自己输入问题（不再是写死的）
2. 对接 wttr.in 免费 API，查真实天气数据（无需注册、无需 Key）
3. Agent 可以多轮对话，输入 quit 退出

和上一个版本的区别：
  上一个：假数据（硬编码的天气字典）
  这一个：真实 API（wttr.in，免费，什么都不用注册）
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from config.settings import API_KEY, API_BASE, MODEL_FAST

# ← 关键：直接复用项目里的天气工具（免费 wttr.in，零配置）
from tools.weather_tool import get_weather as _get_weather_raw


# ==========================================
# 定义工具
# ==========================================

@tool
def get_weather(city: str) -> str:
    """查询城市实时天气（调用 wttr.in 免费 API）

    Args:
        city: 城市名称，如"北京"、"深圳"
    """
    return _get_weather_raw(city)


@tool
def get_time(city: str) -> str:
    """查询城市当前时间（简单实现）"""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{city}当前时间：{now}"


# 所有工具
tools = [get_weather, get_time]


# ==========================================
# 状态定义
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ==========================================
# 节点函数
# ==========================================

model = ChatOpenAI(
    model_name=MODEL_FAST,
    openai_api_key=API_KEY,
    openai_api_base=API_BASE
)
model_with_tools = model.bind_tools(tools)


def ai_think(state: AgentState) -> dict:
    """AI 思考节点"""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def execute_tool(state: AgentState) -> dict:
    """工具执行节点"""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    tool_messages = []
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]

        tool_function = {t.name: t for t in tools}[tool_name]
        result = tool_function.invoke(tool_args)
        print(f"  🔧 调用工具: {tool_name}({tool_args})")

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

    return {"messages": tool_messages}


def should_use_tool(state: AgentState) -> str:
    """路由：判断是否需要调用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tool"
    return "done"


# ==========================================
# 组装流程图
# ==========================================

graph = StateGraph(AgentState)
graph.add_node("ai_think", ai_think)
graph.add_node("execute_tool", execute_tool)
graph.add_edge(START, "ai_think")
graph.add_conditional_edges(
    "ai_think",
    should_use_tool,
    {"use_tool": "execute_tool", "done": END}
)
graph.add_edge("execute_tool", "ai_think")
agent = graph.compile()


# ==========================================
# 交互式运行
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 天气查询 Agent（交互式 + wttr.in 真实 API）")
    print("输入城市名查天气，输入 quit 退出")
    print("=" * 60)

    conversation = []

    while True:
        user_input = input("\n🧑 你: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q", "退出"):
            print("👋 再见！")
            break

        conversation.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": conversation})
        conversation = result["messages"]

        ai_reply = result["messages"][-1].content
        print(f"\n🤖 Agent: {ai_reply}")
