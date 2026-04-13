"""第二个练习：用 StateGraph 从零搭建一个 Agent

和 weather_agent.py 的区别：
- weather_agent.py 用 create_react_agent() 一键生成（黑盒）
- 这个文件用 StateGraph 一步步搭建（白盒，你能看到每一步在干什么）

学完这个你就理解了什么是"真正的 Agent"。
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Annotated
from typing_extensions import TypedDict

# LangGraph 核心：状态图
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from config.settings import API_KEY, API_BASE, MODEL_FAST


# ==========================================
# 第一步：定义工具
# ==========================================

@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    weather_data = {
        "北京": "晴，25°C",
        "深圳": "多云，28°C",
        "上海": "小雨，22°C",
    }
    return weather_data.get(city, f"{city}的天气未知")


@tool
def get_time(city: str) -> str:
    """查询城市当前时间"""
    time_data = {
        "北京": "2026-04-12 17:30",
        "深圳": "2026-04-12 17:30",
        "上海": "2026-04-12 17:30",
    }
    return time_data.get(city, f"{city}的时间未知")


# 所有工具放一个列表里，后面绑定到模型用
tools = [get_weather, get_time]


# ==========================================
# 第二步：定义状态（State）
# ==========================================
# 什么是状态？
# 状态就是 Agent 在整个流程中"记住的信息"
# 类似一个 Java 对象，每个节点函数都可以读它、改它
#
# Annotated[list, add_messages] 的意思：
# - list 类型（消息列表）
# - add_messages 是"合并策略"——每次返回新消息时，不是覆盖，而是追加到列表里
# 类似 Java 里的 List<Message>，每次 add() 而不是 set()

class AgentState(TypedDict):
    # messages 存放整个对话历史
    # 每个节点往里追加消息，下一个节点就能看到之前所有的对话
    messages: Annotated[list, add_messages]


# ==========================================
# 第三步：创建大模型（带工具能力的版本）
# ==========================================

model = ChatOpenAI(
    model_name=MODEL_FAST,
    openai_api_key=API_KEY,
    openai_api_base=API_BASE
)

# bind_tools()：告诉模型"你有这些工具可以用"
# 这一步不会让模型自动调用工具，只是让模型知道工具的存在
# 模型会在回复中"建议"调用哪个工具，但不会真的去调
model_with_tools = model.bind_tools(tools)


# ==========================================
# 第四步：定义节点函数（图的节点）
# ==========================================
# 每个节点就是一个普通 Python 函数
# 输入：当前状态（AgentState）
# 输出：状态的部分更新（只返回要改的字段）
#
# 对比任何流程引擎：
# 节点 = 流程图里的一个方框（处理步骤）
# 状态 = 流程图里在步骤之间传递的数据

def ai_think(state: AgentState) -> dict:
    """AI 思考节点：拿到对话历史，让大模型思考下一步该干什么"""
    # state["messages"] 是当前所有对话消息
    # 把它发给大模型，大模型会返回：
    #   - 如果需要调工具：返回一个 tool_call（建议调用哪个工具）
    #   - 如果可以直接回答：返回一段文字
    response = model_with_tools.invoke(state["messages"])

    # 返回一个 dict，LangGraph 会自动把它合并到状态里
    # 因为 messages 用了 add_messages 策略，这条 response 会追加到消息列表
    return {"messages": [response]}


def execute_tool(state: AgentState) -> dict:
    """工具执行节点：拿到 AI 的建议，真正调用工具函数"""
    # 取最后一条消息（就是 AI 的回复）
    last_message = state["messages"][-1]

    # AI 的回复里可能有多个 tool_calls（一次想调多个工具）
    tool_calls = last_message.tool_calls

    results = []
    for tc in tool_calls:
        # tc 是一个字典：{"name": "get_weather", "args": {"city": "北京"}}
        tool_name = tc["name"]
        tool_args = tc["args"]

        # 根据名字找到对应的工具函数
        tool_function = {t.name: t for t in tools}[tool_name]

        # 调用工具，拿到结果
        result = tool_function.invoke(tool_args)
        results.append(result)
        print(f"  🔧 调用工具: {tool_name}({tool_args}) → {result}")

    # 把工具结果追加到消息列表
    from langchain_core.messages import ToolMessage
    tool_messages = []
    for tc, result in zip(tool_calls, results):
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

    return {"messages": tool_messages}


# ==========================================
# 第五步：定义路由逻辑（条件边）
# ==========================================
# 路由 = 根据当前状态决定下一步走哪条路
# 类似流程图里的菱形判断框（Gateway）

def should_use_tool(state: AgentState) -> str:
    """判断 AI 是否需要调用工具"""
    last_message = state["messages"][-1]

    # 如果 AI 的回复里有 tool_calls，说明它想调工具
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tool"  # → 走工具执行节点
    else:
        return "done"      # → 流程结束


# ==========================================
# 第六步：组装流程图（StateGraph）
# ==========================================
# 这里就是在画流程图！
# add_node = 画一个方框
# add_edge = 画一条连线
# add_conditional_edges = 画一个菱形判断

# 创建一个空图，指定状态类型
graph = StateGraph(AgentState)

# 添加节点（画方框）
graph.add_node("ai_think", ai_think)        # AI 思考节点
graph.add_node("execute_tool", execute_tool)  # 工具执行节点

# 设置入口：从 START 连到 ai_think
# 类似流程图里的"开始"节点
graph.add_edge(START, "ai_think")

# 添加条件边（画菱形判断）
# 从 ai_think 出发，根据 should_use_tool 的返回值决定走哪条路：
#   "use_tool" → execute_tool
#   "done"     → END（结束）
graph.add_conditional_edges(
    "ai_think",
    should_use_tool,
    {"use_tool": "execute_tool", "done": END}
)

# 工具执行完之后，回到 AI 思考节点（形成循环）
# 这就是 Agent 能"多步推理"的关键！
graph.add_edge("execute_tool", "ai_think")

# 编译图（生成可执行的对象）
agent = graph.compile()


# ==========================================
# 第七步：运行
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("StateGraph 版 Agent 启动！")
    print("=" * 60)

    # 测试 1：需要调工具的问题
    print("\n📌 测试 1：北京天气怎么样（需要调工具）")
    print("-" * 40)
    result = agent.invoke({
        "messages": [{"role": "user", "content": "北京天气怎么样？"}]
    })

    print("\n--- 对话过程 ---")
    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content}")

    print("\n--- 最终回答 ---")
    print(result["messages"][-1].content)

    # 测试 2：不需要调工具的问题
    print("\n\n📌 测试 2：你好，你是谁？（不需要调工具）")
    print("-" * 40)
    result2 = agent.invoke({
        "messages": [{"role": "user", "content": "你好，你是谁？"}]
    })

    print("\n--- 对话过程 ---")
    for msg in result2["messages"]:
        print(f"[{msg.type}] {msg.content}")

    print("\n--- 最终回答 ---")
    print(result2["messages"][-1].content)
