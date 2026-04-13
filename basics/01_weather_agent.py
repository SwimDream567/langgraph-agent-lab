"""第一个 Agent：天气查询

学习目标：
1. @tool 装饰器 = 定义工具（类似 Java 里的 @Service）
2. ChatOpenAI = 大模型（类似调第三方 API）
3. create_react_agent = 把工具和模型组装成 Agent
4. agent.invoke() = 运行 Agent（类似 Java 里的方法调用）

对比 Java：
- @tool → @Service（定义一个可被调用的服务）
- create_react_agent() → Spring 的依赖注入（把所有 Bean 组装起来）
- agent.invoke() → controller 调用 service
"""

import sys
import os

# sys.path.insert(0, ...) 把项目根目录加到 Python 的搜索路径里
# 这样才能 from config.settings import ... 找到 config 包
# 类似 Java 里在 pom.xml 加依赖，让 JVM 能找到类
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ==========================================
# 导入依赖（类似 Java 的 import）
# ==========================================

# @tool 装饰器：把一个普通 Python 函数变成"Agent 可调用的工具"
# 类似 Java 里的 @Service 注解，标记这个类/方法是一个可被调用的服务
from langchain_core.tools import tool

# ChatOpenAI：大语言模型的客户端
# 类似 Java 里的 RestTemplate 或 HttpClient
# 它负责把你的请求发给 AI 模型（GLM/MiniMax），拿回回复
from langchain_openai import ChatOpenAI

# create_react_agent：LangGraph 提供的"一键创建 Agent"方法
# ReAct = Reasoning（思考） + Acting（行动）
# 这是 Agent 最经典的工作模式：先想再干
# 类似 Spring Boot 的 @Autowired，自动把模型和工具组装在一起
from langgraph.prebuilt import create_react_agent

# 从配置文件导入 API Key 和模型参数
# 类似 Java 里 @Value("${api.key}") 读 application.yml
from config.settings import API_KEY, API_BASE, MODEL_FAST


# ==========================================
# 第一步：定义工具（类似 Java 里写 @Service）
# ==========================================
# 什么是"工具"？
# 工具就是 Agent 可以调用的函数。Agent 自己只能"说话"（生成文本），
# 但通过工具，它可以"做事"——查天气、查数据库、调 API、发邮件……
#
# 对比 Flowable：
# @tool 定义的函数 = Flowable 里的 ServiceTask
# Agent 在运行时自动决定要不要调用 = Flowable 里的网关（Gateway）

@tool  # 这个装饰器把下面的普通函数注册为"Agent 工具"
def get_weather(city: str) -> str:
    """查询城市天气"""
    # """...""" 这个文档字符串非常重要！
    # Agent 靠读这个字符串来理解"这个工具是干什么的"
    # 类似 Java 里接口的 JavaDoc，Agent 会读它来决定是否调用
    # 如果不写，Agent 就不知道什么时候该用这个工具

    # 这里是假数据，真实项目会调用天气 API（和风天气、心知天气等）
    weather_data = {
        "北京": "晴，25°C",
        "深圳": "多云，28°C",
        "上海": "小雨，22°C",
    }
    # dict.get(key, default) —— 找不到城市时返回"未知"
    return weather_data.get(city, f"{city}的天气未知")

# 你可以继续加更多工具，Agent 会根据用户问题自动选择调用哪个
# 比如：
# @tool
# def search_web(query: str) -> str:
#     """搜索互联网"""
#     ...
#
# @tool
# def calculate(expression: str) -> str:
#     """计算数学表达式"""
#     ...


# ==========================================
# 第二步：配置大模型（类似 Java 里配置 HttpClient）
# ==========================================
# ChatOpenAI 是一个通用的 LLM 客户端
# 虽然名字里有 "OpenAI"，但因为 GLM/MiniMax 都兼容 OpenAI 格式
# 所以可以直接用它连任何兼容的模型

model = ChatOpenAI(
    model_name=MODEL_FAST,     # 用哪个模型（MiniMax-M / glm-4-flash）
    openai_api_key=API_KEY,    # 认证密钥（类似 Bearer Token）
    openai_api_base=API_BASE   # API 地址（指向 MiniMax 或 GLM 的服务器）
)

# model 这个对象本身就能对话，但还没有"工具"能力
# 它只是一个纯粹的聊天模型，不会主动调用 get_weather()
# 下一步就是把 model + tools 组装起来


# ==========================================
# 第三步：组装 Agent（类似 Spring 的依赖注入）
# ==========================================
# create_react_agent(model, tools) 做了什么？
# 1. 把 model（大脑）和 tools（手脚）组装在一起
# 2. 创建一个"思考-行动"循环：
#    用户提问 → model 思考要不要用工具 → 调用工具 → 拿到结果 → model 再思考 → 回答
# 3. 这个循环会自动重复，直到 model 认为已经可以回答用户了
#
# 对比 Flowable：
# create_react_agent() 相当于定义了一个流程：
#   StartEvent → ServiceTask(AI思考) → Gateway(需要工具?) → ServiceTask(调工具) → 回到AI思考 → EndEvent

agent = create_react_agent(model, [get_weather])
#                              ↑          ↑
#                           大脑        可用工具列表
#                         (做决策)     (执行具体操作)


# ==========================================
# 第四步：运行 Agent（类似 main 方法）
# ==========================================

if __name__ == "__main__":
    print("=" * 50)
    print("天气查询 Agent 启动！")
    print("=" * 50)

    # agent.invoke() 是同步调用，传入一个状态字典
    # messages 是一个消息列表，格式和 OpenAI API 一样：
    # {"role": "user", "content": "..."} 表示用户消息
    # Agent 会自动往这个列表里追加：AI 的思考、工具调用、工具返回、最终回答
    result = agent.invoke({
        "messages": [{"role": "user", "content": "北京天气怎么样？"}]
    })
    # result["messages"] 现在包含了完整的对话链：
    # [0] human: "北京天气怎么样？"          ← 用户提问
    # [1] ai: "我需要调用天气查询工具..."    ← AI 思考（决定调工具）
    # [2] tool: "晴，25°C"                 ← 工具返回结果
    # [3] ai: "北京今天晴天，25°C..."       ← AI 最终回答

    # 打印完整的思考过程（可以看到 Agent 是怎么一步步推理的）
    print("\n--- Agent 思考过程 ---")
    for msg in result["messages"]:
        # msg.type 是消息类型：human（用户）/ ai（AI）/ tool（工具）
        # msg.content 是消息内容
        print(f"[{msg.type}] {msg.content}")

    # 最后一条消息就是 Agent 的最终回答
    print("\n--- 最终回答 ---")
    print(result["messages"][-1].content)
