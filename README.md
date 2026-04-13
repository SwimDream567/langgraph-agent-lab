# 🤖 LangGraph Agent Lab

> 基于 ReAct 循环的智能对话 Agent，集成混合检索 RAG（ChromaDB + BM25 + RRF）

## 📸 项目架构

```
                    ┌─────────────────────────────┐
                    │         用户输入              │
                    └──────────────┬──────────────┘
                                   ↓
                    ┌─────────────────────────────┐
                    │     LangGraph StateGraph      │
                    │                              │
                    │   ┌─────────┐    ┌─────────┐ │
                    │   │ ai_think │───→│ 路由判断  │ │
                    │   └────┬────┘    └────┬────┘ │
                    │        ↑              ↓      │
                    │        │      ┌──────────┐   │
                    │        └──────│ exec_tool│   │
                    │               └──────────┘   │
                    └─────────────────────────────┘
                           │              │
                ┌──────────↓──┐    ┌──────↓──────┐
                │  天气查询     │    │  RAG 检索    │
                │  (wttr.in)   │    │ (混合检索)   │
                └─────────────┘    └─────────────┘
                                          │
                          ┌───────────────┼───────────────┐
                          ↓               ↓               ↓
                    ┌──────────┐   ┌──────────┐   ┌──────────┐
                    │ ChromaDB │   │  BM25    │   │   RRF    │
                    │ 向量检索  │   │ 关键词检索│   │ 融合排序  │
                    └──────────┘   └──────────┘   └──────────┘
```

## 🛠️ 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **Agent 编排** | LangGraph StateGraph | ReAct 循环（思考→行动→观察） |
| **LLM** | MiniMax GLM / 智谱 GLM | 兼容 OpenAI API 格式 |
| **Embedding** | BAAI/bge-large-zh-v1.5 | 1024 维，中文 MTEB 霸榜 |
| **向量库** | ChromaDB | 本地持久化，零配置 |
| **关键词检索** | BM25 (rank_bm25) | 稀疏检索，精确匹配 ID/术语 |
| **融合排序** | RRF (Reciprocal Rank Fusion) | 向量 + BM25 结果合并 |
| **文件解析** | PyPDF2 / python-docx / python-pptx | 支持 46 种文件格式 |

## 📁 项目结构

```
langgraph-agent-lab/
├── agents/                     # Agent 实现
│   ├── chat_agent.py           # ⭐ 核心：流式 ReAct Agent（LangGraph）
│   ├── rag_agent.py            # RAG 完整版（切块对比 + MMR）
│   └── rag_quickstart.py       # RAG 入门版（Obsidian 笔记）
├── basics/                     # 学习练习（从零到一）
│   ├── 01_weather_agent.py     # 练习 1：第一个 Agent
│   ├── 02_stategraph_agent.py  # 练习 2：StateGraph 白盒版
│   ├── 03_interactive_agent.py # 练习 3：交互式 + 真实 API
│   └── 04_free_weather_agent.py# 练习 4：零配置 Skill 风格
├── tools/                      # 工具模块
│   ├── rag_tool.py             # ⭐ 企业级 RAG 工具（混合检索 + 增量入库）
│   └── weather_tool.py         # 天气查询（wttr.in，免费无 Key）
├── config/                     # 配置
│   └── settings.py             # API Key 从 .env 读取
├── multi_agent/                # [规划中] 多 Agent 协作
├── workflows/                  # [规划中] Human-in-the-loop
├── .env.example                # 环境变量模板
├── .gitignore
├── requirements.txt
└── run_chat_agent.bat          # 一键启动脚本
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/SwimDream567/langgraph-agent-lab.git
cd langgraph-agent-lab
```

### 2. 创建虚拟环境

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

### 5. 启动 Agent

```bash
# 方式 1：一键启动（Windows）
run_chat_agent.bat

# 方式 2：直接运行
python agents/chat_agent.py
```

## 🔍 核心功能

### RAG 混合检索

```python
from tools.rag_tool import rag_search, ingest

# 第一步：入库（扫描文件夹 → 切块 → 向量化）
ingest("D:/MyDocs/wiki")

# 第二步：检索（向量 + BM25 + RRF 融合）
result = rag_search("游梦的技术栈是什么？")
```

### LangGraph ReAct Agent

```python
# 三节点循环：思考 → 判断 → 执行 → 回到思考
# ai_think → should_use_tool? → execute_tool → ai_think
```

### 支持的文件格式（46 种）

| 类别 | 格式 |
|------|------|
| 文本 | .md .txt .rst .log |
| 代码 | .py .js .ts .java .go .rs .cpp .c 等 19 种 |
| Web | .html .css .json .yaml .toml 等 11 种 |
| 文档 | .pdf .docx .pptx |
| 数据 | .csv .tsv |

## 📊 学习路线

| 练习 | 文件 | 学到什么 |
|------|------|---------|
| 01 | `basics/01_weather_agent.py` | @tool 装饰器、create_react_agent |
| 02 | `basics/02_stategraph_agent.py` | StateGraph 白盒搭建 |
| 03 | `basics/03_interactive_agent.py` | 交互式对话 + 真实 API |
| 04 | `basics/04_free_weather_agent.py` | 零配置 Skill 风格 |
| ⭐ | `agents/chat_agent.py` | 流式输出 + 思考动画 + LangGraph 循环 |

## 📄 License

MIT License
