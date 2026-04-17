# 🤖 My Agent

> 基于 LangGraph 的智能对话 Agent，支持多 LLM Provider、联网搜索、混合检索 RAG

## ✨ 特性

- **多 Provider 切换** — Ollama / 智谱 GLM / MiniMax，改一行配置即可
- **流式输出 + 思考动画** — 实时流式响应，思考过程可视化
- **工具容错 + LLM 重试** — 工具异常不崩溃，API 错误自动重试（最多 10 次）
- **联网搜索** — SearXNG / DuckDuckGo，主备降级
- **企业级 RAG** — ChromaDB + BM25 + RRF 融合检索，增量入库，支持 46 种文件格式
- **多会话管理** — SQLite 持久化，支持新建/切换/重命名/删除会话

## 📸 项目架构

```
                    ┌─────────────────────────────────┐
                    │            用户输入               │
                    └───────────────┬─────────────────┘
                                    ↓
                    ┌─────────────────────────────────┐
                    │      LangGraph StateGraph         │
                    │                                   │
                    │   ┌──────────┐    ┌──────────┐   │
                    │   │ ai_think  │───→│ 路由判断   │   │
                    │   └─────┬────┘    └────┬─────┘   │
                    │         ↑              ↓         │
                    │         │       ┌───────────┐    │
                    │         └───────│ exec_tool │    │
                    │                 └───────────┘    │
                    └─────────────────────────────────┘
                           │         │          │
                 ┌─────────↓──┐ ┌────↓─────┐ ┌──↓──────────┐
                 │  天气查询    │ │ RAG 检索  │ │  联网搜索    │
                 │  (wttr.in)  │ │(混合检索) │ │(SearXNG/    │
                 └────────────┘ └────┬─────┘ │ DuckDuckGo) │
                                     │       └─────────────┘
                   ┌─────────────────┼─────────────────┐
                   ↓                 ↓                 ↓
             ┌──────────┐     ┌──────────┐     ┌──────────┐
             │ ChromaDB │     │  BM25    │     │   RRF    │
             │ 向量检索   │     │ 关键词检索│     │ 融合排序  │
             └──────────┘     └──────────┘     └──────────┘
```

## 🛠️ 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **Agent 编排** | LangGraph StateGraph | ReAct 循环（思考→行动→观察） |
| **LLM** | Ollama / 智谱 GLM / MiniMax | 兼容 OpenAI API 格式，一键切换 |
| **Embedding** | BAAI/bge-large-zh-v1.5 | 1024 维，中文 MTEB 霸榜 |
| **向量库** | ChromaDB | 本地持久化，零配置 |
| **关键词检索** | BM25 (rank_bm25) | 稀疏检索，精确匹配 ID/术语 |
| **融合排序** | RRF (Reciprocal Rank Fusion) | 向量 + BM25 结果合并 |
| **联网搜索** | SearXNG / DuckDuckGo | 主备降级，可自建搜索服务 |
| **会话持久化** | AsyncSqliteSaver | LangGraph 原生 checkpoint |
| **文件解析** | PyPDF2 / python-docx / python-pptx | 支持 46 种文件格式 |

## 📁 项目结构

```
langgraph-agent-lab/
├── agents/                     # Agent 实现
│   └── chat_agent.py           # ⭐ 核心：流式 ReAct Agent（多会话 + 多Provider）
├── basics/                     # 学习练习（从零到一）
│   ├── 01_weather_agent.py     # 练习 1：第一个 Agent
│   ├── 02_stategraph_agent.py  # 练习 2：StateGraph 白盒版
│   ├── 03_interactive_agent.py # 练习 3：交互式 + 真实 API
│   └── 04_free_weather_agent.py# 练习 4：零配置 Skill 风格
├── tools/                      # 工具模块
│   ├── rag_tool.py             # ⭐ 企业级 RAG（混合检索 + 增量入库 + BM25 缓存）
│   ├── weather_tool.py         # 天气查询（wttr.in，免费无 Key）
│   └── search/                 # 联网搜索
│       ├── base.py             # 搜索引擎抽象基类
│       ├── engine_factory.py   # 引擎工厂（主备降级）
│       ├── web_search_tool.py  # LangGraph Tool 封装
│       ├── web_fetch_tool.py   # 网页内容抓取
│       └── engines/            # 搜索引擎实现
│           ├── searxng.py      # SearXNG（自建）
│           └── duckduckgo.py   # DuckDuckGo（免 Key）
├── config/
│   └── settings.py             # ⭐ 动态 Provider 切换
├── multi_agent/                # [规划中] 多 Agent 协作
├── workflows/                  # [规划中] Human-in-the-loop
├── .env.example                # 环境变量模板
└── requirements.txt
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
# 设置 PROVIDER=ollama / glm / minimax 选择 LLM
```

### 5. 启动 Agent

```bash
python agents/chat_agent.py
```

### 斜杠命令

| 命令 | 说明 |
|------|------|
| `/sessions` | 列出所有会话 |
| `/new` | 新建会话 |
| `/switch <id>` | 切换到指定会话 |
| `/rename <name>` | 重命名当前会话 |
| `/delete` | 删除当前会话 |
| `/help` | 显示帮助 |

## 🔍 核心功能

### 多 Provider 切换

只需修改 `.env` 中一行配置：

```bash
# 使用 Ollama 本地模型（免费）
PROVIDER=ollama

# 使用智谱 GLM
PROVIDER=glm

# 使用 MiniMax
PROVIDER=minimax
```

### RAG 混合检索

```python
from tools.rag_tool import rag_search, ingest

# 第一步：入库（扫描文件夹 → 切块 → 向量化）
ingest("D:/MyDocs/wiki")

# 第二步：检索（向量 + BM25 + RRF 融合）
result = rag_search("游梦的技术栈是什么？")
```

### 联网搜索

支持 SearXNG（自建）和 DuckDuckGo（免 Key），主引擎不可用时自动降级：

```bash
# .env 配置
SEARCH_ENGINE=searxng          # 主搜索引擎
SEARXNG_BASE_URL=http://localhost:8888
SEARCH_FALLBACK=duckduckgo     # 备用引擎
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
| ⭐ | `agents/chat_agent.py` | 流式输出 + 多Provider + 工具容错 + 多会话 |

## 📄 License

MIT License
