# 🤖 My Agent 2.0

> 基于 LangGraph 的 Multi-Agent 对话系统 — Supervisor 编排 + 主 Agent 全能 + 子 Agent 专家协作

## ✨ 特性

- **Multi-Agent 架构** — Supervisor 路由：主 Agent（全能）+ Coder（代码专家）+ Planner（研究规划）
- **主 Agent 全能** — 13 个工具全部可用，大部分对话直接处理，不再绕一圈委托
- **多 Provider 切换** — Ollama / 智谱 GLM / MiniMax，改一行配置即可
- **流式输出 + 思考动画** — 实时流式响应，Braille 点阵思考动画
- **联网搜索** — SearXNG / DuckDuckGo，工厂模式 + 主备降级
- **网页抓取** — URL 自动识别并抓取内容
- **企业级 RAG** — ChromaDB + BM25 + RRF 融合检索，增量入库，支持 46 种文件格式
- **文件操作** — 读取/搜索/写入/编辑文件，执行 Shell 命令
- **多会话管理** — SQLite 持久化，支持新建/切换/重命名/删除会话
- **工具容错 + LLM 重试** — 工具异常不崩溃，API 错误自动重试（最多 10 次）
- **MiniMax 兼容** — 不依赖 SystemMessage，所有指令合并到 HumanMessage

## 📸 项目架构

```
                    ┌─────────────────────────────────┐
                    │            用户输入              │
                    └───────────────┬─────────────────┘
                                    ↓
                    ┌─────────────────────────────────┐
                    │         Supervisor 路由          │
                    │                                  │
                    │  ┌─────────┐  ┌───────┐  ┌────┐ │
                    │  │  chat   │  │ coder │  │plan│ │
                    │  │ (全能)  │  │(代码) │  │(研)│ │
                    │  │ 13 工具 │  │ 7 工具│  │4工具│ │
                    │  └─────────┘  └───────┘  └────┘ │
                    └─────────────────────────────────┘
                           │              │
              ┌────────────┼──────────────┼──────────────┐
              ↓            ↓              ↓              ↓
        ┌──────────┐ ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ 天气查询 │ │ 联网搜索 │  │ RAG 检索 │  │ 文件操作 │
        │(wttr.in) │ │(SearXNG/ │  │(ChromaDB+│  │(read/write│
        │          │ │ DuckDuck)│  │ BM25+RRF)│  │ /edit)   │
        └──────────┘ └──────────┘  └──────────┘  └──────────┘
```

## 🏗️ Multi-Agent 设计

| Agent | 工具数 | 职责 | 何时触发 |
|-------|--------|------|----------|
| **chat（主 Agent）** | 13（全部） | 默认处理所有请求 | 闲聊、问答、搜索、查天气、看网址等 |
| **coder** | 7 | 深度代码开发 | 需要写完整功能、重构代码、修复杂 Bug |
| **planner** | 4 | 深度研究分析 | 多轮搜索、竞品分析、技术调研 |

**路由原则**：不确定时默认走 chat。只有明确的「深度代码开发」或「多轮深度研究」才委托子 Agent。

**设计参考**：LangGraph 官方（按技能域分）、CrewAI（按工作流阶段分）、AutoGen（按角色分）—— Agent 按「能力域」而非「工具种类」划分。

## 🛠️ 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **Agent 编排** | LangGraph StateGraph | 手动 think → exec_tool 循环 |
| **多 Agent** | Supervisor 模式 | 路由节点 + 子 Agent StateGraph |
| **LLM** | MiniMax M2.7 / 智谱 GLM / Ollama | 兼容 OpenAI API 格式，一键切换 |
| **Embedding** | BAAI/bge-large-zh-v1.5 | 1024 维，中文 MTEB 霸榜 |
| **向量库** | ChromaDB | 本地持久化，零配置 |
| **关键词检索** | BM25 (rank_bm25) | 稀疏检索，精确匹配 ID/术语 |
| **融合排序** | RRF (Reciprocal Rank Fusion) | 向量 + BM25 结果合并 |
| **联网搜索** | SearXNG / DuckDuckGo | 工厂模式，主备降级 |
| **网页抓取** | httpx + selectolax | 轻量级 HTML 解析 |
| **文件操作** | read/write/edit/search | 代码文件读写编辑 |
| **Shell** | asyncio.create_subprocess | 命令执行，30s 超时 + 危险命令拦截 |
| **会话持久化** | AsyncSqliteSaver | LangGraph 原生 checkpoint |
| **文件解析** | PyPDF2 / python-docx / python-pptx | 支持 46 种文件格式 |

## 📁 项目结构

```
langgraph-agent-lab/
├── agents/                     # Agent 实现
│   ├── chat_agent.py           # ⭐ 核心：Supervisor + Multi-Agent（~800 行）
│   ├── session_manager.py      # 多会话管理（SQLite + JSON 索引）
│   └── ui/                     # UI 模块
│       ├── spinner.py          # Braille 点阵思考动画
│       ├── stream_parser.py    # 流式输出解析（think 标签过滤）
│       ├── display.py          # 终端颜色/布局
│       ├── colors.py           # ANSI 颜色常量
│       └── layout.py           # 终端宽度检测
├── basics/                     # 学习练习（从零到一）
│   ├── 01_weather_agent.py     # 练习 1：第一个 Agent
│   ├── 02_stategraph_agent.py  # 练习 2：StateGraph 白盒版
│   ├── 03_interactive_agent.py # 练习 3：交互式 + 真实 API
│   └── 04_free_weather_agent.py# 练习 4：零配置 Skill 风格
├── tools/                      # 工具模块
│   ├── rag_tool.py             # ⭐ 企业级 RAG（混合检索 + 增量入库 + BM25 缓存）
│   ├── weather_tool.py         # 天气查询（wttr.in，免费无 Key）
│   ├── time_tool.py            # 当前时间查询
│   ├── file_ops.py             # 文件读取/列表/搜索（read_file, list_dir, search_file, search_content）
│   ├── file_edit.py            # 文件编辑（write_file, edit_file）
│   ├── shell_tool.py           # Shell 命令执行（run_command）
│   └── search/                 # 联网搜索
│       ├── base.py             # 搜索引擎抽象基类
│       ├── engine_factory.py   # 引擎工厂（主备降级）
│       ├── web_search_tool.py  # LangGraph Tool 封装
│       ├── web_fetch_tool.py   # 网页内容抓取
│       └── engines/            # 搜索引擎实现
│           ├── searxng.py      # SearXNG（自建）
│           └── duckduckgo.py   # DuckDuckGo（免 Key）
├── config/
│   └── settings.py             # ⭐ 动态 Provider 切换 + 环境变量管理
├── .env.example                # 环境变量模板
├── run_chat_agent.bat          # Windows 一键启动
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
| `/models` | 列出可用模型 |
| `/help` | 显示帮助 |
| `/quit` | 退出 |

## 🔍 核心功能

### Multi-Agent 路由

Supervisor 根据用户意图自动路由：

```
用户: "你好"                          → Agent（主 Agent 直接回复）
用户: "这个网站是什么？https://..."    → Agent（主 Agent 调用 web_fetch）
用户: "帮我写一个完整的爬虫项目"       → Agent | Coder（委托代码专家）
用户: "帮我做 Spring Boot vs Quarkus 技术选型" → Agent | Planner（委托研究专家）
```

### 多 Provider 切换

只需修改 `.env` 中一行配置：

```bash
# 使用 MiniMax（推荐）
PROVIDER=minimax

# 使用智谱 GLM
PROVIDER=glm

# 使用 Ollama 本地模型（免费）
PROVIDER=ollama
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

## 📊 学习路线

| 练习 | 文件 | 学到什么 |
|------|------|---------|
| 01 | `basics/01_weather_agent.py` | @tool 装饰器、create_react_agent |
| 02 | `basics/02_stategraph_agent.py` | StateGraph 白盒搭建 |
| 03 | `basics/03_interactive_agent.py` | 交互式对话 + 真实 API |
| 04 | `basics/04_free_weather_agent.py` | 零配置 Skill 风格 |
| ⭐ | `agents/chat_agent.py` | Multi-Agent Supervisor + 全能主 Agent + 工具容错 + 多会话 |

## 📄 License

MIT License
