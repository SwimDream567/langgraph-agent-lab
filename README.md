# 🤖 My Agent 2.0

> 基于 LangGraph 的 Multi-Agent 对话系统 — Supervisor 编排 + 主 Agent 全能 + 子 Agent 专家协作

## ✨ 特性

- **Multi-Agent 架构** — Supervisor 路由：主 Agent（全能）+ Coder（代码专家）+ Planner（研究规划）
- **主 Agent 全能** — 13 个工具全部可用，大部分对话直接处理，不再绕一圈委托
- **多模型注册表** — `/models` 运行时热切换，支持继承配置（共用 KEY/BASE，只换 MODEL_ID）
- **三层上下文压缩** — L1 MicroCompact（0 成本）+ L2 会话记忆（0 成本）+ L3 LLM 摘要
- **MCP 动态加载器** — 读取 `.mcp.json`，持久 session + 热重载 + 斜杠命令管理
- **工具输出预算** — 固定上限 + 截断感知提示，agent 主动补读
- **流式输出 + 思考动画** — 实时流式响应，Braille 点阵思考动画
- **Textual TUI 界面** — 弹窗选择器（ModalScreen）+ HITL 人机协同 + Tab 命令补全
- **HITL 人机协同** — AI 可暂停执行弹出选项让用户确认（ask_user / confirm / 多问题调查）
- **联网搜索** — SearXNG / DuckDuckGo，工厂模式 + 主备降级
- **网页抓取** — URL 自动识别并抓取内容
- **企业级 RAG** — ChromaDB + BM25 + RRF 融合检索，增量入库，支持 46 种文件格式
- **文件操作** — 读取/搜索/写入/编辑文件，执行 Shell 命令
- **多会话管理** — SQLite 持久化，支持新建/切换/重命名/删除会话
- **会话记忆追踪** — 零 API 调用，自动追踪文件/工具/话题，注入每轮对话
- **工具容错 + LLM 重试** — 工具异常不崩溃，API 错误自动重试（最多 10 次）
- **Ollama 预热防重入** — 后台加载模型，Banner 先行显示
- **熔断器 + 冷却机制** — 压缩失败自动降频，防止死循环

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
              │            │              │              │
              └────────────┼──────────────┼──────────────┘
                           ↓
              ┌─────────────────────────────────┐
              │     三层上下文管理（Claude Code 风格） │
              │  L1 MicroCompact → L2 记忆追踪 → L3 LLM 摘要  │
              └─────────────────────────────────┘
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
| **LLM** | MiniMax / 智谱 GLM / DeepSeek / 通义千问 / Ollama | OpenAI 兼容格式，`/models` 热切换 |
| **多模型注册表** | ModelRegistry | 环境变量声明 + 继承配置 + 运行时切换 |
| **上下文管理** | core/context_manager.py | L1 清除旧工具输出 → L2 会话记忆 → L3 LLM 摘要 |
| **MCP** | core/mcp_loader.py | 读取 `.mcp.json`，持久 session + 热重载 |
| **Embedding** | BAAI/bge-large-zh-v1.5 | 1024 维，中文 MTEB 霸榜 |
| **向量库** | ChromaDB | 本地持久化，零配置 |
| **关键词检索** | BM25 (rank_bm25) | 稀疏检索，精确匹配 ID/术语 |
| **融合排序** | RRF (Reciprocal Rank Fusion) | 向量 + BM25 结果合并 |
| **联网搜索** | SearXNG / DuckDuckGo | 工厂模式，主备降级 |
| **网页抓取** | httpx + selectolax | 轻量级 HTML 解析 |
| **文件操作** | read/write/edit/search | 代码文件读写编辑 |
| **Shell** | asyncio.create_subprocess | 命令执行，30s 超时 + 危险命令拦截 |
| **会话持久化** | AsyncSqliteSaver | LangGraph 原生 checkpoint |
| **TUI 框架** | Textual 8.2.3 | ModalScreen 弹窗 + OptionList 选择 + 流式 Markdown |
| **HITL** | asyncio.Future 同步桥 | Agent 暂停等用户确认，支持选项/确认/多问题调查 |
| **文件解析** | PyPDF2 / python-docx / python-pptx | 支持 46 种文件格式 |

## 📁 项目结构

```
langgraph-agent-lab/
├── agents/                     # Agent 实现
│   ├── chat_agent.py           # ⭐ 核心：Supervisor + Multi-Agent + TUI 入口
│   ├── session_manager.py      # 多会话管理（SQLite + JSON 索引）
│   └── ui/                     # UI 模块
│       ├── tui_app.py          # ⭐ Textual TUI 界面（弹窗选择器 + HITL + 命令补全）
│       ├── stream_parser.py    # 流式输出解析（think 标签过滤）
│       └── colors.py           # ANSI 颜色常量
├── basics/                     # 学习练习（从零到一）
│   ├── 01_weather_agent.py     # 练习 1：第一个 Agent
│   ├── 02_stategraph_agent.py  # 练习 2：StateGraph 白盒版
│   ├── 03_interactive_agent.py # 练习 3：交互式 + 真实 API
│   └── 04_free_weather_agent.py# 练习 4：零配置 Skill 风格
├── core/                      # Agent 内部基础设施（非 LLM @tool）
│   ├── context_manager.py     # ⭐ 三层上下文压缩（Claude Code 风格）
│   ├── mcp_loader.py          # ⭐ MCP 动态加载器（持久 session + 热重载）
│   ├── output_budget.py       # ⭐ 工具输出预算（固定上限 + 截断感知提示）
│   └── session_memory.py      # ⭐ 会话记忆追踪（零成本，纯规则提取）
├── tools/                     # LLM @tool（对外可调用的工具）
│   ├── rag_tool.py             # ⭐ 企业级 RAG（混合检索 + 增量入库 + BM25 缓存）
│   ├── hitl_tool.py            # ⭐ HITL 人机协同（ask_user / confirm / ask_questions）
│   ├── weather_tool.py         # 天气查询（wttr.in，免费无 Key）
│   ├── time_tool.py            # 当前时间查询
│   ├── file_ops.py             # 文件读取/列表/搜索
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
│   └── settings.py             # ⭐ 多模型注册表（环境变量声明 + 继承 + 热切换）
├── .env.example                # 环境变量模板
├── .mcp.json                   # MCP 服务器配置（.gitignore 排除）
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
```

`.env` 配置示例：

```bash
# ── 方式一：简单配置（单模型）──
LLM_API_KEY=你的API_Key
LLM_API_BASE=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# ── 方式二：多模型注册表（推荐）──
ACTIVE_MODEL=minimax

# MiniMax（主模型）
MODEL_MINIMAX_KEY=你的MiniMax_Key
MODEL_MINIMAX_BASE=https://api.minimaxi.com/v1
MODEL_MINIMAX_ID=MiniMax-M2.7

# DeepSeek（继承 MiniMax 的 KEY/BASE，只换模型 ID）
# MODEL_DEEPSEEK_INHERIT=minimax
# MODEL_DEEPSEEK_ID=deepseek-chat

# Ollama 本地（免费）
# MODEL_OLLAMA_KEY=ollama
# MODEL_OLLAMA_BASE=http://localhost:11434/v1
# MODEL_OLLAMA_ID=gemma4:e4b
```

### 5. 启动 Agent

```bash
python agents/chat_agent.py
```

### 斜杠命令

| 命令 | 说明 |
|------|------|
| `/sessions` `/ls` | 弹窗列出/切换会话（↑↓选择，支持 Tab 补全） |
| `/new` [名称] | 新建并切换到新会话 |
| `/rename` <名称> | 重命名当前会话 |
| `/delete` | 删除指定会话 |
| `/models` | 弹窗列出/切换可用模型 |
| `/mcp` | 管理 MCP 服务器 |
| `/context` | 查看当前上下文状态 |
| `/compact` | 手动压缩上下文 |
| `/quit` `/q` | 退出 |

## 🔍 核心功能

### 三层上下文压缩

对标 Claude Code 的上下文管理策略：

```
Layer 1: MicroCompact（0 成本）
  → 清除旧工具输出（白名单机制），保留最近 N 条
  → 回收 60-70% token，不破坏消息结构

Layer 2: 会话记忆追踪（0 成本）
  → 纯规则提取文件路径、工具调用、用户话题
  → 自动去重，超容量时淘汰旧条目
  → 注入每轮对话，控制在 800 字符内

Layer 3: LLM 摘要压缩（有成本）
  → 触发阈值：估算 token 数 > CONTEXT_COMPACT_THRESHOLD
  → LLM 一次性摘要 + 修剪旧消息
  → 熔断器：压缩失败自动冷却 3 轮
```

### Multi-Agent 路由

Supervisor 根据用户意图自动路由：

```
用户: "你好"                          → chat（主 Agent 直接回复）
用户: "这个网站是什么？https://..."    → chat（调用 web_fetch）
用户: "帮我写一个完整的爬虫项目"       → coder（委托代码专家）
用户: "帮我做 Spring Boot vs Quarkus 技术选型" → planner（委托研究专家）
```

### 多模型注册表

支持运行时 `/models` 热切换，继承配置避免重复写 KEY/BASE：

```bash
# .env 中声明多个模型
ACTIVE_MODEL=minimax
MODEL_MINIMAX_KEY=sk-xxx
MODEL_MINIMAX_BASE=https://api.minimaxi.com/v1
MODEL_MINIMAX_ID=MiniMax-M2.7

# 同供应商多模型：继承 KEY/BASE，只换 ID
MODEL_MINIMAX_PRO_INHERIT=minimax
MODEL_MINIMAX_PRO_ID=MiniMax-M2.7-Pro

# 不同供应商：完整声明
MODEL_DEEPSEEK_KEY=sk-yyy
MODEL_DEEPSEEK_BASE=https://api.deepseek.com/v1
MODEL_DEEPSEEK_ID=deepseek-chat
```

### MCP 动态加载器

读取项目根目录 `.mcp.json`，连接外部 MCP 服务器：

```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp@latest"],
      "transport": "stdio"
    }
  }
}
```

特性：
- 持久 session（工具调用复用同一连接）
- 热重载（检测配置文件变化，自动重新加载）
- MCP 工具自动分配给所有 Agent

### 工具输出预算

固定上限 + 截断感知提示（对标 Claude Code）：

```
read_file      → 30,000 字符
run_command    → 10,000 字符
search_content →  8,000 字符
...
截断时自动提示 agent：内容被截断，请用 offset/limit 分段读取
```

### RAG 混合检索

```python
from tools.rag_tool import rag_search, ingest

# 第一步：入库（扫描文件夹 → 切块 → 向量化）
ingest("D:/MyDocs/wiki")

# 第二步：检索（向量 + BM25 + RRF 融合）
result = rag_search("用户的技术栈是什么？")
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
| ⭐ | `agents/chat_agent.py` | Multi-Agent Supervisor + 上下文管理 + MCP + 多模型 |

## 📄 License

MIT License
