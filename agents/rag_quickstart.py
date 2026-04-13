"""
RAG 快速起步 - Obsidian 笔记版
===============================

直接从 Obsidian 第二大脑加载笔记作为 RAG 文档库。

技术选型：
    - Embedding：Ollama 本地模型 nomic-embed-text（完全免费，中英文支持好）
    - LLM：MiniMax（复用项目已有的 API 配置）
    - 向量库：ChromaDB（本地存储，零配置）
    - 文档源：Obsidian 笔记（游梦的个人知识库）

文档来源（Obsidian wiki）：
    - 00-用户信息.md       — 基本资料、技术栈、偏好
    - 01-项目记录.md       — OpenClaw 迁移、B站视频制作等项目
    - 10-AI-Agent学习路线/00-学习计划.md  — 学习计划 v2、简历诊断
    - 10-AI-Agent学习路线/daily/Day1 2026-04-12.md — Day1 学习记录

运行：
    python agents/rag_quickstart.py
"""

import os
import sys

# ============================================================
# 第一步：配置
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import API_KEY, API_BASE, MODEL_FAST, OLLAMA_EMBED_MODEL, OLLAMA_EMBED_DIM

# ---- Ollama Embedding（向量化用，本地免费）----
from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
print(f"[INFO] Embedding: {OLLAMA_EMBED_MODEL}（Ollama 本地）")

# ---- LLM（对话用）----
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=MODEL_FAST,
    api_key=API_KEY,
    base_url=API_BASE,
    temperature=0.7,
)

# ============================================================
# 第二步：加载 Obsidian 笔记（文档源）
# ============================================================
# Obsidian 的 Markdown 文件用 UnstructuredMarkdownLoader 加载
# 保留标题等结构信息到 metadata，方便后续追溯来源
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Obsidian wiki 根目录
OBSIDIAN_WIKI = r"D:\MyObsidian\MyObsidian\wiki"

# 递归扫描 wiki 下所有 .md 文件（排除 index.md 等总览性文件）
OBSIDIAN_DOC_FILES = []
for root, dirs, files in os.walk(OBSIDIAN_WIKI):
    for fname in files:
        if fname.endswith(".md") and not fname.startswith("index"):
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, OBSIDIAN_WIKI)
            # 排除根目录的说明性文件（信息密度低）
            if rel not in ("00-用户信息.md",) and not rel.startswith("_"):
                OBSIDIAN_DOC_FILES.append(fpath)

print(f"[INFO] Obsidian wiki 路径: {OBSIDIAN_WIKI}")
print(f"[INFO] 扫描到 {len(OBSIDIAN_DOC_FILES)} 个笔记文件：")
for fp in sorted(OBSIDIAN_DOC_FILES):
    rel = os.path.relpath(fp, OBSIDIAN_WIKI)
    print(f"  📄 {rel}")

# 逐个加载并打印内容摘要
all_docs = []
for file_path in OBSIDIAN_DOC_FILES:
    if os.path.exists(file_path):
        # UnstructuredMarkdownLoader 能保留 YAML frontmatter 和标题结构
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        # 在 metadata 里补上文件名（相对路径），方便追溯来源
        for doc in docs:
            doc.metadata["source"] = os.path.relpath(file_path, OBSIDIAN_WIKI)
        all_docs.extend(docs)
        print(f"  ✅ {os.path.relpath(file_path, OBSIDIAN_WIKI)} — {len(docs)} 个片段")
    else:
        print(f"  ⚠️  文件不存在: {file_path}")

print(f"[INFO] 共加载 {len(OBSIDIAN_DOC_FILES)} 个笔记文件 → {len(all_docs)} 个文档片段")

# ============================================================
# 第三步：文本切块
# ============================================================
# 块大小的经验值：
#   - 200-300 字符：完整句子，适合精确问答
#   - 500 字符：段落级别，适合摘要类问题
#   - overlap 保持 10-20%，防止句子被切断
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=40,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    add_start_index=True,
)

chunks = text_splitter.split_documents(all_docs)

print(f"[INFO] 切块完成：{len(chunks)} 个块")
print(f"[INFO] 块大小：250 字符，重叠：40 字符")

# 打印每个块的来源和前50字预览（方便调试）
print("\n[块预览]")
for i, chunk in enumerate(chunks[:8]):
    source = chunk.metadata.get("source", "未知")
    preview = chunk.page_content[:50].replace("\n", " ")
    print(f"  块{i+1} [{source}]: {preview}...")

# ============================================================
# 第四步：存入 ChromaDB
# ============================================================
persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db_obsidian")

import shutil
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=persist_dir,
    collection_metadata={"source": "Obsidian wiki"},
)

print(f"\n[INFO] 向量库已构建：{vectorstore._collection.count()} 条向量")
print(f"[INFO] 存储路径: {persist_dir}")

# ============================================================
# 第五步：检索器
# ============================================================
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},   # 召回 4 个块
)

# 测试检索
test_query = "游梦的基本资料和技术栈是什么？"
retrieved_docs = retriever.invoke(test_query)
print(f"\n[RETRIEVAL] 查询: {test_query}")
print(f"[RETRIEVAL] 召回 {len(retrieved_docs)} 个块：")
for i, doc in enumerate(retrieved_docs):
    source = doc.metadata.get("source", "未知")
    print(f"  块{i+1} [{source}]: {doc.page_content[:60].replace(chr(10), ' ')}...")

# ============================================================
# 第六步：构建 RAG Chain
# ============================================================
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个基于个人知识库的问答助手。
严格只使用提供的上下文信息来回答问题。
如果上下文中没有答案，请直接说"我没有足够的信息来回答这个问题"。
不要编造答案。
上下文信息来自 Obsidian 第二大脑笔记。"""),
    ("human", """上下文信息：
{context}

用户问题：{question}

请基于上下文回答："""),
])

def format_docs(docs: list) -> str:
    """格式化成带来源标注的字符串"""
    return "\n\n".join(
        f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
        for d in docs
    )

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ============================================================
# 第七步：RAG 问答测试
# ============================================================
print("\n" + "=" * 60)
print("Obsidian RAG 问答测试")
print("=" * 60)

questions = [
    "游梦是谁？他的基本资料是什么？",
    "他的技术栈有哪些？",
    "他在哪家公司实习过？",
    "他正在学习什么？学习计划是什么？",
    "他有什么项目经历？",
    "今天天气怎么样？",
    "他的 GPA 是多少？",
    "他的期望薪资是多少？",
]

for q in questions:
    print(f"\n❓ 问题: {q}")
    print(f"🤖 回答: ", end="", flush=True)
    answer = rag_chain.invoke(q)
    print(answer)
    print("-" * 40)

print("\n✅ Obsidian RAG 快速起步完成！")
print("""
这个版本直接读取 Obsidian 笔记，信息量比模拟字符串丰富得多。

接下来可以：
    1. 把 RAG 工具接入 StateGraph Agent（agents/full_agent.py）
    2. 增加更多 Obsidian 笔记（项目记录、学习日志等）
    3. 对比不同块大小的效果
    4. 添加 memory 模块让 Agent 记住对话历史
""")
