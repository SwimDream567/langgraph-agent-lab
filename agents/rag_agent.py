"""
RAG Agent - 完整版
=================

功能：
    1. 从 Markdown 文件加载文档（用简历作为测试）
    2. 多种切块策略对比（256 / 512 / 1024）
    3. MMR 检索模式（Maximum Marginal Relevance，避免检索结果太相似）
    4. 带来源标注的 RAG 回答
    5. 封装成工具，接入 LangGraph StateGraph Agent

技术选型：
    - Embedding：HuggingFace paraphrase-multilingual-MiniLM-L12-v2（本地免费，中文优化，384 维）
    - LLM：MiniMax GLM（复用项目配置）
    - 向量库：ChromaDB（本地持久化）
    - 工具封装：LangGraph Tool

对应学习计划：
    - Day 4 (4.14) - RAG 基础
    - Day 5 (4.15) - RAG 进阶

依赖安装：
    pip install langchain-huggingface chromadb langchain-chroma langchain-openai langchain-community
    pip install langchain-community  # 用于文档加载器
"""

import os
import sys
from typing import Literal

# ============================================================
# 第一部分：配置
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import API_KEY, API_BASE, MODEL_FAST, HF_EMBED_MODEL, HF_EMBED_DIM

# ---- LLM 配置（对话用）----
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=MODEL_FAST,
    api_key=API_KEY,
    base_url=API_BASE,
    temperature=0.3,          # RAG 场景降低创意度，更专注事实
)

# ---- HuggingFace Embedding 配置（向量化用）----
# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2：
#   384 维，50+ 语言，中文效果好，免费本地运行
# 国内 HuggingFace 连接慢时自动用 hf-mirror.com 镜像
import os
if os.environ.get("HF_ENDPOINT") is None:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embedding_model = HuggingFaceEmbeddings(
    model_name=HF_EMBED_MODEL,
    model_kwargs={"device": "cuda"},            # GPU 加速
    encode_kwargs={"normalize_embeddings": True},  # 归一化
)

# 测试 embedding（顺便预热模型，首次会下载/加载）
print(f"[INFO] Embedding 模型: {HF_EMBED_MODEL}（HuggingFace 本地）")
test_vec = embedding_model.embed_query("测试")
print(f"[INFO] Embedding 维度: {len(test_vec)}（预期 {HF_EMBED_DIM}）")

# 测试文档路径
DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
RESUME_PATH = os.path.join(DOCS_DIR, "..", "..", "resume_new.html")

# ============================================================
# 第二部分：文档加载
# ============================================================
def load_documents(file_path: str):
    """
    加载文档文件

    LangChain 支持多种加载器：
        - TextLoader: 纯文本，encoding 指定编码
        - UnstructuredMarkdownLoader: Markdown，保留标题等结构信息
        - PyPDFLoader: PDF
        - DocxLoader: Word 文档
        - SeleniumURLLoader: 网页

    metadata 的作用：给每个文档块打标签，方便后续追踪来源
    例如：{"source": "resume.md", "section": "工作经历"}
    """
    if not os.path.exists(file_path):
        print(f"[WARN] 文件不存在: {file_path}，使用内置测试文档")
        return None

    # Markdown 文件用 UnstructuredMarkdownLoader，能保留标题结构
    if file_path.endswith(".md"):
        loader = UnstructuredMarkdownLoader(file_path, mode="elements")
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()
    print(f"[INFO] 加载文档: {file_path}")
    print(f"[INFO] 文档块数量: {len(docs)}")
    print(f"[INFO] 前100字符: {docs[0].page_content[:100]}...")
    return docs


# ============================================================
# 第三部分：文本切块（Chunking）
# ============================================================
def split_documents(
    documents,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    file_source: str = "document"
):
    """
    文本切块 - 把大文档切成小段落

    参数说明：
        chunk_size: 每个块的最大字符数
            - 太小（<100）：丢失上下文，检索精准但回答片面
            - 太大（>1000）：包含太多无关内容，检索不精准
            - 经验值：256-512 适合问答场景

        chunk_overlap: 块之间的重叠字符数
            - 重叠让语义不被切断
            - 但重叠太多浪费 token，通常 10-20%

    切块策略（RecursiveCharacterTextSplitter）：
        按 separators 列表的顺序递归切分：
            1. 先按 \\n\\n（段落）切
            2. 再按 \\n（行）切
            3. 再按句子（。！？）切
            4. 最后按字符数限制切
        这样尽量保持语义完整性
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 优先按段落 > 换行 > 句子 > 字符的顺序切
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
        add_start_index=True,
    )

    # split_documents 输入 Document 列表，输出切好的 Document 列表
    chunks = text_splitter.split_documents(documents)

    # 给每个 chunk 的 metadata 加上来源信息
    for chunk in chunks:
        chunk.metadata["source"] = file_source

    print(f"[INFO] 切块完成: {len(chunks)} 个块")
    print(f"[INFO] 块大小: {chunk_size} 字符, 重叠: {chunk_overlap} 字符")
    return chunks


def compare_chunk_sizes(documents, sizes=[256, 512, 1024]):
    """
    对比不同切块大小的效果
    用于 Day 5 的切块策略优化

    对比维度：
        1. 块数量：越大块数越少
        2. 语义完整性：大块保留更多上下文
        3. 检索精准度：小块检索更精准，但可能丢失上下文
    """
    print("\n" + "=" * 50)
    print("切块策略对比实验")
    print("=" * 50)

    for size in sizes:
        chunks = split_documents(documents, chunk_size=size, chunk_overlap=int(size * 0.1))
        total_chars = sum(len(c.page_content) for c in chunks)
        avg_chars = total_chars / len(chunks) if chunks else 0
        print(f"  size={size}: {len(chunks)} 块, 平均 {avg_chars:.0f} 字符/块")


# ============================================================
# 第四部分：构建向量数据库
# ============================================================
def build_vectorstore(chunks, persist_dir: str = None, collection_name: str = "rag"):
    """
    把文档块向量化，存入 ChromaDB

    from_texts vs from_documents：
        - from_texts: 输入字符串列表 + 元数据列表
        - from_documents: 输入 Document 对象列表（推荐，自动保留 metadata）

    Chroma 的底层：
        - 文档 → HuggingFace Embedding模型 → 向量（384 维 for paraphrase-multilingual）
        - 向量存入库，原始文本存在 metadata 里
        - 检索时：query → Embedding → 在向量空间里找最近的 k 个
    """
    if persist_dir is None:
        persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

    # 清理旧数据
    import shutil
    if os.path.exists(persist_dir):
        print(f"[INFO] 清理旧向量库: {persist_dir}")
        shutil.rmtree(persist_dir)

    # 向量化并存入 Chroma（使用 HuggingFace Embeddings）
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    count = vectorstore._collection.count()
    print(f"[INFO] 向量库构建完成: {count} 条向量")
    print(f"[INFO] 存储路径: {persist_dir}")
    return vectorstore


def load_vectorstore(persist_dir: str, collection_name: str = "rag"):
    """
    加载已有的向量库（避免每次重新向量化）
    实际项目每次重启时从磁盘加载，不需要重新处理文档
    """
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name=collection_name,
    )


# ============================================================
# 第五部分：检索器（Retriever）配置
# ============================================================
def create_retriever(
    vectorstore,
    search_type: Literal["similarity", "mmr"] = "similarity",
    k: int = 3,
    fetch_k: int = 10,
):
    """
    创建检索器

    search_type 两种模式：
        1. similarity（默认）：只返回最相似的 k 个
            - 优点：最相关的通常排前面
            - 缺点：结果可能太相似（都是同一个主题），缺少多样性

        2. MMR（Maximum Marginal Relevance）：
            - 先从向量库取 fetch_k 个候选
            - 再从中选 k 个：既考虑相关性，又考虑多样性
            - 适合文档库很大的场景，避免返回一堆重复的内容
    """
    search_kwargs = {"k": k}

    if search_type == "mmr":
        # MMR 模式：先取 fetch_k 个候选，再从中选 k 个最均衡的
        search_kwargs["fetch_k"] = fetch_k

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    mode_name = "MMR" if search_type == "mmr" else "Similarity"
    print(f"[INFO] 检索器创建: {mode_name}, k={k}")
    return retriever


# ============================================================
# 第六部分：RAG Chain - 核心问答链路
# ============================================================
def build_rag_chain(retriever, prompt_template: str = None):
    """
    构建 RAG Chain（检索 → Prompt → LLM → 回答）

    LangChain LCEL 语法：
        {"key1": component1, "key2": component2}  →  组装输入字典
        | prompt                                   →  把输入传给 Prompt 渲染
        | llm                                      →  把 Prompt 发给 LLM
        | output_parser                            →  解析 LLM 输出

    RunnablePassthrough：
        - 让输入的某个字段原封不动传下去
        - 这里是让 "question" 字段跳过 transform，直接传给下一步
    """

    # 默认 Prompt（带来源标注）
    if prompt_template is None:
        prompt_template = """你是一个基于文档的问答助手。
严格只使用提供的上下文信息来回答问题。
如果上下文中没有答案，请说"我没有足够的信息来回答这个问题"。
在回答时，用 [来源] 标注每个信息点的出处。

上下文信息：
{context}

用户问题：{question}

请基于上下文回答："""

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("human", "{question}"),
    ])

    def format_docs(docs: list) -> str:
        """把检索到的 Document 列表格式化成带编号的字符串"""
        return "\n\n".join(
            f"[文档{i+1}] {doc.page_content}\n[来源]: {doc.metadata.get('source', '未知')}"
            for i, doc in enumerate(docs)
        )

    # RAG Chain 组装
    rag_chain = (
        {
            # 检索：question → retrieved docs
            # format_docs：doc list → context string
            # question：原样传递
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ============================================================
# 第七部分：RAG 工具 - 接入 LangGraph Agent
# ============================================================
def create_rag_tool(vectorstore, name: str = "rag_search", description: str = None):
    """
    把 RAG 检索器封装成 LangGraph 工具

    为什么需要封装成工具？
        - 在 StateGraph Agent 里，工具是 Agent "调用外部能力" 的方式
        - 给 Agent 一个"文档检索"工具，它就能在回答问题前先查文档
        - 工具定义 = name（调用的名字）+ description（Agent 理解什么时候该用）
    """
    if description is None:
        description = (
            "当用户问到关于简历、项目经历、技术栈、工作经验等问题时，"
            "先使用这个工具搜索相关文档获取信息，再回答用户问题。"
        )

    # LangGraph 工具格式：name + description + callable
    from langchain_core.tools import Tool

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10},
    )

    def rag_search(query: str) -> str:
        """
        RAG 搜索工具的具体实现

        输入：用户的自然语言问题
        输出：检索到的相关文档片段（格式化为字符串）
        """
        docs = retriever.invoke(query)
        if not docs:
            return "没有找到相关文档，请尝试换一种问法。"
        return "\n\n".join(f"[文档{i+1}] {d.page_content}" for i, d in enumerate(docs))

    tool = Tool(
        name=name,
        description=description,
        func=rag_search,
    )

    print(f"[INFO] RAG 工具创建: {name}")
    return tool


# ============================================================
# 第八部分：主函数 - 演示完整流程
# ============================================================
def main():
    print("=" * 60)
    print("RAG Agent - 完整演示（HuggingFace Embedding）")
    print("=" * 60)

    # --- 步骤 1：加载文档 ---
    print("\n[步骤 1] 加载文档...")
    docs = load_documents(RESUME_PATH)
    if docs is None:
        # 使用内置测试文档（当简历文件不存在时）
        from langchain_core.documents import Document
        docs = [
            Document(page_content="游梦是一名 Java 后端开发工程师，熟悉 Spring Boot、Flowable 工作流、Nacos 微服务。", metadata={"source": "简历"}),
            Document(page_content="他有全栈开发经验，前端使用 Vue 3 + Vite + Pinia + Element Plus。", metadata={"source": "简历"}),
            Document(page_content="他在广西塔易技术公司实习了 5 个月，参与教学管理系统开发。", metadata={"source": "简历"}),
            Document(page_content="他正在学习 AI Agent 开发，主攻 LangGraph，目标在深圳找 AI Agent 岗位。", metadata={"source": "简历"}),
            Document(page_content="期望薪资 6-8k，地点深圳，技术栈包括 Java、Python、Vue。", metadata={"source": "简历"}),
        ]

    # --- 步骤 2：切块 ---
    print("\n[步骤 2] 切块...")
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=30)

    # --- 步骤 3：构建向量库 ---
    print("\n[步骤 3] 向量化 + 存入 ChromaDB（HuggingFace paraphrase-multilingual）...")
    persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    vectorstore = build_vectorstore(chunks, persist_dir)

    # --- 步骤 4：创建检索器 ---
    print("\n[步骤 4] 创建检索器（对比 Similarity vs MMR）...")

    # Similarity 模式：只返回最相似的 k 个
    retriever_sim = create_retriever(vectorstore, search_type="similarity", k=3)
    # MMR 模式：既考虑相关性，又考虑多样性
    retriever_mmr = create_retriever(vectorstore, search_type="mmr", k=3, fetch_k=10)

    # --- 步骤 5：构建 RAG Chain ---
    print("\n[步骤 5] 构建 RAG Chain...")
    rag_chain_sim = build_rag_chain(retriever_sim)
    rag_chain_mmr = build_rag_chain(retriever_mmr)

    # --- 步骤 6：对比测试 ---
    print("\n" + "=" * 60)
    print("RAG 问答测试")
    print("=" * 60)

    questions = [
        "游梦想找什么工作？",
        "他用什么技术栈？",
        "他在哪家公司实习过？",
        "今天深圳天气怎么样？",  # 文档里没有，应该回答不知道
    ]

    for q in questions:
        print(f"\n{'='*40}")
        print(f"❓ 问题: {q}")

        print(f"\n[Similarity 模式]:")
        answer = rag_chain_sim.invoke(q)
        print(f"🤖 回答: {answer}")

        print(f"\n[MMR 模式]:")
        answer = rag_chain_mmr.invoke(q)
        print(f"🤖 回答: {answer}")

    # --- 步骤 7：RAG 工具演示 ---
    print("\n" + "=" * 60)
    print("RAG 工具封装演示（可接入 LangGraph Agent）")
    print("=" * 60)

    rag_tool = create_rag_tool(vectorstore)
    print(f"\n[工具信息]")
    print(f"  名称: {rag_tool.name}")
    print(f"  描述: {rag_tool.description}")

    print(f"\n[工具调用测试]")
    tool_result = rag_tool.func("游梦的技术栈是什么？")
    print(f"  检索结果: {tool_result[:200]}...")

    print("\n✅ RAG Agent 演示完成！")
    print("""
接下来可以：
    1. 把 RAG 工具接入 StateGraph Agent（agents/full_agent.py）
    2. 对比不同 Embedding 模型的效果（nomic-embed-text vs mxbai-embed-large）
    3. 添加重排（Reranking）提升检索精度
    4. 接入真实简历文件测试
    """)


if __name__ == "__main__":
    main()
