# -*- coding: utf-8 -*-
import sys
# 安全的 UTF-8 重配置：Textual 等 TUI 框架会替换 stdout 为自定义对象（无 reconfigure）
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# RAG 日志开关：被 chat_agent.py 导入时静默，独立运行时输出
_QUIET = not sys.argv[0].endswith("rag_tool.py")
def _log(*args, **kw):
    if not _QUIET:
        print(*args, **kw)

"""RAG 搜索工具 - 混合检索版（向量 + BM25）

架构：企业级分层设计
  ┌─────────────┐     ┌─────────────┐
  │  入库服务     │     │  检索服务     │
  │ (Ingestion)  │     │  (Query)     │
  │ 扫描+切块+向量化│     │ 混合检索+过滤 │
  └──────┬──────┘     └──────┬──────┘
         ↓                   ↓
    ┌─────────────────────────────┐
    │  ChromaDB (持久化向量库)       │
    │  + BM25 (内存关键词索引)       │
    └─────────────────────────────┘

检索流程：
  用户问题
    ├── 向量检索（ChromaDB）→ 语义相似度匹配
    ├── BM25 关键词检索      → 精确匹配（ID、术语、人名）
    └── RRF 融合排序         → 取两者精华，去重后返回 top-k

用法：
  # 方式1：命令行入库（独立步骤）
  python -m tools.rag_tool --ingest /path/to/knowledge/base

  # 方式2：代码调用
  from tools.rag_tool import rag_search
  result = rag_search("用户的技术栈是什么？")
"""

import os
import re
import hashlib
import json
from langchain_core.tools import tool

# ==========================================
# 配置（复用项目 settings）
# ==========================================
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import HF_EMBED_MODEL

# 默认知识库路径（可通过环境变量 KB_PATH 覆盖）
DEFAULT_KB_PATH = os.environ.get("KB_PATH", "")

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = {
    # 文本类
    '.md', '.txt', '.markdown', '.rst', '.log',
    # 代码类
    '.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c', '.h', '.cs',
    '.rb', '.php', '.swift', '.kt', '.scala', '.sh', '.bash', '.ps1',
    '.sql', '.r', '.m', '.lua', '.pl',
    # Web 类
    '.html', '.htm', '.css', '.scss', '.xml', '.yaml', '.yml', '.json',
    '.toml', '.ini', '.cfg', '.conf', '.env',
    # 文档类（需要额外解析）
    '.pdf', '.docx', '.pptx',
    # 数据类
    '.csv', '.tsv',
}


def _is_text_file(ext: str) -> bool:
    """判断是否是纯文本文件（可以直接读取）"""
    return ext in SUPPORTED_EXTENSIONS and ext not in ('.pdf', '.docx', '.pptx')


def _read_file(fpath: str) -> str:
    """通用文件读取：根据扩展名选择解析方式

    支持格式：
      - 纯文本：.md .txt .py .js .java .html .json .yaml ... 等 40+ 种
      - PDF：   .pdf  → PyPDF2
      - Word：  .docx → python-docx
      - PPT：   .pptx → python-pptx
      - CSV：   .csv  → 逐行读取
    """
    ext = os.path.splitext(fpath)[1].lower()

    # ---- 纯文本类（直接读） ----
    if _is_text_file(ext):
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        for enc in encodings:
            try:
                with open(fpath, 'r', encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        return ""

    # ---- PDF ----
    if ext == '.pdf':
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(fpath)
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
        except Exception as e:
            print(f"  ⚠️ PDF 解析失败 {fpath}: {e}")
            return ""

    # ---- Word (.docx) ----
    if ext == '.docx':
        try:
            from docx import Document
            doc = Document(fpath)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            # 也提取表格内容
            tables_text = []
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text for cell in row.cells)
                    if row_text.strip():
                        tables_text.append(row_text)
            return "\n".join(paragraphs + tables_text)
        except Exception as e:
            print(f"  ⚠️ DOCX 解析失败 {fpath}: {e}")
            return ""

    # ---- PPT (.pptx) ----
    if ext == '.pptx':
        try:
            from pptx import Presentation
            prs = Presentation(fpath)
            slides = []
            for i, slide in enumerate(prs.slides, 1):
                texts = []
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for para in shape.text_frame.paragraphs:
                            if para.text.strip():
                                texts.append(para.text)
                if texts:
                    slides.append(f"[PPT{i}]\n" + "\n".join(texts))
            return "\n\n".join(slides)
        except Exception as e:
            print(f"  ⚠️ PPTX 解析失败 {fpath}: {e}")
            return ""

    # ---- CSV ----
    if ext in ('.csv', '.tsv'):
        try:
            sep = '\t' if ext == '.tsv' else ','
            with open(fpath, 'r', encoding='utf-8-sig') as f:
                return f.read()
        except Exception:
            try:
                with open(fpath, 'r', encoding='gbk') as f:
                    return f.read()
            except Exception as e:
                print(f"  ⚠️ CSV 解析失败 {fpath}: {e}")
                return ""

    # ---- 不支持的格式 ----
    return ""

# 持久化目录
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(PROJECT_DIR, "chroma_db_rag")
META_FILE = os.path.join(CHROMA_DIR, "ingest_meta.json")
# BM25 持久化：避免每次启动都从头重建（10万条分词很慢）
BM25_INDEX_FILE = os.path.join(CHROMA_DIR, "bm25_index.json")


# ==========================================
# 全局缓存
# ==========================================
_vectorstore = None
_bm25_retriever = None
_embedding = None
# 持久化 BM25 时同步缓存 chunks_meta，用于 _hybrid_search 按 index 查文档
# 结构：list[{"source": str, "start_index": int, "page_content": str}]
_chunks_meta = []


def _get_embedding():
    """懒加载 Embedding 模型"""
    global _embedding
    if _embedding is None:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HUB_HTTP_TIMEOUT"] = "60"

        # 静默 transformers / sentence_transformers / tqdm 的加载日志
        import logging
        import io
        import contextlib
        for _n in ("transformers", "transformers.modeling_utils",
                    "sentence_transformers", "torch", "tqdm"):
            logging.getLogger(_n).setLevel(logging.ERROR)
        # 环境变量禁用 tqdm（最可靠，影响所有 tqdm 实例）
        _old_tqdm = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"

        from langchain_huggingface import HuggingFaceEmbeddings

        # stderr 重定向兜底（抓住漏网之鱼）
        with contextlib.redirect_stderr(io.StringIO()):
            _embedding = HuggingFaceEmbeddings(
                model_name=HF_EMBED_MODEL,
                encode_kwargs={"normalize_embeddings": True},
            )

        # 恢复 tqdm 环境变量
        if _old_tqdm is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = _old_tqdm
        _log(f"[RAG] Embedding 模型: {HF_EMBED_MODEL}")
    return _embedding


def _tokenize_zh(text: str) -> list:
    """中文分词器（正则切分，不依赖 jieba）"""
    return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+|[0-9]+', text.lower())


def _file_hash(fpath: str) -> str:
    """计算文件 MD5，用于增量更新检测"""
    h = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_meta():
    """加载入库元数据"""
    if os.path.exists(META_FILE):
        with open(META_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"files": {}, "kb_path": None}


def _save_meta(meta):
    """保存入库元数据"""
    os.makedirs(os.path.dirname(META_FILE), exist_ok=True)
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _save_bm25_index(tokenized_corpus: list, chunks: list):
    """保存 BM25 分词结果到文件，避免每次启动重建

    为什么不存整个 BM25Okapi 对象（pickle）？
    - pickle 版本强耦合，无法跨 Python 版本/环境使用
    - 只存原始分词结果和文档内容，load 时重建更可控

    为什么存 page_content？
    - ChromaDB 只存向量，BM25 结果需要返回原文
    - 加载缓存后没有 _chunks，需要靠这里存的 page_content 重建 Document 对象
    """
    index_data = {
        "tokenized_corpus": tokenized_corpus,
        # 同时存 page_content，加载后用这些重建 Document 对象
        "chunks_meta": [
            {
                "source": doc.metadata.get("source", ""),
                "start_index": doc.metadata.get("start_index", 0),
                "page_content": doc.page_content,
            }
            for doc in chunks
        ]
    }
    with open(BM25_INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, ensure_ascii=False)
    _log(f"[RAG] BM25 索引已持久化：{len(tokenized_corpus)} 条 -> {BM25_INDEX_FILE}")


def _load_bm25_index():
    """从文件加载 BM25 分词结果，返回 (tokenized_corpus, chunks_meta)

    chunks_meta 结构：list[{"source", "start_index", "page_content"}]
    如果文件不存在或损坏，返回 (None, None) 由调用方走重建逻辑。
    """
    if not os.path.exists(BM25_INDEX_FILE):
        return None, None
    try:
        with open(BM25_INDEX_FILE, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        tokenized_corpus = index_data.get("tokenized_corpus", [])
        chunks_meta = index_data.get("chunks_meta", [])
        return tokenized_corpus, chunks_meta
    except (json.JSONDecodeError, IOError) as e:
        _log(f"[RAG] BM25 索引文件损坏，将重建: {e}")
        return None, None


# ==========================================
# 入库服务（Ingestion）- 独立运行
# ==========================================

def ingest(kb_path: str = None, force: bool = False):
    """扫描文件夹 → 切块 → 向量化 → 存入 ChromaDB

    Args:
        kb_path: 知识库目录路径，默认用 DEFAULT_KB_PATH
        force: 是否强制重建（忽略缓存）

    企业级做法：
        1. 这个函数应该作为独立脚本运行，或者由定时任务触发
        2. 支持增量更新（只处理新增/修改的文件）
        3. 入库完成后，检索服务直接加载现成的向量库
    """
    global _vectorstore, _bm25_retriever, _chunks_meta

    kb_path = kb_path or DEFAULT_KB_PATH
    if not os.path.exists(kb_path):
        _log(f"[RAG] ❌ 目录不存在: {kb_path}")
        return

    embedding = _get_embedding()

    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from rank_bm25 import BM25Okapi

    # ---- 扫描文件（支持所有格式） ----
    doc_files = []
    skip_dirs = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build", "chroma_db_rag"}
    for root, dirs, files in os.walk(kb_path):
        dirs[:] = [d for d in dirs if not d.startswith("_") and d not in skip_dirs]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                doc_files.append(os.path.join(root, fname))

    _log(f"[RAG] 扫描到 {len(doc_files)} 个文件")

    # ---- 增量更新检测 ----
    meta = _load_meta()
    need_rebuild = force or meta.get("kb_path") != kb_path

    if not need_rebuild:
        changed_files = []
        deleted_files = []
        current_paths = {os.path.relpath(f, kb_path) for f in doc_files}
        stored_paths = set(meta.get("files", {}).keys())

        for fpath in doc_files:
            rel_path = os.path.relpath(fpath, kb_path)
            current_hash = _file_hash(fpath)
            stored_hash = meta.get("files", {}).get(rel_path)
            if stored_hash != current_hash:
                changed_files.append(fpath)

        # 检出已删除的文件（存在于 meta 但不存在于磁盘）
        deleted_paths = stored_paths - current_paths
        if deleted_paths:
            deleted_files = list(deleted_paths)
            _log(f"[RAG] 🗑️ 发现 {len(deleted_files)} 个已删除文件: {deleted_files[:3]}{'...' if len(deleted_files) > 3 else ''}")

        if not changed_files and not deleted_files and os.path.exists(
            os.path.join(CHROMA_DIR, "chroma.sqlite3")
        ):
            _log(f"[RAG] ✅ 所有文件未变化，跳过入库")
            # 仍然需要加载 BM25
            _load_vectorstore_and_bm25(kb_path, embedding)
            return

        if changed_files or deleted_files:
            # 有文件变化或删除时，必须重建向量库（ChromaDB 不支持按 metadata 选择性删除向量）
            # 虽然我们也可以实现选择性删除（查 ID → delete_by_ids），但
            # 1. 实现复杂且容易出错
            # 2. 删除向量后 BM25 索引也需要同步更新
            # 3. 增量模式下文档少，全量重建也很快
            reason = []
            if changed_files: reason.append(f"{len(changed_files)} 个变化")
            if deleted_files: reason.append(f"{len(deleted_files)} 个删除")
            _log(f"[RAG] 📝 {', '.join(reason)}，需要重建向量库")
            need_rebuild = True

    # ---- 加载并切块（通用文件读取） ----
    from langchain_core.documents import Document as LCDocument
    all_docs = []
    for fpath in sorted(doc_files):
        ext = os.path.splitext(fpath)[1].lower()
        content = _read_file(fpath)
        if not content or not content.strip():
            continue
        doc = LCDocument(
            page_content=content,
            metadata={"source": os.path.relpath(fpath, kb_path), "ext": ext}
        )
        all_docs.append(doc)
        _log(f"  📄 {os.path.relpath(fpath, kb_path)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(all_docs)
    _log(f"[RAG] 切块完成：{len(chunks)} 个块")

    # ---- 存入 ChromaDB ----
    if need_rebuild:
        # 删除旧库
        if os.path.exists(CHROMA_DIR):
            import shutil
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)

        _log(f"[RAG] 构建向量库: {CHROMA_DIR}")
        _vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=CHROMA_DIR,
        )
    else:
        _load_vectorstore_and_bm25(kb_path, embedding)
        return

    count = _vectorstore._collection.count()
    _log(f"[RAG] 向量库就绪：{count} 条向量")

    # ---- BM25 索引 ----
    tokenized_corpus = [_tokenize_zh(doc.page_content) for doc in chunks]
    _bm25_retriever = BM25Okapi(tokenized_corpus)
    _chunks_meta = [
        {"source": doc.metadata.get("source", ""), "start_index": doc.metadata.get("start_index", 0), "page_content": doc.page_content}
        for doc in chunks
    ]
    _log(f"[RAG] BM25 就绪：{len(chunks)} 篇文档")

    # 持久化 BM25 分词结果，下次启动直接加载，不用重新分词
    _save_bm25_index(tokenized_corpus, chunks)

    # ---- 保存元数据 ----
    # 同步磁盘文件列表到 meta：新增的加入，已删除的移除
    meta = {"kb_path": kb_path, "files": {}}
    for fpath in doc_files:
        rel_path = os.path.relpath(fpath, kb_path)
        try:
            meta["files"][rel_path] = _file_hash(fpath)
        except (FileNotFoundError, PermissionError):
            # 文件可能被并发删除或锁定（如向量库目录被重建）
            pass
    # meta 里残留的已删除文件路径会在下次入库时被 deleted_paths 检测到并触发重建
    # 这里不需要手动删（need_rebuild=True 时整个 CHROMA_DIR 会被清空）
    _save_meta(meta)
    _log(f"[RAG] ✅ 入库完成，元数据已保存")


def _load_vectorstore_and_bm25(kb_path, embedding):
    """从已有的 ChromaDB 加载向量库，BM25 优先读缓存文件（避免每次启动重建）

    加载策略：
      1. 尝试从 bm25_index.json 加载（快速路径）
      2. 文件不存在或损坏 → 从磁盘文件重建分词结果 → 再存缓存
    """
    global _vectorstore, _bm25_retriever, _chunks_meta

    from langchain_chroma import Chroma
    from rank_bm25 import BM25Okapi

    _log(f"[RAG] 加载已有向量库: {CHROMA_DIR}")
    _vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding,
    )

    # ---- 优先从持久化文件加载 BM25 ----
    tokenized_corpus, chunks_meta = _load_bm25_index()

    if tokenized_corpus is not None:
        # 快速路径：直接用缓存，不需要重新读文件 + 分词
        _bm25_retriever = BM25Okapi(tokenized_corpus)
        _chunks_meta = chunks_meta  # 填充全局，供 _hybrid_search 按 index 查文档
        count = _vectorstore._collection.count()
        _log(f"[RAG] 向量库：{count} 条 | BM25：{len(tokenized_corpus)} 条（来自缓存）")
        return

    # ---- 降级路径：缓存不存在，从磁盘重建 ----
    # 此时 chunks_meta 不为空但 tokenized_corpus 为 None（文件损坏）
    # 需要重新分词并存入缓存，下次就不需要了
    _log("[RAG] BM25 缓存不存在，开始重建...")

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    skip_dirs = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build", "chroma_db_rag"}
    doc_files = []
    for root, dirs, files in os.walk(kb_path):
        dirs[:] = [d for d in dirs if not d.startswith("_") and d not in skip_dirs]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                doc_files.append(os.path.join(root, fname))

    from langchain_core.documents import Document as LCDocument
    all_docs = []
    for fpath in sorted(doc_files):
        content = _read_file(fpath)
        if not content or not content.strip():
            continue
        doc = LCDocument(
            page_content=content,
            metadata={"source": os.path.relpath(fpath, kb_path), "ext": os.path.splitext(fpath)[1].lower()}
        )
        all_docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(all_docs)

    # 重建完成后也存缓存（并存入 page_content，供 _hybrid_search 使用）
    tokenized_corpus = [_tokenize_zh(doc.page_content) for doc in chunks]
    _bm25_retriever = BM25Okapi(tokenized_corpus)
    _chunks_meta = [
        {"source": doc.metadata.get("source", ""), "start_index": doc.metadata.get("start_index", 0), "page_content": doc.page_content}
        for doc in chunks
    ]
    _save_bm25_index(tokenized_corpus, chunks)

    count = _vectorstore._collection.count()
    _log(f"[RAG] 向量库：{count} 条 | BM25：{len(chunks)} 篇（已重建并缓存）")


# ==========================================
# 检索服务（Query）- Agent 调用
# ==========================================

def _ensure_loaded():
    """确保向量库和 BM25 已加载（懒加载）

    注意：_chunks_meta 也必须已填充，由 ingest()/_load_vectorstore_and_bm25 保证。
    这里不需要额外 global，因为只调用 ingest() 修改全局状态。
    """
    global _chunks_meta
    if _vectorstore is not None and _bm25_retriever is not None and _chunks_meta:
        return
    ingest()  # 自动入库（如果还没入的话）


def _rrf_merge(vector_results, bm25_results, k=60):
    """RRF 融合排序"""
    scores = {}
    for rank, (doc, _score) in enumerate(vector_results):
        key = (doc.metadata.get("source", ""), doc.metadata.get("start_index", 0))
        if key not in scores:
            scores[key] = 0
        scores[key] += 1.0 / (k + rank + 1)

    for rank, doc in enumerate(bm25_results):
        key = (doc.metadata.get("source", ""), doc.metadata.get("start_index", 0))
        if key not in scores:
            scores[key] = 0
        scores[key] += 1.0 / (k + rank + 1)

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_keys, scores


def _hybrid_search(query: str, top_k: int = 4):
    """混合检索：向量 + BM25 + RRF + 相关性过滤"""
    _ensure_loaded()

    from langchain_core.documents import Document as LCDocument

    # ---- 向量检索 ----
    vector_docs = _vectorstore.similarity_search_with_score(query, k=top_k * 2)
    vector_results = [(doc, score) for doc, score in vector_docs]

    # ---- BM25 检索 ----
    # 从缓存的 _chunks_meta 按 index 取文档信息，重建 Document 对象
    query_tokens = _tokenize_zh(query)
    bm25_scores = _bm25_retriever.get_scores(query_tokens)
    bm25_top_indices = bm25_scores.argsort()[::-1][:top_k * 2]
    bm25_results = []
    for idx in bm25_top_indices:
        if idx < len(_chunks_meta):
            meta = _chunks_meta[idx]
            bm25_results.append(LCDocument(
                page_content=meta["page_content"],
                metadata={"source": meta["source"], "start_index": meta["start_index"]}
            ))

    # ---- RRF 融合 ----
    doc_map = {}
    for doc, _ in vector_results:
        key = (doc.metadata.get("source", ""), doc.metadata.get("start_index", 0))
        doc_map[key] = doc
    for doc in bm25_results:
        key = (doc.metadata.get("source", ""), doc.metadata.get("start_index", 0))
        doc_map[key] = doc

    sorted_keys, scores = _rrf_merge(vector_results, bm25_results)

    # ---- 相关性过滤 ----
    MAX_DISTANCE = 1.2
    result_docs = []
    for key in sorted_keys[:top_k]:
        doc = doc_map[key]
        distance = None
        for vdoc, vdist in vector_results:
            vkey = (vdoc.metadata.get("source", ""), vdoc.metadata.get("start_index", 0))
            if vkey == key:
                distance = vdist
                break
        if distance is not None and distance > MAX_DISTANCE:
            continue
        result_docs.append((doc, scores[key]))

    return result_docs


@tool
def rag_search(query: str) -> str:
    """Search local knowledge base for documents relevant to a user query. Use when answers require referencing existing knowledge.

    Args:
        query: Search query
    """
    results = _hybrid_search(query, top_k=4)

    if not results:
        return "没有找到相关文档，请尝试换一种问法。"

    lines = []
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("source", "未知来源")
        lines.append(f"[文档{i+1}] 来源: {source}\n{doc.page_content}")

    result = "\n\n".join(lines)
    _log(f"[RAG] 混合检索「{query}」→ 召回 {len(results)} 个文档块 (向量+BM25+RRF)")

    from core.output_budget import truncate_output
    return truncate_output("rag_search", result)


@tool
def add_knowledge(folder_path: str) -> str:
    """Add documents from a folder to the knowledge base (force rebuild index). Path must be absolute.

    Args:
        folder_path: Absolute path to the folder containing documents to ingest
    """
    if not os.path.isabs(folder_path):
        return "请提供绝对路径"
    if not os.path.isdir(folder_path):
        return f"目录不存在: {folder_path}"
    ingest(folder_path, force=True)
    return "入库完成！"


def create_rag_tool():
    """创建 LangGraph Agent 可用的 Tool 对象"""
    from langchain_core.tools import Tool

    description = (
        "当用户提出的问题可能需要参考已有文档或资料时，"
        "先调用这个工具搜索知识库获取相关内容，再回答。不要自己编造答案。"
    )

    return Tool(
        name="rag_search",
        description=description,
        func=rag_search,
    )


# ==========================================
# 命令行入口
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='RAG 工具')
    parser.add_argument('--ingest', '-i', type=str, default=None,
                        help='入库：指定知识库目录路径')
    parser.add_argument('--force', '-f', action='store_true',
                        help='强制重建向量库')
    parser.add_argument('--query', '-q', type=str, default=None,
                        help='检索：测试查询')
    args = parser.parse_args()

    if args.ingest:
        # 入库模式
        ingest(args.ingest, force=args.force)
    elif args.query:
        # 检索模式
        print(rag_search(args.query))
    else:
        # 默认测试
        test_questions = [
            "用户的技术栈是什么？",
            "知识库里有关于项目经历的信息吗？",
            "今天天气怎么样？",
        ]
        print("=" * 50)
        print("RAG 混合检索测试（向量 + BM25 + RRF）")
        print("=" * 50)
        for q in test_questions:
            print(f"\n❓ {q}")
            print("-" * 40)
            result = rag_search(q)
            preview = result[:300].replace("\n", " ")
            print(f"📦 {preview}{'...' if len(result) > 300 else ''}")
