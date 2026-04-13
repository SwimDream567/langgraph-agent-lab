"""全局配置 - API Key 和模型参数

这里集中管理所有 API 配置，避免在每个文件里重复写 Key。
类似 Java 里的 application.yml

⚠️ API Key 从 .env 文件读取，不要硬编码到代码里！
"""

# ==========================================
# 环境变量加载（必须在最前面）
# ==========================================
import os
from dotenv import load_dotenv
load_dotenv()

# ==========================================
# HuggingFace 镜像（国内必须设置）
# ==========================================
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if not os.environ.get("HF_HUB_HTTP_TIMEOUT"):
    os.environ["HF_HUB_HTTP_TIMEOUT"] = "300"  # 5分钟超时

# ==========================================
# GLM API（智谱，兼容 OpenAI 格式）
# ==========================================
GLM_API_KEY = os.environ.get("GLM_API_KEY", "")
GLM_API_BASE = os.environ.get("GLM_API_BASE", "https://open.bigmodel.cn/api/coding/paas/v4")

GLM_FAST = "glm-5.1"       # 快速便宜，日常用
GLM_STRONG = "glm-5.1"      # 更强但更贵，复杂任务用

# ==========================================
# MiniMax API（兼容 OpenAI 格式）
# ==========================================
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_API_BASE = os.environ.get("MINIMAX_API_BASE", "https://api.minimaxi.com/v1")

MINIMAX_CHAT = "MiniMax-M2.7"     # MiniMax 文本对话模型

# ==========================================
# Embedding 模型配置
# ==========================================

# Ollama 本地 Embedding（免费，英文为主）
# OLLAMA_EMBED_MODEL = "nomic-embed-text"
# OLLAMA_EMBED_DIM = 768

# HuggingFace 本地 Embedding（免费，中文优化，推荐）
# BAAI/bge-large-zh-v1.5：1024维，中文 MTEB 霸榜，语义表达能力强
# 模型已缓存在 ~/.cache/huggingface/hub/，首次运行自动下载（约 1.2GB）
HF_EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
HF_EMBED_DIM = 1024  # 输出 1024 维向量，中文语义表达更强

# OpenAI Embedding（收费，仅当以上都不可用时启用）
# EMBEDDING_MODEL = "text-embedding-3-small"  # 1536维
# EMBEDDING_MODEL = "text-embedding-3-large"  # 3072维

# ==========================================
# 默认配置（改这里切换模型）
# ==========================================
# API_KEY = GLM_API_KEY
# API_BASE = GLM_API_BASE
# MODEL_FAST = GLM_FAST
# MODEL_STRONG = GLM_STRONG
API_KEY = MINIMAX_API_KEY
API_BASE = MINIMAX_API_BASE
MODEL_FAST = MINIMAX_CHAT
MODEL_STRONG = GLM_STRONG
