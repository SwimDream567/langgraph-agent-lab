"""全局配置 - 多模型注册表

.env 格式:
  ACTIVE_MODEL=minimax              ← 当前激活模型名
  MODEL_<名称>_KEY=sk-xxx           ← API Key
  MODEL_<名称>_BASE=https://...     ← API Base URL
  MODEL_<名称>_ID=model-name        ← 模型ID

同供应商多模型（继承，避免重复写 KEY/BASE）:
  MODEL_<名称>_INHERIT=基础模型名    ← 从哪个模型继承 KEY/BASE
  MODEL_<名称>_ID=model-name        ← 只写不同的 ID 即可

切模型：改 .env 的 ACTIVE_MODEL，或用 /models 命令运行时切换
加模型：在 .env 照格式加三行 MODEL_<名称>_{KEY,BASE,ID}，重启生效
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ==========================================
# HuggingFace 镜像（国内必须设置）
# ==========================================
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if not os.environ.get("HF_HUB_HTTP_TIMEOUT"):
    os.environ["HF_HUB_HTTP_TIMEOUT"] = "300"

# ==========================================
# Embedding 配置
# ==========================================
HF_EMBED_MODEL = os.environ.get("HF_EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
HF_EMBED_DIM   = int(os.environ.get("HF_EMBED_DIM", "1024"))

# ==========================================
# 多模型注册表
# ==========================================

def _parse_models() -> dict:
    """从环境变量解析所有模型配置

    格式: MODEL_<NAME>_KEY / MODEL_<NAME>_BASE / MODEL_<NAME>_ID
    继承: MODEL_<NAME>_INHERIT=<基础模型名>  自动复制 KEY/BASE
    返回: {name: {key, base, id}, ...}
    """
    models = {}
    inherits = {}  # {name: base_name} 收集继承关系
    for key, value in os.environ.items():
        if not key.startswith("MODEL_") or not value:
            continue
        parts = key[6:]  # 去掉 "MODEL_"
        idx = parts.rfind("_")
        if idx == -1:
            continue
        name = parts[:idx].lower()
        field = parts[idx + 1:].lower()
        if field == "inherit":
            inherits[name] = value.strip().lower()
            continue
        if field not in ("key", "base", "id"):
            continue
        if name not in models:
            models[name] = {}
        models[name][field] = value

    # 处理继承：从基础模型复制缺失的 key/base
    for name, base_name in inherits.items():
        if name not in models:
            models[name] = {}
        if base_name in models:
            for field in ("key", "base"):
                if field not in models[name] and field in models[base_name]:
                    models[name][field] = models[base_name][field]

    # 至少需要 KEY + ID 才算有效
    models = {k: v for k, v in models.items()
              if "key" in v and "id" in v}

    # 兼容旧格式：如果没有 MODEL_* 但有 LLM_*，自动迁移
    if not models:
        _k = os.environ.get("LLM_API_KEY", "")
        _b = os.environ.get("LLM_API_BASE", "")
        _m = os.environ.get("LLM_MODEL", "")
        if _k and _m:
            models["default"] = {"key": _k, "base": _b, "id": _m}

    return models


ALL_MODELS = _parse_models()
ACTIVE_MODEL_NAME = os.environ.get("ACTIVE_MODEL", "").lower()


class ModelRegistry:
    """运行时模型管理：列表、切换、解析引用"""

    def __init__(self):
        self._models = ALL_MODELS
        self._names = list(ALL_MODELS.keys())
        self._active = ACTIVE_MODEL_NAME
        if self._active not in self._models and self._names:
            self._active = self._names[0]

    # --- 属性 ---

    @property
    def active_name(self) -> str:
        return self._active

    @property
    def active_config(self) -> dict:
        return self._models.get(self._active, {})

    # --- 查询 ---

    def list_all(self) -> list:
        """返回 [(name, config), ...] 保持 .env 定义顺序"""
        return [(n, self._models[n]) for n in self._names]

    def resolve(self, ref: str) -> str:
        """解析引用（名称 / 序号 / 部分匹配）→ 模型名"""
        ref = ref.strip().lower()
        # 1. 精确名称
        if ref in self._models:
            return ref
        # 2. 序号
        try:
            idx = int(ref) - 1
            if 0 <= idx < len(self._names):
                return self._names[idx]
        except ValueError:
            pass
        # 3. 前缀匹配
        matches = [n for n in self._names if n.startswith(ref)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"多个模型匹配 '{ref}': {', '.join(matches)}")
        raise ValueError(
            f"未知模型: {ref}（可用: {', '.join(self._names)}）")

    # --- 切换 ---

    def switch(self, ref: str) -> tuple:
        """切换模型。返回 (旧名称, 新名称)"""
        name = self.resolve(ref)
        old = self._active
        self._active = name
        return old, name


# ==========================================
# 兼容旧代码的属性导出
# ==========================================
_registry = ModelRegistry()
_active = _registry.active_config

API_KEY    = _active.get("key", "")
API_BASE   = _active.get("base", "")
MODEL_FAST = _active.get("id", "gpt-4o-mini")
MODEL_STRONG = os.environ.get("LLM_MODEL_STRONG", MODEL_FAST)
