"""搜索引擎工厂 — 根据配置创建引擎，支持自动降级

设计模式：工厂模式 + 降级策略

降级逻辑：
  1. 优先使用 .env 中配置的主引擎（SearXNG）
  2. 如果主引擎不可用（服务未启动/超时），自动降级到备用引擎（DuckDuckGo）
  3. 降级过程对 Agent 透明，Agent 只关心搜索结果

配置示例（.env）：
  SEARCH_ENGINE=searxng          # 主引擎
  SEARXNG_BASE_URL=http://localhost:8888
  SEARCH_FALLBACK=duckduckgo     # 备用引擎
"""

import os
import logging
from typing import Optional

from .base import SearchEngine, SearchEngineError
from .engines.searxng import SearXNGEngine
from .engines.duckduckgo import DuckDuckGoEngine

logger = logging.getLogger(__name__)


def _create_searxng(base_url: Optional[str] = None) -> SearXNGEngine:
    """创建 SearXNG 引擎实例"""
    url = base_url or os.environ.get("SEARXNG_BASE_URL", "http://localhost:8888")
    return SearXNGEngine(base_url=url)


def _create_duckduckgo() -> DuckDuckGoEngine:
    """创建 DuckDuckGo 引擎实例"""
    return DuckDuckGoEngine()


# 引擎注册表 — 新增引擎只需要在这里加一行
_ENGINE_REGISTRY = {
    "searxng": _create_searxng,
    "duckduckgo": _create_duckduckgo,
}


class SearchEngineFactory:
    """搜索引擎工厂 — 创建主引擎 + 管理降级"""

    def __init__(self):
        self._primary_name: str = ""
        self._primary: Optional[SearchEngine] = None
        self._fallback: Optional[SearchEngine] = None
        self._setup()

    def _setup(self):
        """从环境变量读取配置，初始化引擎"""
        primary_name = os.environ.get("SEARCH_ENGINE", "searxng").lower()
        fallback_name = os.environ.get("SEARCH_FALLBACK", "duckduckgo").lower()

        # 创建主引擎
        if primary_name in _ENGINE_REGISTRY:
            if primary_name == "searxng":
                self._primary = _create_searxng()
            else:
                self._primary = _ENGINE_REGISTRY[primary_name]()
            self._primary_name = primary_name
        else:
            logger.warning(f"未知引擎: {primary_name}，降级到 duckduckgo")
            self._primary = _create_duckduckgo()
            self._primary_name = "duckduckgo"

        # 创建备用引擎
        if fallback_name in _ENGINE_REGISTRY and fallback_name != self._primary_name:
            if fallback_name == "searxng":
                self._fallback = _create_searxng()
            else:
                self._fallback = _ENGINE_REGISTRY[fallback_name]()
        else:
            # 确保始终有兜底
            self._fallback = _create_duckduckgo()

    @property
    def primary_name(self) -> str:
        return self._primary_name

    def search(self, query: str, max_results: int = 5) -> tuple[list, str]:
        """执行搜索，带自动降级

        Returns:
            (results, engine_name) — 搜索结果 + 实际使用的引擎名
        """
        # 尝试主引擎
        try:
            if self._primary and self._primary.is_available():
                results = self._primary.search(query, max_results)
                logger.info(f"[{self._primary.name}] 搜索成功: {query}")
                return results, self._primary.name
        except SearchEngineError as e:
            logger.warning(f"主引擎 {self._primary.name} 失败: {e}")

        # 降级到备用引擎
        if self._fallback:
            try:
                results = self._fallback.search(query, max_results)
                logger.info(f"[{self._fallback.name}] 降级搜索成功: {query}")
                return results, f"{self._fallback.name}（降级）"
            except SearchEngineError as e:
                logger.error(f"备用引擎 {self._fallback.name} 也失败: {e}")
                raise SearchEngineError(
                    "AllEngines",
                    f"主引擎和备用引擎均失败: {e}",
                )

        raise SearchEngineError("Factory", "无可用搜索引擎")
