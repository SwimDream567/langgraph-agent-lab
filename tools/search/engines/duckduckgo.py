"""DuckDuckGo 搜索引擎实现 — 免费、无需 API Key、兜底方案

特点：
  - 完全免费，无调用次数限制
  - 不需要 API Key，零配置
  - 只有摘要，没有正文内容
  - 适合作为 SearXNG 的兜底引擎

依赖：
  pip install ddgs   # duckduckgo_search 已废弃，请用此库

注意：
  - DuckDuckGo 在部分区域可能不稳定，返回 None 视为不可用
  - 搜索频率过高可能被限流
"""

from ddgs import DDGS

from ..base import SearchEngine, SearchResult, SearchEngineError


class DuckDuckGoEngine(SearchEngine):
    """DuckDuckGo 搜索引擎（兜底方案）"""

    def __init__(self, region: str = "wt-wt", timeout: int = 15):
        # wt-wt = 全球，cn-zh 在部分地区可能失效
        self.region = region
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "DuckDuckGo"

    def is_available(self) -> bool:
        """DuckDuckGo 始终可用（无需自建服务）"""
        return True

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """调用 DuckDuckGo 搜索（新版 ddgs API）"""
        try:
            with DDGS(timeout=self.timeout) as ddgs:
                raw_results = ddgs.text(
                    query=query,
                    region=self.region,
                    max_results=max_results,
                )

        except Exception as e:
            raise SearchEngineError(self.name, f"搜索失败: {e}", e)

        if not raw_results:
            raise SearchEngineError(self.name, f"搜索「{query}」无结果")

        results = []
        for item in raw_results:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", ""),
                    snippet=item.get("body", ""),
                    content="",
                    engine=self.name,
                )
            )

        return results
