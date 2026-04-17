"""SearXNG 搜索引擎实现 — 自建、隐私优先、无限制

SearXNG 是什么：
  - 开源元搜索引擎，聚合 Google/Bing/DuckDuckGo 等多个引擎的结果
  - 自建部署，完全免费，无调用次数限制
  - 支持 JSON API 输出，天然适合 AI Agent

部署方式：
  1. Docker 一键部署：docker run -d -p 8888:8080 searxng/searxng
  2. 配置 settings.yml 启用 JSON 格式：
     search:
       formats:
         - html
         - json
  3. 本项目中 .env 配置 SEARXNG_BASE_URL=http://localhost:8888
"""

import requests
from typing import Optional

from ..base import SearchEngine, SearchResult, SearchEngineError


class SearXNGEngine(SearchEngine):
    """SearXNG 搜索引擎

    特点：
      - 自建，完全免费无限制
      - 聚合多个搜索引擎结果
      - 支持中文搜索
      - 可配置 categories（general/images/news 等）
    """

    def __init__(
        self,
        base_url: str,
        categories: str = "general",
        language: str = "zh-CN",
        timeout: int = 15,
    ):
        self.base_url = base_url.rstrip("/")
        self.categories = categories
        self.language = language
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "SearXNG"

    def is_available(self) -> bool:
        """健康检查：探测 SearXNG 服务是否在线"""
        try:
            resp = requests.get(
                f"{self.base_url}/healthz",
                timeout=5,
                headers={"User-Agent": "QianQianAgent/1.0"},
            )
            return resp.status_code == 200
        except Exception:
            return False

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """调用 SearXNG JSON API 搜索

        API 文档：https://docs.searxng.org/dev/search_api.html
        """
        try:
            resp = requests.get(
                f"{self.base_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": self.categories,
                    "language": self.language,
                },
                headers={
                    "User-Agent": "QianQianAgent/1.0",
                    "Accept": "application/json",
                },
                timeout=self.timeout,
            )

            if resp.status_code != 200:
                raise SearchEngineError(
                    self.name,
                    f"HTTP {resp.status_code}: {resp.text[:200]}",
                )

            data = resp.json()

        except requests.exceptions.ConnectionError as e:
            raise SearchEngineError(
                self.name,
                f"无法连接 SearXNG 服务 ({self.base_url})，请确认服务已启动",
                e,
            )
        except requests.exceptions.Timeout as e:
            raise SearchEngineError(self.name, "搜索超时（15s）", e)
        except requests.exceptions.JSONDecodeError as e:
            raise SearchEngineError(self.name, "返回数据格式异常", e)
        except SearchEngineError:
            raise
        except Exception as e:
            raise SearchEngineError(self.name, f"未知错误: {e}", e)

        # 解析结果
        results = []
        for item in data.get("results", [])[:max_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    content="",  # SearXNG 不直接返回正文，只有摘要
                    engine=self.name,
                )
            )

        if not results:
            raise SearchEngineError(self.name, f"搜索「{query}」无结果")

        return results
