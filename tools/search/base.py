"""搜索引擎抽象基类 — 所有搜索引擎统一接口

设计原则：
  - 面向接口编程，Agent 不关心底层用哪个引擎
  - 换引擎只换实现类，调用方零改动
  - SearchResult 统一数据结构，方便格式化输出
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """统一搜索结果数据结构"""
    title: str                           # 页面标题
    url: str                             # 原始链接
    snippet: str                         # 搜索引擎给的摘要
    content: str = ""                    # 正文内容（部分引擎可提取）
    engine: str = ""                     # 来自哪个引擎（方便调试）

    @property
    def has_content(self) -> bool:
        return len(self.content) > 50


class SearchEngine(ABC):
    """搜索引擎抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """引擎名称，用于日志和降级提示"""
        ...

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """执行搜索

        Args:
            query: 搜索关键词
            max_results: 最多返回几条结果

        Returns:
            SearchResult 列表，按相关性排序

        Raises:
            SearchEngineError: 搜索失败时抛出，触发降级
        """
        ...

    def is_available(self) -> bool:
        """检查引擎是否可用（可选重写，用于健康检查）"""
        return True


class SearchEngineError(Exception):
    """搜索引擎异常 — 触发自动降级到备用引擎"""

    def __init__(self, engine_name: str, message: str, original_error: Exception = None):
        self.engine_name = engine_name
        self.original_error = original_error
        super().__init__(f"[{engine_name}] {message}")
