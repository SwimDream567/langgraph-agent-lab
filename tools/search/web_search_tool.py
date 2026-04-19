"""联网搜索 Tool — Search 职责：找到相关 URL，给出摘要

工具职责拆分：
  - web_search（Search）：找到相关 URL，返回标题 + 摘要
  - web_fetch（Fetch）：抓取指定 URL 的正文内容

使用方式：
  from tools.search.web_search_tool import web_search

  tools = [get_weather, rag_search, web_search, web_fetch]
"""

from langchain_core.tools import tool

from .engine_factory import SearchEngineFactory
from .base import SearchEngineError

# 工厂单例 — 进程生命周期内只创建一次
_factory = SearchEngineFactory()


@tool
def web_search(query: str) -> str:
    """联网搜索，找到与关键词相关的网页 URL 和摘要。

    当需要查找实时信息、最新新闻、实时数据或核实事实时调用。
    如果需要详细阅读某个页面的完整内容，再用 web_fetch 抓取。

    Args:
        query: 搜索关键词，用简短精准的中文或英文描述
    """
    try:
        results, engine_name = _factory.search(query, max_results=5)

    except SearchEngineError as e:
        return f"搜索失败: {e}"

    if not results:
        return f"未找到与「{query}」相关的结果。"

    # 格式：编号 + 标题 + 摘要 + URL（方便 Agent 提取后传给 web_fetch）
    lines = [f"搜索「{query}」（via {engine_name}，共 {len(results)} 条）\n"]

    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title}")
        if r.snippet:
            lines.append(f"    摘要: {r.snippet}")
        lines.append(f"    URL: {r.url}")
        lines.append("")

    result = "\n".join(lines)

    from tools.output_budget import truncate_output
    return truncate_output("web_search", result)
