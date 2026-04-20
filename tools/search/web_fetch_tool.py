"""网页内容抓取 Tool — Fetch 职责：根据 URL 抓取完整正文

工具职责拆分：
  - web_search（Search）：找到相关 URL，返回标题 + 摘要
  - web_fetch（Fetch）：抓取指定 URL 的正文内容

使用方式：
  from tools.search.web_fetch_tool import web_fetch

  tools = [get_weather, rag_search, web_search, web_fetch]
"""

import httpx
from langchain_core.tools import tool

try:
    from markdownify import markdownify as html_to_md
except ImportError:
    html_to_md = None


MAX_URLS = 3          # 最多抓取 3 个 URL，避免耗时过长
MAX_CHARS_PER_PAGE = 3000  # 每个页面最多取前 3000 字符


def _fetch_one(url: str) -> str:
    """抓取单个 URL，返回 Markdown 格式正文"""
    try:
        resp = httpx.get(
            url,
            timeout=15,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,*/*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
            follow_redirects=True,
        )
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            return f"[{url}]\n内容类型不支持: {content_type}"

        text = resp.text[:MAX_CHARS_PER_PAGE]

        if html_to_md:
            md = html_to_md(text, heading_style="atx")
        else:
            # 没有 markdownify 时，粗暴去掉 HTML 标签
            import re
            md = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
            md = re.sub(r"<style[^>]*>.*?</style>", "", md, flags=re.DOTALL)
            md = re.sub(r"<[^>]+>", "", md)
            md = re.sub(r"\n{3,}", "\n\n", md).strip()

        return f"[{url}]\n{md}"

    except httpx.Timeout:
        return f"[{url}]\n抓取超时（15s）"
    except httpx.HTTPStatusError as e:
        return f"[{url}]\nHTTP 错误: {e.response.status_code}"
    except Exception as e:
        return f"[{url}]\n抓取失败: {e}"


@tool
def web_fetch(urls: list[str]) -> str:
    """Fetch webpage content from URLs and return as Markdown. Use after web_search finds relevant URLs.
    Max 3 URLs per call. Each page truncated at 3000 chars.

    Args:
        urls: List of URLs to fetch, e.g. ["https://example.com", "https://foo.com"]
    """
    if not urls:
        return "未提供 URL 列表。"

    # 限制数量
    urls = urls[:MAX_URLS]

    results = []
    for url in urls:
        results.append(_fetch_one(url))

    result = "\n\n---\n\n".join(results)

    from core.output_budget import truncate_output
    return truncate_output("web_fetch", result)
