"""时间工具 — 获取当前日期时间"""

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """获取当前日期、时间和星期几（系统本地时区）"""
    from datetime import datetime
    now = datetime.now()
    try:
        import time as _t
        tz = _t.tzname[0]
    except Exception:
        tz = ""
    return f"{now.strftime('%Y-%m-%d %A %H:%M:%S')} ({tz})"
