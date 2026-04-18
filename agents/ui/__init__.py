"""UI package — 终端动画、流式解析、会话渲染"""
from .colors import A
from .spinner import start_spinner, stop_spinner
from .stream_parser import StreamParser, THINK_TAGS
from .display import type_text, render_session_list, select_session, relative_time
