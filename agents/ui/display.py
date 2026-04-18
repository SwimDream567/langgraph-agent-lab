# display.py — 终端渲染：打字机效果、会话列表、交互选择
from __future__ import annotations

import random, time
from typing import TYPE_CHECKING

from .colors import A as _A

if TYPE_CHECKING:
    from agents.session_manager import SessionManager, Session


# ── 打字机效果 ────────────────────────────────────────

def type_text(text: str, delay: float = 0.025):
    for c in text:
        print(c, end="", flush=True)
        time.sleep(delay * (0.5 + random.random()))


# ── 相对时间 ──────────────────────────────────────────

def relative_time(ts: float) -> str:
    """时间戳 → 相对时间描述（刚刚 / 5分钟前 / 2小时前 / 昨天 / 3天前）"""
    e = time.time() - ts
    if e < 60: return "刚刚"
    if e < 3600: return f"{int(e // 60)}分钟前"
    if e < 86400: h = int(e // 3600); return "1小时前" if h == 1 else f"{h}小时前"
    if e < 172800: return "昨天"
    return f"{int(e // 86400)}天前"


# ── 会话列表渲染 ──────────────────────────────────────

def render_session_list(mgr: SessionManager):
    """渲染会话列表（表格式）"""
    ss = mgr.list_sessions()
    aid = mgr.active_id
    print(f"\n{_A['c']}{_A['b']}{'#':<4} {'名称':<20} {'ID':<12} {'更新'}{_A['0']}")
    print(f"{_A['d']}{'─' * 52}{_A['0']}")
    for i, s in enumerate(ss, 1):
        ia = s.thread_id == aid
        p = f"{_A['g']}*{_A['0']}" if ia else " "
        nc = _A["g"] if ia else _A["gr"]
        nm = s.name[:18] if len(s.name) <= 18 else s.name[:16] + ".."
        ts = relative_time(s.updated_at)
        mk = f"{p}{nc}{i:<3}{_A['0']}"
        if ia:
            print(f"{mk} {_A['b']}{_A['c']}{nm:<20}{_A['0']} {s.thread_id:<12} {_A['d']}{ts}{_A['0']}")
        else:
            print(f"{mk} {nm:<20} {s.thread_id:<12} {_A['gr']}{ts}{_A['0']}")
    cur = mgr.active_session
    print(f"\n  {_A['g']}当前:{_A['0']} {cur.name if cur else '默认'} · 共 {len(ss)} 个会话")


# ── 交互式会话选择 ────────────────────────────────────

async def select_session(mgr: SessionManager) -> Session | None:
    """交互式选择会话（输入序号或名称）"""
    ss = mgr.list_sessions()
    if not ss:
        print(f"\n  {_A['y']}没有可切换的会话，输入 /new 创建{_A['0']}")
        return None
    aid = mgr.active_id
    print(f"\n  {_A['c']}{_A['b']}选择会话{_A['0']}{_A['d']} (Enter确认 Esc取消){_A['0']}")
    for i, s in enumerate(ss, 1):
        ia = s.thread_id == aid
        mk = f"{_A['g']}*{_A['0']}" if ia else " "
        dn = s.name[:24] if len(s.name) <= 24 else s.name[:22] + ".."
        if ia:
            print(f"  {mk} {_A['b']}{_A['g']}{i}.{_A['0']} {_A['b']}{dn}{_A['0']}  {_A['d']}{relative_time(s.updated_at)}{_A['0']}")
        else:
            print(f"  {mk} {i}. {dn}  {_A['gr']}{relative_time(s.updated_at)}{_A['0']}")
    while True:
        try:
            ch = input(f"\r  {_A['c']}> {_A['0']}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_A['d']}(已取消){_A['0']}")
            return None
        if not ch: continue
        try:
            idx = int(ch)
            if 1 <= idx <= len(ss): return ss[idx - 1]
            print(f"  {_A['r']}序号超出范围 (1-{len(ss)}){_A['0']}")
        except ValueError:
            try: return mgr.switch_by_name(ch)
            except ValueError as e: print(f"  {_A['r']}{e}{_A['0']}")
