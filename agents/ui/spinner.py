# spinner.py — 终端加载动画（Braille 旋转 + 耗时显示 + 工具信息截断）
import os, sys, time, threading

from .colors import A as _A

# ── 模块内部状态 ──────────────────────────────────────
_SPIN_STOP = threading.Event()
_SPIN_LABEL = "Thinking"
_SPIN_INFO = ""
_SPIN_THREAD: threading.Thread | None = None
_SPIN_START = 0.0
_SP = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_LC = {"Thinking": _A["y"], "ToolCalling": _A["c"]}

# ── 工具函数 ──────────────────────────────────────────

def _elapsed(s: float) -> str:
    """秒数 → 可读时间（6s / 1m30s / 1h5m30s）"""
    s = int(s)
    if s < 60: return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60: return f"{m}m{s}s"
    h, m = divmod(m, 60)
    return f"{h}h{m}m{s}s"


def _cw(s: str) -> int:
    """计算字符串终端显示宽度（CJK/全角 = 2 列）"""
    w = 0
    for c in s:
        cp = ord(c)
        w += 2 if (0x4E00 <= cp <= 0x9FFF or 0x3000 <= cp <= 0x303F or
                    0xFF01 <= cp <= 0xFF60 or 0x2018 <= cp <= 0x201F) else 1
    return w


def _trunc(text: str, avail: int) -> str:
    """按终端宽度截断，中间用 ... 连接"""
    if _cw(text) <= avail: return text
    h2 = (avail - 3) // 2
    def _w(c): cp = ord(c); return 0x4E00 <= cp <= 0x9FFF or 0x3000 <= cp <= 0x303F or 0xFF01 <= cp <= 0xFF60 or 0x2018 <= cp <= 0x201F
    head, w = [], 0
    for c in text:
        cw = 2 if _w(c) else 1
        if w + cw > h2: break
        head.append(c); w += cw
    tail, w = [], 0
    for c in reversed(text):
        cw = 2 if _w(c) else 1
        if w + cw > h2: break
        tail.append(c); w += cw
    return ''.join(head) + '...' + ''.join(reversed(tail))


# ── Spinner 核心 ──────────────────────────────────────

def _spin_loop():
    i = 0
    while not _SPIN_STOP.is_set():
        sp = _SP[i % len(_SP)]
        color = _LC.get(_SPIN_LABEL, _A["y"])
        e = _elapsed(time.time() - _SPIN_START)
        timer = f"（{e}）" if time.time() - _SPIN_START >= 5 else ""
        if _SPIN_INFO:
            sys.stdout.write(f"\r\033[2K  {color}{sp} {_SPIN_LABEL}...{timer}{_A['0']}\n"
                             f"\r\033[2K    {_A['d']}⎿ {_SPIN_INFO}{_A['0']}\033[1A")
        else:
            sys.stdout.write(f"\r\033[2K  {color}{sp} {_SPIN_LABEL}...{timer}{_A['0']}")
        sys.stdout.flush()
        time.sleep(0.08)
        i += 1


def start_spinner(label: str = "Thinking", info: str = ""):
    """启动 spinner 动画（自动停止上一个）"""
    global _SPIN_THREAD, _SPIN_LABEL, _SPIN_INFO, _SPIN_START
    if _SPIN_THREAD and _SPIN_THREAD.is_alive():
        _SPIN_STOP.set(); _SPIN_THREAD.join(timeout=1); _SPIN_STOP.clear()
    _SPIN_LABEL = label
    try: cols = os.get_terminal_size().columns
    except OSError: cols = 80
    _SPIN_INFO = _trunc(info, max(cols - 8, 20))
    _SPIN_START = time.time()
    _SPIN_STOP.clear()
    sys.stdout.write("\n"); sys.stdout.flush()
    _SPIN_THREAD = threading.Thread(target=_spin_loop, daemon=True)
    _SPIN_THREAD.start()


def stop_spinner(success: bool = True):
    """停止 spinner，留下 ● Thinking（耗时）"""
    global _SPIN_THREAD, _SPIN_INFO
    was = _SPIN_THREAD and _SPIN_THREAD.is_alive()
    if was:
        _SPIN_STOP.set(); _SPIN_THREAD.join(timeout=2)
        mk = f"{_A['g']}●{_A['0']}" if success else f"{_A['r']}●{_A['0']}"
        elapsed = _elapsed(time.time() - _SPIN_START)
        if _SPIN_INFO:
            sys.stdout.write(f"\r\033[2K  {mk} {_SPIN_LABEL}（{elapsed}）{_A['0']}\n"
                             f"\r\033[2K    {_A['d']}⎿ {_SPIN_INFO}{_A['0']}\n")
        else:
            sys.stdout.write(f"\r\033[2K  {mk} {_SPIN_LABEL}（{elapsed}）{_A['0']}\n")
        sys.stdout.flush()
    _SPIN_THREAD = None; _SPIN_INFO = ""
