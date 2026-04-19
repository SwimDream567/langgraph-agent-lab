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

# ── stdout 守卫：spinner 活跃时缓冲所有非 spinner 的输出 ──

class _StdoutGuard:
    """当 spinner 活跃时，拦截所有 stdout 写入到缓冲区。
    spinner 线程直接写 _real（真实 stdout），不受拦截。
    stop_spinner 时刷出缓冲区内容。"""
    
    def __init__(self, real):
        self._real = real
        self._buf = []
        self.active = False
        self._lock = threading.Lock()
    
    def write(self, s):
        with self._lock:
            if self.active:
                self._buf.append(s)
                return len(s)
        return self._real.write(s)
    
    def flush(self):
        with self._lock:
            if self.active:
                return
        self._real.flush()

    def __getattr__(self, name):
        """代理所有未定义的属性到真实 stdout（reconfigure, fileno, encoding 等）"""
        return getattr(self._real, name)
    
    def drain(self):
        """取出并清空缓冲区"""
        with self._lock:
            text = ''.join(self._buf)
            self._buf.clear()
            return text

_real_stdout = sys.stdout
_guard = _StdoutGuard(_real_stdout)


def _elapsed(s: float) -> str:
    """秒数 → 可读时间（6s / 1m / 1m30s / 1h / 1h30s / 1h5m30s）"""
    s = int(s)
    if s < 60: return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s}s" if s else f"{m}m"
    h, m = divmod(m, 60)
    if s and m:
        return f"{h}h{m}m{s}s"
    if m:
        return f"{h}h{m}m"
    if s:
        return f"{h}h{s}s"
    return f"{h}h"


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
    """Spinner 后台线程 — 直接写 _real_stdout，不经过 guard"""
    i = 0
    while not _SPIN_STOP.is_set():
        sp = _SP[i % len(_SP)]
        color = _LC.get(_SPIN_LABEL, _A["y"])
        e = _elapsed(time.time() - _SPIN_START)
        timer = f"（{e}）" if time.time() - _SPIN_START >= 5 else ""
        if _SPIN_INFO:
            _real_stdout.write(
                f"\r\033[2K  {color}{sp} {_SPIN_LABEL}...{timer}{_A['0']}\n"
                f"\r\033[2K    {_A['d']}⎿ {_SPIN_INFO}{_A['0']}\033[1A"
            )
        else:
            _real_stdout.write(
                f"\r\033[2K  {color}{sp} {_SPIN_LABEL}...{timer}{_A['0']}"
            )
        _real_stdout.flush()
        _SPIN_STOP.wait(0.08)
        i += 1


def start_spinner(label: str = "Thinking", info: str = ""):
    """启动 spinner 动画（自动停止上一个，拦截 stdout）"""
    global _SPIN_THREAD, _SPIN_LABEL, _SPIN_INFO, _SPIN_START

    # 如果已有 spinner，先停止
    if _SPIN_THREAD and _SPIN_THREAD.is_alive():
        _SPIN_STOP.set()
        _SPIN_THREAD.join(timeout=1)
        _SPIN_STOP.clear()

    _SPIN_LABEL = label
    try: cols = os.get_terminal_size().columns
    except OSError: cols = 80
    _SPIN_INFO = _trunc(info, max(cols - 8, 20))
    _SPIN_START = time.time()
    _SPIN_STOP.clear()

    # 激活 stdout 守卫（拦截所有非 spinner 的输出）
    with _guard._lock:
        _guard._buf.clear()
        _guard.active = True
    sys.stdout = _guard

    _SPIN_THREAD = threading.Thread(target=_spin_loop, daemon=True)
    _SPIN_THREAD.start()


def stop_spinner(success: bool = True):
    """停止 spinner，刷出缓冲内容"""
    global _SPIN_THREAD, _SPIN_INFO

    was = _SPIN_THREAD and _SPIN_THREAD.is_alive()
    if was:
        _SPIN_STOP.set()
        _SPIN_THREAD.join(timeout=2)

    # 关闭 stdout 守卫，恢复真实 stdout
    with _guard._lock:
        _guard.active = False
    sys.stdout = _real_stdout

    # 清除 spinner 帧（双行模式）
    if _SPIN_INFO:
        _real_stdout.write(f"\r\033[2K\033[1B\r\033[2K\033[1A\r\033[2K")
    else:
        _real_stdout.write("\r\033[2K")

    if was:
        mk = f"{_A['g']}●{_A['0']}" if success else f"{_A['r']}●{_A['0']}"
        elapsed = _elapsed(time.time() - _SPIN_START)
        if _SPIN_INFO:
            _real_stdout.write(
                f"  {mk} {_SPIN_LABEL}（{elapsed}）{_A['0']}\n"
                f"    {_A['d']}⎿ {_SPIN_INFO}{_A['0']}\n"
            )
        else:
            _real_stdout.write(f"  {mk} {_SPIN_LABEL}（{elapsed}）{_A['0']}\n")
    _real_stdout.flush()

    # DEBUG: 看看缓冲区里有什么
    buffered = _guard.drain()
    if buffered:
        _real_stdout.write(f"    {_A['d']}[buf: {repr(buffered[:200])}]{_A['0']}\n")
        _real_stdout.flush()

    _SPIN_THREAD = None; _SPIN_INFO = ""
