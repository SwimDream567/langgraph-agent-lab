# layout.py — 终端底部状态栏（简单版，无固定布局）
#
# 不使用 DECSTBM 滚动区域或 prompt_toolkit 全屏模式。
# 简单可靠：input() 前后没有任何多余渲染。

import sys

from .colors import A as _A


def read_input(prompt_prefix: str = "") -> str:
    """打印简单提示符并读取输入"""
    sys.stdout.write(f"{_A['g']}> {_A['0']}")
    sys.stdout.flush()
    return input()
