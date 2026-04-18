# stream_parser.py — 流式输出解析器（处理思考标签被分包切断）

THINK_TAGS = [
    ("<" + "think" + ">", "</" + "think" + ">"),  # MiniMax / DeepSeek（旧版确认）
    ("<thinking>", "</thinking>"),                 # Qwen
    ("<think/>", "</think/>"),                     # Gemma
]

MAX_BUFFER = 30  # 流式解析 buffer 上限


class StreamParser:
    """识别 <think/> 标签，分流模型输出为正文/思考两类"""

    def __init__(self):
        self.buf = ""
        self.in_think = False
        self.q: list[tuple[bool, str]] = []

    def feed(self, chunk: str) -> list[tuple[bool, str]]:
        self.buf += chunk
        self._drain()
        return self._flush()

    def done(self) -> list[tuple[bool, str]]:
        if self.buf:
            self.q.append((self.in_think, self.buf))
            self.buf = ""
            self.in_think = False
        return self._flush()

    def _tag(self, text: str, opening: bool) -> tuple[int, int]:
        for o, c in THINK_TAGS:
            idx = text.find(o if opening else c)
            if idx >= 0:
                return idx, len(o if opening else c)
        return -1, 0

    def _drain(self):
        while True:
            if self.in_think:
                i, l = self._tag(self.buf, opening=False)
                if i >= 0:
                    if self.buf[:i]:
                        self.q.append((True, self.buf[:i]))
                    self.buf = self.buf[i + l:]
                    self.in_think = False
                else:
                    break
            else:
                i, l = self._tag(self.buf, opening=True)
                if i >= 0:
                    if self.buf[:i]:
                        self.q.append((False, self.buf[:i]))
                    self.buf = self.buf[i + l:]
                    self.in_think = True
                else:
                    max_l = max(len(o) for o, _ in THINK_TAGS)
                    if len(self.buf) > MAX_BUFFER + max_l:
                        self.q.append((False, self.buf))
                        self.buf = ""
                    break

    def _flush(self) -> list[tuple[bool, str]]:
        out = self.q[:]
        self.q.clear()
        return out
