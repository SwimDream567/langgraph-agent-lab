#!/usr/bin/env python3
"""
tui_app.py — My Agent 2.0 Textual TUI
基于 Textual 8.x 构建的全屏聊天终端界面

布局：
┌──────────────────────────────────────────────────────────┐
│ 🤖 My Agent 2.0 │ MiniMax M2.7 │ 会话:默认 │ 16:00     │ ← Header
├──────────────────────────────────────────────────────────┤
│                                                          │
│                                                          │
│ 帮我看看当前目录有什么文件                                │
│                                                          │ ← 消息滚动区
│                                                          │   (Markdown + 代码高亮)
│  ⏋ Thinking（3s）                                        │
│  ⌇ ToolCalling（2s）⎿ list_dir                           │
│  ┌──────────────────────┐                                │
│  │ agents/  # Agent核心  │                                │
│  └──────────────────────┘                                │
│                                                          │
├──────────────────────────────────────────────────────────┤
│ > 输入消息或命令...                        [Tab 补全]    │ ← 固定输入框
├──────────────────────────────────────────────────────────┤
│ ctx: 32% ████░░░░░░ │ tokens: 1,234 │ MiniMax │ ← 状态栏
└──────────────────────────────────────────────────────────┘

特性：
- Markdown 渲染 + 代码语法高亮（Textual 原生 Markdown 组件）
- 输入框始终在底部（不随消息滚动）
- 命令补全（Tab 键）
- 底部状态栏（模型/token/上下文进度）
- /models 后上下键选择模型
- 流式输出（Markdown.get_stream）
"""

from __future__ import annotations

import os
import sys
import time
import asyncio
import threading
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING

# ── Windows UTF-8 兼容 ──────────────────────────────────
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for _sn in ("stdout", "stderr"):
        _stream = getattr(sys, _sn)
        # 只包装真实的文件流（有 .buffer 属性的），
        # Textual 的 _PrintCapture 等自定义流不处理
        if _stream and hasattr(_stream, "buffer") and hasattr(_stream, "encoding"):
            if _stream.encoding.lower() not in ("utf-8", "utf8"):
                import io as _io
                setattr(sys, _sn, _io.TextIOWrapper(
                    _stream.buffer, encoding="utf-8", errors="replace", line_buffering=True
                ))

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, VerticalScroll, Center
from textual.screen import ModalScreen
from textual.widgets import (
    Header, Footer, Input, Static, Markdown,
    OptionList, Label,
)
from textual.widgets.option_list import Option
from textual.reactive import reactive
from textual import work
from textual.message import Message

from rich.text import Text
from rich.markdown import Markdown as RichMarkdown

if TYPE_CHECKING:
    from agents.session_manager import SessionManager


# ══════════════════════════════════════════════════════════════════════════════
# 自定义 Widget
# ══════════════════════════════════════════════════════════════════════════════

class ChatMessage(Static):
    """单条消息气泡"""

    class Action(Message):
        """消息触发的操作"""
        def __init__(self, action: str, data: str = "") -> None:
            super().__init__()
            self.action = action
            self.data = data

    def __init__(self, role: str, content: str, **kwargs):
        self.role = role
        self._content = content
        css_class = f"msg-{role}"
        super().__init__(**kwargs, classes=css_class)

    def on_mount(self):
        self._show_content(self._content)

    def _show_content(self, content: str):
        if self.role in ("user", "system", "tool", "error", "warning", "success", "info"):
            self.update(content)
        else:
            # Agent 回复用 Markdown 渲染
            self.update(RichMarkdown(content))


class ThinkingWidget(Static):
    """思考内容累积显示（灰色，完成后留在原地）"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs, classes="msg-thinking")
        self._buf: str = ""

    def append(self, text: str):
        """追加一段思考内容（整段）"""
        self._buf += text
        self.update(f"[dim]{self._buf}[/dim]")

    def append_char(self, ch: str):
        """追加单个字符（打字机效果用）"""
        self._buf += ch
        self.update(f"[dim]{self._buf}[/dim]")


class SpinnerWidget(Static):
    """Spinner 状态指示器：Thinking=橙色，ToolCalling=蓝色，完成后留下 ●"""

    # 颜色方案：label → 动画/文字颜色（hex 颜色确保跨终端兼容）
    _LABEL_COLORS = {
        "Thinking": "#FFD700",       # 金黄色
        "ToolCalling": "#63B8FF",    # 钢蓝色
    }
    _DEFAULT_COLOR = "#CCCCCC"

    def __init__(self, **kwargs):
        super().__init__(**kwargs, classes="spinner")
        self._label = ""
        self._info = ""
        self._start_time = 0.0
        self._tick_timer = None
        self._active = False

    @property
    def _color(self) -> str:
        return self._LABEL_COLORS.get(self._label, self._DEFAULT_COLOR)

    def start(self, label: str = "Thinking", info: str = ""):
        self._label = label
        self._info = info
        self._active = True
        self._start_time = time.time()
        self._start_tick()

    def stop(self, success: bool = True):
        self._active = False
        self._stop_tick()
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        dot_color = "#00CC00" if success else "#FF6600"
        c = self._color
        # 超过 5 秒才显示计时
        time_part = f"[{c}]（{_fmt_time(elapsed)}）[/{c}]" if elapsed >= 5 else ""
        line1 = f"[{dot_color}]●[/{dot_color}] [{c}]{self._label}[/{c}]{time_part}"
        if self._info:
            # ⚠️ 不能用 f-string！[/dim] 里包含 {dim} 会被 Python 当表达式解析
            self.update(line1 + "\n   [dim]⎿[/dim] " + self._info)
        else:
            self.update(line1)

    def update_info(self, info: str):
        self._info = info
        if self._active:
            self._refresh()

    def _refresh(self):
        if not self._active:
            return
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        idx = int(time.time() * 10) % len(spinner_chars)
        ch = spinner_chars[idx]
        c = self._color
        # 超过 5 秒才显示计时
        time_part = f"[{c}]（{_fmt_time(elapsed)}）[/{c}]" if elapsed >= 5 else ""
        line1 = f"[{c}]{ch} {self._label}[/{c}]{time_part}"
        if self._info:
            # ⚠️ 不能用 f-string！[/dim] 里包含 {dim} 会被 Python 当表达式解析
            self.update(line1 + "\n   [dim]⎿[/dim] " + self._info)
        else:
            self.update(line1)

    def _start_tick(self):
        self._stop_tick()
        self._tick_timer = self.set_interval(0.1, self._refresh)

    def _stop_tick(self):
        if self._tick_timer is not None:
            self._tick_timer.stop()
            self._tick_timer = None


class InfoSidebar(Vertical):
    """Left sidebar — shows model, session, token info when terminal is wide enough."""

    _model: reactive[str] = reactive("...")
    _tokens: reactive[int] = reactive(0)
    _ctx_pct: reactive[int] = reactive(0)
    _session: reactive[str] = reactive("default")

    def __init__(self, **kwargs):
        super().__init__(**kwargs, classes="info-sidebar", id="info-sidebar")

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]Info[/bold cyan]", classes="sidebar-header")
        # Each row: label left, value right
        with Horizontal(classes="sb-row"):
            yield Label("Model", classes="sb-label")
            yield Label("...", classes="sb-value", id="sb-model")
        with Horizontal(classes="sb-row"):
            yield Label("Session", classes="sb-label")
            yield Label("default", classes="sb-value", id="sb-session")
        with Horizontal(classes="sb-row"):
            yield Label("Tokens", classes="sb-label")
            yield Label("0", classes="sb-value", id="sb-tokens")
        with Horizontal(classes="sb-row"):
            yield Label("Context", classes="sb-label")
            yield Label("0%", classes="sb-value", id="sb-ctx")

    def watch__model(self, val: str):
        try:
            # Extract model ID: "minimax (MiniMax-M2.7)" → "MiniMax-M2.7"
            mid = val
            if " (" in val and val.endswith(")"):
                _, mid = val.rsplit(" (", 1)
                mid = mid.rstrip(")")
            self.query_one("#sb-model", Label).update(mid)
        except Exception:
            pass

    def watch__tokens(self, val: int):
        try:
            self.query_one("#sb-tokens", Label).update(f"{val:,}")
        except Exception:
            pass

    def watch__ctx_pct(self, val: int):
        try:
            color = "green" if val < 60 else "yellow" if val < 85 else "red"
            self.query_one("#sb-ctx", Label).update(f"[{color}]{val}%[/{color}]")
        except Exception:
            pass

    def watch__session(self, val: str):
        try:
            self.query_one("#sb-session", Label).update(val)
        except Exception:
            pass

    def update_status(self, *, model: str = None, tokens: int = None,
                      ctx_pct: int = None, session: str = None):
        if model is not None:
            self._model = model
        if tokens is not None:
            self._tokens = tokens
        if ctx_pct is not None:
            self._ctx_pct = ctx_pct
        if session is not None:
            self._session = session


class StatusBar(Static):
    """Bottom status bar — responsive: 1 row when wide, 2 rows when narrow."""

    _model: reactive[str] = reactive("...")
    _tokens: reactive[int] = reactive(0)
    _ctx_pct: reactive[int] = reactive(0)
    _session: reactive[str] = reactive("default")

    _COMPACT_WIDTH = 72  # below this → split into two rows

    def __init__(self, **kwargs):
        super().__init__("", **kwargs, classes="status-bar")

    # ── rendering ──────────────────────────────────────────────────────
    _show_content: bool = True  # False when sidebar is visible (wide screen)

    def _build_text(self) -> str:
        if not self._show_content:
            return ""
        w = self.size.width or 80
        model = self._model
        tokens = self._tokens
        session = self._session
        pct = self._ctx_pct

        bar_len = 10
        filled = int(pct / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        color = "green" if pct < 60 else "yellow" if pct < 85 else "red"
        ctx_str = f"ctx: [{color}]{bar}[/{color}] {pct}%"

        if w >= self._COMPACT_WIDTH:
            return f" 🤖 {model}  tokens: {tokens:,}  {ctx_str}  session: {session} "
        else:
            return (
                f" 🤖 {model}  tokens: {tokens:,}\n"
                f" {ctx_str}  session: {session} "
            )

    def _refresh_display(self):
        try:
            self.update(self._build_text())
        except Exception:
            pass

    def refresh_layout(self):
        """Called by parent on resize / visibility change."""
        self.call_after_refresh(self._refresh_display)

    # ── reactive watches ───────────────────────────────────────────────
    def watch__model(self, val: str):
        self._refresh_display()

    def watch__tokens(self, val: int):
        self._refresh_display()

    def watch__ctx_pct(self, val: int):
        self._refresh_display()

    def watch__session(self, val: str):
        self._refresh_display()

    def update_status(self, *, model: str = None, tokens: int = None,
                      ctx_pct: int = None, session: str = None):
        if model is not None:
            self._model = model
        if tokens is not None:
            self._tokens = tokens
        if ctx_pct is not None:
            self._ctx_pct = ctx_pct
        if session is not None:
            self._session = session


# ══════════════════════════════════════════════════════════════════════════════
# 命令面板 — 输入 / 时自动弹出，过滤匹配命令
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# 弹窗选择器（ModalScreen）— 用于 /models、/sessions 等
# ══════════════════════════════════════════════════════════════════════════════

class SelectorModal(ModalScreen[str | None]):
    """弹窗式选择器，支持输入过滤。

    返回值：
        - str: 选中的 option id
        - None: 用户按 Esc 取消
    """

    DEFAULT_CSS = """
    SelectorModal {
        align: center middle;
    }
    #modal-container {
        width: 50;
        max-width: 96;
        height: auto;
        max-height: 25;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    #modal-title {
        text-align: center;
    }
    #modal-list-wrap {
        height: auto;
        max-height: 15;
        overflow-y: auto;
    }
    #modal-option-list {
        height: auto;
    }
    #modal-hint {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        title: str,
        options: list[dict],  # [{"id": str, "label": Text|str, "disabled": bool}]
        *,
        option_id_prefix: str = "opt-",
    ):
        super().__init__()
        self._title = title
        self._option_id_prefix = option_id_prefix
        self._options = options

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Label(f"[bold cyan]{self._title}[/bold cyan]", id="modal-title")
            yield Label(
                "[dim]↑↓ Navigate · Enter Confirm · Esc Cancel[/dim]",
                id="modal-hint",
            )
            opt_items = []
            for opt in self._options:
                label = opt["label"] if isinstance(opt["label"], str) else str(opt["label"])
                disabled = opt.get("disabled", False)
                oid = f"{self._option_id_prefix}{opt['id']}"
                opt_items.append(Option(label, id=oid, disabled=disabled))
            with Vertical(id="modal-list-wrap"):
                yield OptionList(*opt_items, id="modal-option-list")

    def on_mount(self) -> None:
        ol = self.query_one("#modal-option-list", OptionList)
        ol.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        oid = event.option_id
        if oid and oid.startswith(self._option_id_prefix):
            value = oid[len(self._option_id_prefix):]
            self.dismiss(value)

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.prevent_default()
            event.stop()


class InteractiveSelector(Vertical):
    """Generic interactive selector — reusable for model selection, session switching, Human-in-the-Loop, etc.

    When allow_input=True, adds a "chat about this" option as the last item.
    Navigating to this option reveals an inline input field for free-form text.

    Keyboard controls:
      ↑↓  Navigate options
      Enter  Confirm selection (or submit typed input)
      Esc   Cancel/dismiss
      Type  (when chat option highlighted) Custom free-form input

    Usage:
        def on_selector_selected(event: InteractiveSelector.Selected): ...
        def on_dismissed(event: InteractiveSelector.Dismissed): ...
        def on_user_input(event: InteractiveSelector.UserInput): ...
    """

    _CHAT_OPTION_ID = "__chat__"

    class Selected(Message):
        """User selected an option or confirmed typed input."""
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    class Dismissed(Message):
        """User pressed Esc to dismiss without selection."""
        def __init__(self) -> None:
            super().__init__()

    class UserInput(Message):
        """User typed free-form input and pressed Enter."""
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(
        self,
        title: str,
        options: list[dict],  # [{"id": str, "label": Text|str, "disabled": bool}]
        *,
        allow_input: bool = False,
        placeholder: str = "Type a custom value...",
        option_id_prefix: str = "opt-",
        **kwargs
    ):
        super().__init__(**kwargs, classes="interactive-selector")
        self._title = title
        self._allow_input = allow_input
        self._placeholder = placeholder
        self._option_id_prefix = option_id_prefix
        self._chat_idx: int = -1  # index of the "chat about this" option; -1 = not used

        if allow_input:
            self._chat_idx = len(options)
            chat_num = len(options) + 1
            options = list(options)  # copy to avoid mutating caller's list
            options.append({
                "id": self._CHAT_OPTION_ID,
                "label": Text.from_markup(f"[cyan]  {chat_num}. chat about this[/]"),
            })

        self._options = options
        self._just_highlighted = False  # True when highlight just changed (click → highlight only)

    def compose(self) -> ComposeResult:
        yield Label(
            f"[bold cyan]{self._title}[/bold cyan]"
            f"\n(↑↓ Select · Enter Confirm · Esc Cancel"
            f"{', · Type to input' if self._allow_input else ''})",
            classes="selector-title",
        )

        opt_items = []
        for opt in self._options:
            label = opt["label"] if isinstance(opt["label"], str) else str(opt["label"])
            disabled = opt.get("disabled", False)
            oid = f"{self._option_id_prefix}{opt['id']}"
            opt_items.append(Option(label, id=oid, disabled=disabled))

        yield OptionList(*opt_items, id="selector-option-list")

        # Inline input — only created when allow_input, visible when chat option highlighted
        if self._allow_input:
            inp = Input(
                placeholder=self._placeholder,
                classes="selector-input",
                id="selector-input",
            )
            inp.display = False  # hidden by default; shown via Highlighted event
            yield inp

    def on_mount(self):
        """Set default highlight — always focus OptionList for navigation."""
        try:
            ol = self.query_one("#selector-option-list", OptionList)
            if self._chat_idx >= 0:
                ol.highlight(self._chat_idx)
                # Show input visually but keep focus on OptionList
                inp = self.query_one("#selector-input", Input)
                inp.display = True
            ol.focus()
        except Exception:
            pass
        # Reset flag so first keyboard Enter works (on_mount highlight set the flag)
        self._just_highlighted = False

    # ── Inline input visibility ──────────────────────────────

    def _show_inline_input(self):
        """Show the inline input (visual only, does NOT steal focus)."""
        try:
            self.query_one("#selector-input", Input).display = True
        except Exception:
            pass

    def _hide_inline_input(self):
        """Hide the inline input."""
        try:
            self.query_one("#selector-input", Input).display = False
        except Exception:
            pass

    def _update_input_visibility(self):
        """Show/hide inline input based on currently highlighted option."""
        if self._chat_idx < 0:
            return
        try:
            ol = self.query_one("#selector-option-list", OptionList)
            if ol.highlighted == self._chat_idx:
                self._show_inline_input()
            else:
                self._hide_inline_input()
        except Exception:
            pass

    def _input_has_focus(self) -> bool:
        try:
            inp = self.query_one("#selector-input", Input)
            return inp.has_focus
        except Exception:
            return False

    def _focus_option_list(self):
        try:
            self.query_one("#selector-option-list", OptionList).focus()
        except Exception:
            pass

    # ── Event handlers ───────────────────────────────────────

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """OptionList highlight changed — mark as just-highlighted + update input visibility."""
        self._just_highlighted = True
        if self._chat_idx >= 0:
            self._update_input_visibility()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        # First click on a new option: highlight just changed → ignore, don't confirm
        if self._just_highlighted:
            self._just_highlighted = False
            return

        option_id = event.option_id
        chat_full_id = f"{self._option_id_prefix}{self._CHAT_OPTION_ID}"
        if option_id == chat_full_id:
            # "chat about this" selected via Enter — focus input for typing
            # (if user already typed something and pressed Enter again, submit it)
            try:
                inp = self.query_one("#selector-input", Input)
                if inp.value.strip():
                    self.post_message(self.UserInput(inp.value.strip()))
                else:
                    inp.focus()
            except Exception:
                pass
            return
        if option_id and option_id.startswith(self._option_id_prefix):
            value = option_id[len(self._option_id_prefix):]
            self.post_message(self.Selected(value))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "selector-input":
            text = event.value.strip()
            if text:
                self.post_message(self.UserInput(text))

    def on_key(self, event) -> None:
        # ── Escape ──
        if event.key == "escape":
            if self._input_has_focus():
                # First Esc: leave input, return to OptionList
                self._focus_option_list()
                event.prevent_default()
                return
            # Second Esc (or no input focused): dismiss selector
            self.post_message(self.Dismissed())
            event.prevent_default()

        # ── Up / Down — only handle when input has focus ──
        elif event.key in ("up", "down"):
            # If input has focus, give it back to OptionList so navigation works
            if self._input_has_focus():
                self._focus_option_list()
                # Don't prevent_default — let OptionList handle the actual navigation

        # ── Enter while input focused — submit ──
        elif event.key == "enter":
            if self._input_has_focus():
                try:
                    inp = self.query_one("#selector-input", Input)
                    text = inp.value.strip()
                    if text:
                        self.post_message(self.UserInput(text))
                        event.prevent_default()
                except Exception:
                    pass


class MultiQuestionSurvey(Vertical):
    """Multi-question survey widget — AI asks user multiple questions at once.

    Layout:
        ╭─ Survey Title ───────────────────╮
        │  (←→ Switch · Enter Confirm · Esc)│
        │                                   │
        │ [ ✓ ] Q1: What's your name?      │
        │        Answer: Alice              │
        │                                   │
        │ [ > ] Q2: Preferred approach?     │
        │  ┌────────────────────────────┐   │
        │  │ Type your answer...        │   │
        │  └────────────────────────────┘   │
        │                                   │
        │ [   ] Q3: Any other notes?       │
        ╰───────────────────────────────────╯

    Messages:
        Submitted(results)  — User submitted all answers (or pressed Enter on last)
        Dismissed()         — User pressed Esc
    """

    class Submitted(Message):
        """User submitted the survey."""
        def __init__(self, results: dict[str, str]) -> None:
            super().__init__()
            self.results = results  # {question_id: answer_text}

    class Dismissed(Message):
        """User dismissed without submitting."""
        def __init__(self) -> None:
            super().__init__()

    def __init__(
        self,
        title: str,
        questions: list[dict],  # [{"id": str, "question": str, "options": list[str]|None}]
        *,
        option_id_prefix: str = "survey-",
        **kwargs
    ):
        """
        Args:
            title: Survey title (e.g., "需要确认几个问题")
            questions: List of question dicts:
                - id: unique identifier
                - question: question text
                - options: optional predefined choices (None = free text)
        """
        super().__init__(**kwargs, classes="interactive-selector survey-widget")
        self._title = title
        self._questions = questions
        self._option_id_prefix = option_id_prefix
        self._active_idx = 0
        self._answers: dict[str, str] = {}  # question_id -> answer

    @property
    def _num_questions(self) -> int:
        return len(self._questions)

    def compose(self) -> ComposeResult:
        yield Label(
            f"[bold cyan]{self._title}[/bold cyan]"
            "\n[dim]←/→ Switch question · Enter confirm current · Esc cancel[/dim]",
            classes="selector-title"
        )
        yield from self._compose_question_widgets()

    # ─── Internal: render question widgets ───

    def _compose_question_widgets(self):
        """Yield widgets for all questions based on current state."""
        for i, q in enumerate(self._questions):
            qid = q["id"]
            qtext = q["question"]
            opts = q.get("options")
            is_active = (i == self._active_idx)
            is_answered = qid in self._answers

            if is_answered:
                status_icon = "[green]✓[/green]"
            elif is_active:
                status_icon = "[cyan]>[/cyan]"
            else:
                status_icon = "[dim] [/dim]"

            prefix = f"  {status_icon} "
            num_label = f"[bold]{i+1}.[/bold] "

            if is_active:
                # Active question: highlighted + input area
                yield Static(f"{prefix}{num_label}[reverse]{qtext}[/reverse]", classes="survey-q-active")
                if opts:
                    opt_items = []
                    for j, opt_text in enumerate(opts):
                        oid = f"{self._option_id_prefix}{qid}-opt{j}"
                        opt_items.append(Option(opt_text, id=oid))
                    yield OptionList(*opt_items, id=f"{self._option_id_prefix}{qid}-list")
                else:
                    default_val = q.get("default", "")
                    placeholder = default_val if default_val else "Type your answer..."
                    yield Input(
                        placeholder=placeholder,
                        classes="survey-input",
                        id=f"{self._option_id_prefix}{qid}-input"
                    )
            else:
                # Inactive question: compact line
                if is_answered:
                    ans_preview = self._answers[qid]
                    if len(ans_preview) > 40:
                        ans_preview = ans_preview[:37] + "..."
                    yield Static(
                        f"{prefix}{num_label}{qtext}\n       [dim]{ans_preview}[/dim]",
                        classes="survey-q-done"
                    )
                else:
                    yield Static(
                        f"{prefix}{num_label}[dim]{qtext}[/dim]",
                        classes="survey-q-pending"
                    )

    def _rebuild(self):
        """Rebuild widget content after state change."""
        for child in list(self.children):
            child.remove()
        for w in self.compose():
            self.mount(w)
        # Refocus active question's input/list
        self._refocus_active()

    def _refocus_active(self):
        """Set focus to the active question's interactive element."""
        if self._active_idx >= self._num_questions:
            return
        qid = self._questions[self._active_idx]["id"]
        opts = self._questions[self._active_idx].get("options")
        try:
            if opts:
                ol = self.query_one(f"#{self._option_id_prefix}{qid}-list", OptionList)
                ol.focus()
            else:
                inp = self.query_one(f"#{self._option_id_prefix}{qid}-input", Input)
                inp.focus()
        except Exception:
            pass

    # ─── Navigation & Actions ───

    def _navigate_to(self, idx: int):
        """Switch to a different question index."""
        if 0 <= idx < self._num_questions:
            self._active_idx = idx
            self._rebuild()

    def _confirm_current(self):
        """Confirm/save the answer for the currently active question."""
        if self._active_idx >= self._num_questions:
            return
        q = self._questions[self._active_idx]
        qid = q["id"]
        opts = q.get("options")

        if opts:
            try:
                ol = self.query_one(f"#{self._option_id_prefix}{qid}-list", OptionList)
                highlighted = ol.highlighted
                if highlighted is not None:
                    opt_obj = ol.get_option_at_index(highlighted)
                    if opt_obj:
                        self._answers[qid] = str(opt_obj.prompt)
                        self._try_advance_or_submit()
            except Exception:
                pass
        else:
            try:
                inp = self.query_one(f"#{self._option_id_prefix}{qid}-input", Input)
                text = inp.value.strip()
                if not text:
                    text = q.get("default", "")
                self._answers[qid] = text  # 允许空字符串（可选问题）
                self._try_advance_or_submit()
            except Exception:
                pass

    def _try_advance_or_submit(self):
        """After answering, advance to next unanswered or submit if all done."""
        # Try to find next unanswered
        for i in range(self._active_idx + 1, self._num_questions):
            if self._questions[i]["id"] not in self._answers:
                self._navigate_to(i)
                return
        # No more unanswered after current → submit what we have
        self.post_message(self.Submitted(dict(self._answers)))

    # ─── Event Handlers ───

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        oid = event.option_id
        if not oid.startswith(self._option_id_prefix):
            return
        rest = oid[len(self._option_id_prefix):]
        for q in self._questions:
            qid = q["id"]
            if rest.startswith(f"{qid}-opt"):
                try:
                    opt_obj = event.option_list.get_option(oid)
                    if opt_obj:
                        self._answers[qid] = str(opt_obj.prompt)
                        self._try_advance_or_submit()
                except Exception:
                    pass
                break

    def on_input_submitted(self, event: Input.Submitted) -> None:
        inp_id = event.input.id
        if not inp_id.startswith(self._option_id_prefix):
            return
        text = event.value.strip()
        rest = inp_id[len(self._option_id_prefix):]
        for q in self._questions:
            qid = q["id"]
            if rest == f"{qid}-input":
                # 空输入时使用 default 值，无 default 则标记为空（可选问题）
                if not text:
                    text = q.get("default", "")
                self._answers[qid] = text  # 允许空字符串表示跳过
                self._try_advance_or_submit()
                break

    def on_key(self, event) -> None:
        if event.key == "left":
            if self._active_idx > 0:
                self._navigate_to(self._active_idx - 1)
            event.prevent_default()
        elif event.key == "right":
            if self._active_idx < self._num_questions - 1:
                self._navigate_to(self._active_idx + 1)
            event.prevent_default()
        elif event.key in ("up", "down"):
            # Circular navigation for active question's OptionList
            try:
                qid = self._questions[self._active_idx]["id"]
                ol = self.query_one(f"#{self._option_id_prefix}{qid}-list", OptionList)
                total = ol.option_count
                if total == 0:
                    return
                cur = ol.highlighted
                if event.key == "up":
                    if cur is None or cur == 0:
                        ol.highlight(total - 1)
                    else:
                        ol.highlight(cur - 1)
                else:
                    if cur is None or cur >= total - 1:
                        ol.highlight(0)
                    else:
                        ol.highlight(cur + 1)
                event.prevent_default()
            except Exception:
                pass
        elif event.key == "escape":
            self.post_message(self.Dismissed())
            event.prevent_default()





class ChatInput(Input):
    """带命令补全的输入框"""

    def __init__(self, **kwargs):
        super().__init__(
            placeholder="Type a message or command...",
            id="chat-input",
            **kwargs,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 主应用
# ══════════════════════════════════════════════════════════════════════════════

class AgentTUI(App):
    """My Agent 2.0 — Textual TUI"""

    TITLE = "My Agent 2.0"

    COMMANDS = [
        "/sessions", "/ls", "/new",
        "/rename", "/delete", "/models", "/mcp", "/context",
        "/compact", "/quit", "/q",
    ]

    CSS = """
    /* 全局 */
    Screen {
        layout: vertical;
    }

    /* Header */
    Header {
        background: $primary;
        border-left: tall $primary;
        border-right: tall $primary;
    }

    /* 主布局：sidebar + content 水平排列 */
    #main-layout {
        height: 1fr;
        border-left: tall $primary;
        border-right: tall $primary;
    }

    #content-area {
        height: 1fr;
        layout: vertical;
    }

    /* 左侧信息栏 */
    .info-sidebar {
        width: 24;
        height: 1fr;
        background: $surface-darken-1;
        border-right: tall $primary;
        padding: 1;
        display: none;  /* 默认隐藏，宽屏时 on_mount/on_resize 切换 */
    }
    .sidebar-header {
        padding: 0 0 1 0;
        text-align: center;
    }
    .sb-row {
        height: 1;
        padding: 0;
    }
    .sb-label {
        width: auto;
        height: 1;
        color: $text-disabled;
    }
    .sb-value {
        width: 1fr;
        height: 1;
        text-align: center;
    }

    /* 消息区 */
    #message-area {
        height: 1fr;
        scrollbar-size: 1 1;
        padding: 0 1;
    }

    /* 消息样式 */
    .msg-user {
        background: $surface-lighten-1;
        color: $text;
        padding: 0 2;
        margin-top: 1;
        margin-bottom: 1;
    }
    .msg-agent {
        padding: 0 0 0 2;
    }
    .msg-system {
        color: $warning;
        padding: 0 0 0 2;
        margin-top: 1;
    }
    .msg-tool {
        color: $text-muted;
        padding: 0 0 0 4;
    }
    .msg-error {
        color: $error;
        padding: 0 0 0 2;
    }
    .msg-success {
        color: $success;
        padding: 0 0 0 2;
    }
    .msg-info {
        color: $text-muted;
        padding: 0 0 0 2;
    }
    .msg-thinking {
        padding: 0 0 0 2;
        margin-top: 1;
    }

    /* Spinner */
    .spinner {
        padding: 0 0 0 2;
        height: auto;
        min-height: 1;  /* 单行(Thinking)，有 info 时自动扩展到两行 */
    }

    /* 输入框区域 */
    #input-area {
        height: auto;
        dock: bottom;
    }

    /* 输入框 */
    #chat-input {
        height: 3;
        padding: 0 1;
        margin: 1 0 0 0;
        border: tall $primary;
    }

    /* 状态栏 — 响应式：宽屏 1 行，窄屏 2 行 */
    .status-bar {
        height: auto;
        max-height: 2;
        padding: 0 1;
        color: $text;
        background: $primary-darken-2;
        dock: bottom;
        border-left: tall $primary;
        border-right: tall $primary;
    }

    /* HITL panel — 输入框上方 */
    #hitl-panel {
        height: auto;
        padding: 0 1;
        dock: top;
    }
    .interactive-selector {
        height: auto;
        max-height: 15;
        background: $surface;
        padding: 0 1;
        border-top: tall $accent;
        border-bottom: none;
    }
    .selector-title {
        padding: 0 0 0 0;
    }
    .selector-input {
        margin-top: 0;
        border: tall $accent;
        background: $surface-darken-1;
    }

    /* MultiQuestionSurvey */
    .survey-widget {
        height: auto;
        max-height: 20;
        background: $surface;
        padding: 0 1;
        border-top: tall $accent;
        border-bottom: none;
    }
    .survey-q-active {
        padding: 0 0 1 0;
    }
    .survey-q-done {
        padding: 0 0 0 0;
        color: $text-muted;
    }
    .survey-q-pending {
        padding: 0 0 0 0;
        color: $text-muted;
    }
    .survey-input {
        margin-top: 1;
        border: tall $accent;
        background: $surface-darken-1;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "退出", priority=True),
        # Ctrl+C 不绑定退出，由 on_key 处理双击退出
        # up/down 不全局绑定 — 有 selector/survey 时留给 OptionList，否则走 on_key 历史导航
    ]

    # ── 外部注入的回调 ──
    # App 不直接依赖 chat_agent.py，而是通过回调解耦
    on_user_input = None  # async def (text: str) -> None
    on_model_switch = None  # async def (name: str) -> None
    on_command = None  # async def (cmd: str) -> bool  (返回 True = 退出)
    get_status_info = None  # () -> dict  (获取当前状态)

    # Generic InteractiveSelector callbacks
    on_selector_choice = None       # def (value: str) → user selected an option
    on_selector_cancelled = None    # def () → user pressed Esc
    on_selector_custom_input = None # def (text: str) → user typed custom input

    # MultiQuestionSurvey callbacks
    on_survey_submitted = None      # def (results: dict) → {q_id: answer}
    on_survey_cancelled = None       # def () → user dismissed survey

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._spinner: SpinnerWidget | None = None
        self._thinking_widget: ThinkingWidget | None = None
        self._md_stream = None  # Markdown 流式写入器
        self._current_agent_md: Markdown | None = None
        self._history: list[str] = []
        self._history_idx = -1
        self._tab_completions: list[str] = []
        self._tab_idx = -1
        self._last_ctrl_c_time: float = 0.0  # 双击 Ctrl+C 退出
        self._current_selector_ctx = None     # HITL selector context

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-layout"):
            yield InfoSidebar()
            with Vertical(id="content-area"):
                yield VerticalScroll(id="message-area")
                with Vertical(id="input-area"):
                    yield Vertical(id="hitl-panel")
                    yield ChatInput()
        yield StatusBar()

    def on_mount(self) -> None:
        self.query_one("#chat-input", Input).focus()
        self._apply_layout()

    def on_resize(self, event) -> None:
        """Responsive: toggle sidebar vs bottom status bar based on terminal width."""
        self._apply_layout()

    SIDEBAR_MIN_WIDTH = 100  # 终端宽度 ≥ 100 列时显示侧边栏

    def _apply_layout(self):
        """Show sidebar on wide terminals, bottom bar always visible."""
        try:
            w = self.size.width
            sidebar = self.query_one("#info-sidebar", InfoSidebar)
            bottom_bar = self.query_one(StatusBar)
            if w >= self.SIDEBAR_MIN_WIDTH:
                sidebar.display = True
                bottom_bar._show_content = False
            else:
                sidebar.display = False
                bottom_bar._show_content = True
            bottom_bar.display = True
            bottom_bar.refresh_layout()
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════════
    # 消息渲染 API（供 chat_agent.py 调用）
    # ═══════════════════════════════════════════════════════════════════════

    def add_user_message(self, text: str):
        """添加用户消息（浅灰色背景，> 前缀）"""
        area = self.query_one("#message-area", VerticalScroll)
        msg = Static(f"[bold]>[/bold] {text}", classes="msg-user")
        area.mount(msg)
        self._scroll_to_bottom(force=True)

    def add_agent_label(self, sub_label: str | None = None):
        """Agent 标签 — 不再显示文字标签"""
        pass

    def start_spinner(self, label: str = "Thinking", info: str = ""):
        """启动 Spinner"""
        self._stop_spinner()
        area = self.query_one("#message-area", VerticalScroll)
        self._spinner = SpinnerWidget()
        area.mount(self._spinner)
        self._spinner.start(label, info)
        self._scroll_to_bottom(force=True)

    def stop_spinner(self, success: bool = True):
        """停止 Spinner（保留停止状态在 DOM 中作为痕迹）"""
        if self._spinner:
            self._spinner.stop(success)
            self._spinner = None

    def _stop_spinner(self):
        """内部清理旧 Spinner（从 DOM 移除，用于 start_spinner 前清理）"""
        if self._spinner:
            self._spinner.stop()
            try:
                self._spinner.remove()
            except Exception:
                pass
            self._spinner = None

    def update_spinner_info(self, info: str):
        """更新 Spinner 信息"""
        if self._spinner:
            self._spinner.update_info(info)

    def add_system_message(self, text: str, msg_type: str = "info"):
        """添加系统消息"""
        area = self.query_one("#message-area", VerticalScroll)
        msg = ChatMessage(msg_type, text)
        area.mount(msg)
        self._scroll_to_bottom(force=True)

    def start_thinking(self):
        """开始思考区域（累积灰色文字）"""
        area = self.query_one("#message-area", VerticalScroll)
        self._thinking_widget = ThinkingWidget()
        area.mount(self._thinking_widget)
        self._scroll_to_bottom(force=True)

    def append_thinking(self, text: str):
        """追加思考内容（同步，立即显示）"""
        if self._thinking_widget:
            self._thinking_widget.append(text)
            self._scroll_to_bottom()

    async def stream_thinking(self, text: str):
        """追加思考内容（异步打字机效果，逐字符输出）"""
        if not self._thinking_widget:
            return
        _DELAY = 0.015
        _SCROLL_EVERY = 4
        for i, c in enumerate(text):
            self._thinking_widget.append_char(c)
            if (i + 1) % _SCROLL_EVERY == 0:
                self._scroll_to_bottom()
            await asyncio.sleep(_DELAY * (0.5 + __import__('random').random()))
        self._scroll_to_bottom()

    def start_agent_reply(self):
        """开始 Agent 回复（Markdown 流式）"""
        area = self.query_one("#message-area", VerticalScroll)
        md_widget = Markdown("", classes="msg-agent")
        area.mount(md_widget)
        self._current_agent_md = md_widget
        self._md_stream = Markdown.get_stream(md_widget)
        self._scroll_to_bottom(force=True)

    async def stream_text(self, text: str):
        """流式写入文本到当前 Markdown（打字机效果，对标终端版 type_text）"""
        if self._md_stream:
            # 对标 chat_agent.py 的 type_text：逐字符 + 随机抖动
            _DELAY = 0.025  # 基础延迟（与终端版一致）
            _SCROLL_EVERY = 4  # 每 N 个字符滚动一次（平衡流畅度 vs 性能）
            for i, c in enumerate(text):
                await self._md_stream.write(c)
                if (i + 1) % _SCROLL_EVERY == 0:
                    self._scroll_to_bottom()
                await asyncio.sleep(_DELAY * (0.5 + __import__('random').random()))
            self._scroll_to_bottom(force=True)  # 回复完毕，确保滚到底
        else:
            import sys
            print(f"[WARN] stream_text called but _md_stream is None (len={len(text)})", file=sys.stderr)

    async def finish_stream(self):
        """结束流式写入"""
        if self._md_stream:
            await self._md_stream.stop()
            self._md_stream = None

    def add_tool_call(self, name: str, params: str = ""):
        """显示工具调用"""
        info = f"{name}（{params}）" if params else name
        self.update_spinner_info(info)

    def show_selector(self, selector: InteractiveSelector):
        """Mount a generic interactive selector above the input box."""
        panel = self.query_one("#hitl-panel")
        # 清理旧面板内容
        for child in list(panel.children):
            child.remove()
        panel.mount(selector)
        # Focus handled by InteractiveSelector.on_mount

    def remove_selector(self):
        """Remove any mounted selector from the hitl panel."""
        panel = self.query_one("#hitl-panel")
        for child in list(panel.children):
            child.remove()

    def show_survey(self, survey):
        """Mount a MultiQuestionSurvey above the input box."""
        panel = self.query_one("#hitl-panel")
        for child in list(panel.children):
            child.remove()
        panel.mount(survey)
        survey._refocus_active()

    def remove_survey(self):
        """Remove any mounted survey from the hitl panel."""
        panel = self.query_one("#hitl-panel")
        for child in list(panel.children):
            child.remove()

    # ─── Model selector (modal dialog) ───

    def show_model_selector(self, models: list[tuple[str, dict]], active: str):
        """Show model selector as a modal dialog."""
        options = []
        for name, cfg in models:
            mid = cfg.get("id", "?")
            base = cfg.get("base", "").replace("https://", "").replace("http://", "").split("/")[0]
            is_active = name == active
            prefix = "→ " if is_active else "  "
            prompt = Text.from_markup(
                f"[{'green' if is_active else 'cyan'}]{prefix}{name}[/] — {mid}\n     [dim]{base}[/dim]"
            )
            options.append({"id": name, "label": prompt, "disabled": is_active})

        modal = SelectorModal(
            title="Select Model",
            options=options,
            option_id_prefix="model-",
        )

        def _on_result(result: str | None):
            if result is not None:
                self._do_model_switch(result)

        self.push_screen(modal, _on_result)

    def update_status(self, **kwargs):
        """Update both sidebar and bottom status bar."""
        self.query_one(StatusBar).update_status(**kwargs)
        self.query_one("#info-sidebar", InfoSidebar).update_status(**kwargs)

    # ═══════════════════════════════════════════════════════════════════════
    # 用户输入处理
    # ═══════════════════════════════════════════════════════════════════════

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """用户按 Enter 提交输入（仅处理主聊天框）"""
        # 只处理主聊天输入框，忽略 selector/survey 等组件的 Input 冒泡
        if event.input.id != "chat-input":
            return
        text = event.value.strip()
        if not text:
            return

        # 清空输入框
        input_widget = event.input
        input_widget.value = ""

        # 加入历史
        self._history.append(text)
        self._history_idx = -1

        # 处理输入（异步）
        self._handle_input(text)

    @work(exclusive=True)
    async def _handle_input(self, text: str):
        """处理用户输入（在后台 worker 中执行）"""
        try:
            if text.startswith("/"):
                # 命令
                if self.on_command:
                    should_exit = await self.on_command(text)
                    if should_exit:
                        self.exit()
            else:
                # 普通消息
                self.add_user_message(text)
                if self.on_user_input:
                    await self.on_user_input(text)
        except Exception as e:
            self.add_system_message(f"错误: {e}", "error")

    # ── Tab 补全 ──

    def on_key(self, event) -> None:
        """处理键盘事件"""
        import time as _time
        if event.key == "ctrl+c":
            now = _time.time()
            if now - self._last_ctrl_c_time < 1.5:
                # 双击 Ctrl+C → 退出
                self.exit()
            else:
                self._last_ctrl_c_time = now
            event.prevent_default()
        elif event.key == "escape":
            pass  # no action
        elif event.key == "tab":
            self._handle_tab()
            event.prevent_default()
        elif event.key in ("up", "down"):
            # 有活跃的 selector / survey 时，上下键留给 OptionList 控制选项
            has_selector = False
            try:
                self.query_one(".interactive-selector")
                has_selector = True
            except Exception:
                pass
            if not has_selector:
                try:
                    self.query_one(".survey-widget")
                    has_selector = True
                except Exception:
                    pass
            if has_selector:
                return  # 不拦截，让 OptionList 处理

            # 上下键控制历史导航（仅在输入框聚焦时）
            focused = self.focused
            if focused and focused.id == "chat-input":
                if event.key == "up":
                    self._history_up()
                else:
                    self._history_down()
                event.prevent_default()

    def _handle_tab(self):
        """Tab 命令补全"""
        input_widget = self.query_one("#chat-input", Input)
        text = input_widget.value

        if text.startswith("/"):
            # 命令补全
            if self._tab_idx == -1:
                # 首次 Tab，收集匹配项
                self._tab_completions = [
                    cmd for cmd in self.COMMANDS
                    if cmd.startswith(text)
                ]
                self._tab_idx = 0
            elif self._tab_completions:
                self._tab_idx = (self._tab_idx + 1) % len(self._tab_completions)

            if self._tab_completions:
                input_widget.value = self._tab_completions[self._tab_idx]
                # 光标移到末尾
                input_widget.cursor_position = len(input_widget.value)
        else:
            # 非命令，重置
            self._tab_completions = []
            self._tab_idx = -1

    def _history_up(self):
        """上翻历史"""
        if not self._history:
            return
        if self._history_idx == -1:
            self._history_idx = len(self._history) - 1
        elif self._history_idx > 0:
            self._history_idx -= 1
        input_widget = self.query_one("#chat-input", Input)
        input_widget.value = self._history[self._history_idx]

    def _history_down(self):
        """下翻历史"""
        if self._history_idx == -1:
            return
        if self._history_idx < len(self._history) - 1:
            self._history_idx += 1
            input_widget = self.query_one("#chat-input", Input)
            input_widget.value = self._history[self._history_idx]
        else:
            self._history_idx = -1
            input_widget = self.query_one("#chat-input", Input)
            input_widget.value = ""

    # ── Selector callbacks (generic) ──

    def on_interactive_selector_selected(self, event: InteractiveSelector.Selected) -> None:
        """User selected an option from any InteractiveSelector (HITL etc.)."""
        self.remove_selector()
        if self.on_selector_choice:
            self.on_selector_choice(event.value)

    def on_interactive_selector_dismissed(self, event: InteractiveSelector.Dismissed) -> None:
        """User pressed Esc to dismiss selector."""
        self.remove_selector()
        if self.on_selector_cancelled:
            self.on_selector_cancelled()

    def on_interactive_selector_user_input(self, event: InteractiveSelector.UserInput) -> None:
        """User typed custom input in a selector."""
        self.remove_selector()
        if self.on_selector_custom_input:
            self.on_selector_custom_input(event.text)

    # ── Survey callbacks ──

    def on_multi_question_survey_submitted(self, event: MultiQuestionSurvey.Submitted) -> None:
        """User submitted the multi-question survey."""
        self.remove_survey()
        if self.on_survey_submitted:
            self.on_survey_submitted(event.results)

    def on_multi_question_survey_dismissed(self, event: MultiQuestionSurvey.Dismissed) -> None:
        """User dismissed the survey."""
        self.remove_survey()
        if self.on_survey_cancelled:
            self.on_survey_cancelled()

    # ── Backward compat: model-specific callback ──

    @work(exclusive=True)
    async def _do_model_switch(self, name: str):
        if self.on_model_switch:
            await self.on_model_switch(name)

    # ── 辅助 ──

    def _scroll_to_bottom(self, force: bool = False):
        """自动滚到底部（用户手动上翻时不会强拉，除非 force=True）"""
        area = self.query_one("#message-area", VerticalScroll)
        if not force:
            # 如果用户不在底部附近（距离 > 3 行），不打扰
            max_scroll = area.max_scroll_y
            current = area.scroll_y
            if max_scroll > 0 and (max_scroll - current) > 3:
                return
        area.scroll_end(animate=False)

    def _stop_spinner(self):
        if self._spinner:
            self._spinner.stop()
            self._spinner = None


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.1f}s"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    return f"{m}m{s}s"


# ══════════════════════════════════════════════════════════════════════════════
# 独立测试入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random

    class TestApp(AgentTUI):
        """测试用 App"""

        AUTO_REPLIES = [
            "好的，收到！让我看看。\n\n```python\nimport os\nprint(os.getcwd())\n```",
            "这个很有意思！🤔\n\n| 项目 | 状态 |\n|------|------|\n| AI | ✅ |\n| TUI | ✅ |",
            "**分析结果**：\n\n1. 第一点是...\n2. 第二点是...\n\n> 引用一段话\n\n```bash\nls -la\n```",
        ]

        def on_mount(self):
            super().on_mount()
            self.update_status(model="MiniMax M2.7", tokens=1234, ctx_pct=32, session="default")
            self.add_system_message("欢迎！输入消息测试聊天 · Tab 补全命令 · /models 测试选择器", "info")

        async def on_user_input(self, text: str):
            self.add_agent_label("coder")
            self.start_spinner("Thinking")
            await asyncio.sleep(1.5)

            # 模拟工具调用
            self.update_spinner_info("list_dir（path=\".\"）")
            await asyncio.sleep(0.8)

            self.stop_spinner()

            # 流式回复
            self.start_agent_reply()
            reply = random.choice(self.AUTO_REPLIES)
            # 模拟逐字输出
            chunk_size = 3
            for i in range(0, len(reply), chunk_size):
                await self.stream_text(reply[i:i + chunk_size])
                await asyncio.sleep(0.02)
            await self.finish_stream()

            self.update_status(tokens=random.randint(2000, 5000), ctx_pct=random.randint(20, 70))

        async def on_command(self, cmd: str) -> bool:
            if cmd in ("/quit", "/q"):
                self.add_system_message("再见！👋", "info")
                return True
            elif cmd == "/models":
                models = [
                    ("minimax", {"id": "minimax-M2.7", "base": "https://api.minimax.chat"}),
                    ("deepseek", {"id": "deepseek-chat", "base": "https://api.deepseek.com"}),
                    ("qwen", {"id": "qwen-max", "base": "https://dashscope.aliyuncs.com"}),
                ]
                self.show_model_selector(models, "minimax")
                return False
            else:
                self.add_system_message(f"未知命令: {cmd}", "warning")
                return False

        async def on_model_switch(self, name: str):
            self.add_system_message(f"模型已切换到: {name}", "success")
            self.update_status(model=name)

    app = TestApp()
    app.run()
