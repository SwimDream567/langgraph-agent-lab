"""Human-in-the-Loop (HITL) Tool — allows AI agents to ask users questions.

When the LLM calls this tool, it presents options to the user via an
InteractiveSelector / MultiQuestionSurvey in the TUI. The agent blocks
until the user responds.

Supports two modes:
1. **Single-question** (ask_user/confirm): InteractiveSelector popup
2. **Multi-question** (ask_questions): Survey widget with ←→ navigation

Integration:
    1. Import and register as a LangChain @tool
    2. Set callbacks via get_hitl_bridge() before running the graph
    3. The tool uses asyncio.Future + run_coroutine_threadsafe to sync with TUI events
"""

import asyncio
from typing import Optional
from langchain_core.tools import tool


# ─── Global HITL Bridge ───
# Must be configured by the host application (TUI) before graph execution.

class HitlBridge:
    """Bridge between HITL tool execution and TUI event loop."""

    def __init__(self):
        self._future: Optional[asyncio.Future] = None
        self._pending_request: Optional[dict] = None
        self._on_show_selector: Optional[callable] = None   # single-question callback
        self._on_show_survey: Optional[callable] = None      # multi-question callback
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_callbacks(self, on_show_selector: callable):
        """Set callback for single-question InteractiveSelector."""
        self._on_show_selector = on_show_selector

    def set_survey_callback(self, on_show_survey: callable):
        """Set callback for MultiQuestionSurvey widget.

        Args:
            on_show_survey: callable(request: dict) -> None
                request = {"title": str, "questions": [...]}
                Each question: {"id": str, "question": str, "options": [str]|None}
        """
        self._on_show_survey = on_show_survey

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the asyncio event loop (call from TUI mount)."""
        self._loop = loop

    # ── Single-question API (ask_user / confirm) ──

    def ask(self, question: str, options: list[str],
             allow_input: bool = True,
             placeholder: str = "Type your choice...") -> str:
        """Blocking call: show selector, wait for user response."""
        if self._loop is None:
            return "[HITL Error] No event loop configured"

        if self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self._async_ask(question, options, allow_input, placeholder),
                self._loop,
            )
            return future.result()  # no timeout – human-in-the-loop waits indefinitely
        else:
            return self._loop.run_until_complete(
                self._async_ask(question, options, allow_input, placeholder)
            )

    async def _async_ask(self, question: str, options: list[str],
                         allow_input: bool, placeholder: str) -> str:
        """Async version — creates Future, shows selector, awaits result."""
        self._future = self._loop.create_future()
        self._pending_request = {
            "type": "selector",
            "question": question,
            "options": options,
            "allow_input": allow_input,
            "placeholder": placeholder,
        }
        if self._on_show_selector:
            self._on_show_selector(self._pending_request)
        result = await self._future  # no timeout – waits for user action
        return result

    # ── Multi-question API (ask_questions) ──

    def survey(self, title: str, questions: list[dict]) -> dict[str, str]:
        """Blocking call: show multi-question survey, wait for all answers.

        Args:
            title: Survey title
            questions: List of {"id", "question", "options"?}

        Returns:
            {question_id: answer_text} mapping
        """
        if self._loop is None:
            return {"_error": "[HITL Error] No event loop configured"}

        if self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self._async_survey(title, questions),
                self._loop,
            )
            return future.result()  # no timeout – human-in-the-loop waits indefinitely
        else:
            return self._loop.run_until_complete(
                self._async_survey(title, questions)
            )

    async def _async_survey(self, title: str, questions: list[dict]) -> dict[str, str]:
        """Async version — shows survey, awaits all results."""
        self._future = self._loop.create_future()
        self._pending_request = {
            "type": "survey",
            "title": title,
            "questions": questions,
        }
        if self._on_show_survey:
            self._on_show_survey(self._pending_request)
        result = await self._future  # no timeout – waits for user action
        return result

    # ── Resolution (called by TUI) ──

    def resolve(self, value: str):
        """Resolve pending Future with a string value (single-question)."""
        if self._future and not self._future.done():
            self._future.set_result(value)
        self._pending_request = None
        self._future = None

    def resolve_survey(self, results: dict):
        """Resolve pending Future with survey results dict."""
        if self._future and not self._future.done():
            self._future.set_result(results)
        self._pending_request = None
        self._future = None

    def cancel(self):
        """Cancel pending request (Esc pressed)."""
        req_type = (self._pending_request or {}).get("type", "selector")
        if self._future and not self._future.done():
            if req_type == "survey":
                self._future.set_result({})
            else:
                self._future.set_result("[CANCELLED]")
        self._pending_request = None
        self._future = None

    @property
    def is_pending(self) -> bool:
        """Whether there's a pending HITL request waiting for user input."""
        return self._future is not None and not self._future.done()


# Singleton bridge instance
_hitl_bridge = HitlBridge()


def get_hitl_bridge() -> HitlBridge:
    """Get the global HITL bridge instance."""
    return _hitl_bridge


# ════════════════════════════════════════════════
# Tool Definitions
# ════════════════════════════════════════════════

@tool
def ask_user(question: str, options: list[str] | None = None,
             allow_custom_input: bool = True) -> str:
    """Ask the user a question and present interactive options for them to select.

    IMPORTANT: Use this tool proactively whenever the user's request is ambiguous,
    has multiple possible approaches, or requires a decision. Do NOT guess — ask.

    Common triggers:
    - User request is vague or has multiple interpretations
    - You need to choose between technical approaches
    - User asks for recommendations and there are several options
    - You need clarification before proceeding

    Args:
        question: The question to ask the user (be specific and concise)
        options: List of option strings to present (2-6 recommended). If None or empty,
                 the user can only provide free-text input.
        allow_custom_input: Whether to allow typing custom input beyond the options
        allow_custom_input: Whether to allow typing custom input beyond the options

    Returns:
        The user's selected option or typed text.

    Examples:
        ask_user("Which approach should we take?",
                 ["Refactor existing code", "Write from scratch", "Research first"])
    """
    opts = options or []

    # Guard: no options and no custom input — meaningless call
    if not opts and not allow_custom_input:
        return "[Error] ask_user called with no options and allow_custom_input=False. Provide at least one option or set allow_custom_input=True."

    # No options but custom input allowed — provide a text-only input prompt
    if not opts and allow_custom_input:
        return _hitl_bridge.ask(
            question=question,
            options=["(Type your answer below)"],
            allow_input=True,
            placeholder="Type your answer...",
        )

    return _hitl_bridge.ask(
        question=question,
        options=opts,
        allow_input=allow_custom_input,
        placeholder="Type your choice...",
    )


@tool
def confirm(question: str) -> str:
    """Ask user a yes/no confirmation question.

    Args:
        question: The yes/no question to ask

    Returns:
        "yes" or "no" (or custom input)
    """
    return _hitl_bridge.ask(
        question=f"{question}\n[y/n]",
        options=["Yes", "No"],
        allow_input=False,
        placeholder="y or n",
    )


@tool
def ask_questions(title: str, questions: list[dict]) -> str:
    """Ask the user multiple questions at once in a structured survey.

    Use this when you need to clarify several things before proceeding.
    The user sees all questions, navigates between them with arrow keys,
    and answers each one. All answers are returned together when done.

    Args:
        title: Short title for the survey (e.g., "Quick confirmation")
        questions: List of question dicts. Each must have:
            - id: unique identifier for this question (e.g., "q1")
            - question: the question text
            - options (optional): list of predefined choices. If omitted,
              the user provides free-text input.
            - default (optional): default value for free-text questions.
              The user can press Enter to accept it, or type a different answer.

    Returns:
        JSON string of {question_id: answer_text} for each answered question.
        Example output: '{"q1": "Python", "q2": "FastAPI", "q3": "Use PostgreSQL"}'

    Examples:
        ask_questions(
            "Tech stack confirmation",
            [
                {"id": "lang", "question": "Which language?", "options": ["Python", "TypeScript"]},
                {"id": "framework", "question": "Which framework?", "default": "FastAPI"},
                {"id": "db", "question": "Which database?", "options": ["PostgreSQL", "MySQL", "SQLite"]}
            ]
        )
    """
    results = _hitl_bridge.survey(title=title, questions=questions)
    if "_error" in results:
        return results["_error"]
    # Return as readable JSON-like string for LLM context
    parts = [f'**{q["id"]}**: "{results.get(q["id"], "(skipped)")}"'
              for q in questions if q["id"] in results]
    return "\n".join(parts) if parts else "(no answers)"
