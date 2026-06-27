"""
src/errors.py

Custom exception hierarchy for the Kids Storytelling Agent.

Every failure domain has its own exception class so that
error_handler.py can catch and route them precisely — no
bare `except Exception` that swallows unknown failures silently.

Hierarchy:
    StoryAgentError                 — base for all project exceptions
    ├── AgentError                  — LLM agent call failures
    │   ├── StoryArchitectError
    │   ├── WriterError
    │   ├── ValidatorError
    │   └── AnswerValidatorError
    ├── ToolError                   — external tool failures
    │   ├── TTSError
    │   ├── STTError
    │   └── WebFetchError
    ├── ModelArmorError             — Model Armor specific failures
    │   ├── ModelArmorAPIError      — API unavailable / timeout
    │   └── ModelArmorMatchError    — injection / match detected
    ├── SandboxError                — Docker sandbox failures
    │   ├── SandboxStartError       — container failed to start
    │   ├── SandboxTimeoutError     — execution exceeded time limit
    │   └── SandboxOutputError      — bad / unparseable output
    ├── MemoryError                 — SQLite read / write failures
    └── GraphError                  — LangGraph orchestration failures
"""


# ── Base ──────────────────────────────────────────────────────────────────────

class StoryAgentError(Exception):
    """
    Base exception for all Kids Storytelling Agent errors.
    Carries an optional session_id so log entries can be correlated.
    """

    def __init__(self, message: str, session_id: str = None, **context):
        super().__init__(message)
        self.session_id = session_id
        self.context    = context  # arbitrary extra fields logged alongside

    def __repr__(self) -> str:
        ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.__class__.__name__}({self.args[0]!r}, {ctx})"


# ── Agent errors ──────────────────────────────────────────────────────────────

class AgentError(StoryAgentError):
    """Raised when a Gemini-backed agent call fails."""


class StoryArchitectError(AgentError):
    """Story Architect failed to design story arc or pick topic."""


class WriterError(AgentError):
    """Writer agent failed to generate story text."""


class ValidatorError(AgentError):
    """LLM Judge failed to score or parse validation output."""


class AnswerValidatorError(AgentError):
    """Answer Validator failed to judge child response or generate hint."""


# ── Tool errors ───────────────────────────────────────────────────────────────

class ToolError(StoryAgentError):
    """Raised when an external tool (TTS, STT, web) fails."""


class TTSError(ToolError):
    """
    Microsoft Edge TTS (edge-tts) failed.
    Caller sets narration_failed=True and session continues without audio.
    """


class STTError(ToolError):
    """
    Whisper STT failed to transcribe.
    error_handler asks child to repeat on this exception.
    """


class WebFetchError(ToolError):
    """
    Wikipedia API call failed.
    error_handler skips external facts on this exception.
    """


# ── Model Armor errors ────────────────────────────────────────────────────────

class ModelArmorError(StoryAgentError):
    """Base for Model Armor failures."""


class ModelArmorAPIError(ModelArmorError):
    """
    Model Armor API is unavailable, timed out, or returned an error.
    error_handler retries once then passes raw content through.
    """


class ModelArmorMatchError(ModelArmorError):
    """
    Model Armor detected a prompt injection or policy violation.
    error_handler blocks the content — Story Architect uses Gemini's own knowledge.
    """


# ── Sandbox errors ────────────────────────────────────────────────────────────

class SandboxError(StoryAgentError):
    """Base for Docker sandbox failures."""


class SandboxStartError(SandboxError):
    """
    Docker container failed to start.
    error_handler retries once then skips pronunciation scoring.
    """


class SandboxTimeoutError(SandboxError):
    """
    Sandbox execution exceeded the time limit.
    error_handler skips pronunciation scoring, uses text-only judgment.
    """


class SandboxOutputError(SandboxError):
    """
    Sandbox returned unparseable or malformed output.
    error_handler skips pronunciation score, treats as unavailable.
    """


# ── Memory errors ─────────────────────────────────────────────────────────────

class StoryMemoryError(StoryAgentError):
    """
    SQLite read or write failure.
    Named StoryMemoryError to avoid collision with Python built-in MemoryError.
    """


# ── Graph errors ──────────────────────────────────────────────────────────────

class GraphError(StoryAgentError):
    """
    LangGraph orchestration failure — node exception not caught by
    any agent-level handler, or graph state corruption.
    Caught by the top-level safety net in main.py.
    """
