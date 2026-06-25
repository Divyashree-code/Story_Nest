"""
src/logger.py

Structured JSON logger for the Kids Storytelling Agent.

Every log entry is a JSON object so it is machine-readable,
searchable, and can be piped into any log aggregator later.

Fields in every entry:
    timestamp   — ISO-8601 UTC
    level       — DEBUG / INFO / WARNING / ERROR
    agent       — which agent or module emitted the log
    event       — short snake_case description of what happened
    session_id  — ties all logs from one story session together
    + any extra kwargs passed at call time (duration_ms, attempts, etc.)

Usage:
    from src.logger import get_logger
    log = get_logger("writer")
    log.info("story_generated", session_id="abc123", word_count=342, duration_ms=1820)
    log.warning("rewrite_needed", session_id="abc123", attempt=2, issues=["vocab_too_complex"])
    log.error("gemini_call_failed", session_id="abc123", error="timeout")
"""

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


# ── ensure logs/ directory exists ────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"


# ── JSON formatter ────────────────────────────────────────────────────────────
class JSONFormatter(logging.Formatter):
    """
    Formats every log record as a single-line JSON object.
    Extra keyword arguments passed via log.info("event", **kwargs)
    are captured through the 'extra' dict and merged into the output.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "agent":     getattr(record, "agent", record.name),
            "event":     record.getMessage(),
            "session_id": getattr(record, "session_id", None),
        }

        # Merge any extra fields passed at call time
        skip = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "agent", "session_id",
        }
        for key, value in record.__dict__.items():
            if key not in skip:
                entry[key] = value

        # Include exception traceback if present
        if record.exc_info:
            entry["traceback"] = traceback.format_exception(*record.exc_info)

        return json.dumps(entry, default=str)


# ── console formatter (human-readable) ───────────────────────────────────────
class ConsoleFormatter(logging.Formatter):
    """
    Human-readable console output.
    Keeps the agent name and event clear for fast debugging.
    """
    COLOURS = {
        "DEBUG":   "\033[36m",   # cyan
        "INFO":    "\033[32m",   # green
        "WARNING": "\033[33m",   # yellow
        "ERROR":   "\033[31m",   # red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour  = self.COLOURS.get(record.levelname, "")
        agent   = getattr(record, "agent", record.name)
        session = getattr(record, "session_id", "")
        sid     = f" [{session[:8]}]" if session else ""
        return (
            f"{colour}{record.levelname:<8}{self.RESET} "
            f"{agent:<20} {record.getMessage()}{sid}"
        )


# ── root logger setup ─────────────────────────────────────────────────────────
def _setup_root_logger() -> None:
    root = logging.getLogger("storynest")
    if root.handlers:
        return  # already configured — don't add duplicate handlers

    root.setLevel(logging.DEBUG)

    # File handler — JSON, all levels
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JSONFormatter())

    # Console handler — human-readable, INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(ConsoleFormatter())

    root.addHandler(fh)
    root.addHandler(ch)


_setup_root_logger()


# ── LoggerAdapter adds agent name + session_id to every record ───────────────
class AgentLogger:
    """
    Thin wrapper that binds an agent name to every log call
    and accepts keyword arguments as structured fields.

    Usage:
        log = get_logger("writer")
        log.info("story_generated", session_id="abc", word_count=342)
    """

    def __init__(self, agent_name: str) -> None:
        self._logger = logging.getLogger(f"storynest.{agent_name}")
        self._agent  = agent_name

    def _log(self, level: int, event: str, **kwargs) -> None:
        extra = {"agent": self._agent, **kwargs}
        # session_id goes into extra so JSONFormatter picks it up cleanly
        self._logger.log(level, event, extra=extra)

    def debug(self, event: str, **kwargs) -> None:
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, exc_info: bool = False, **kwargs) -> None:
        extra = {"agent": self._agent, **kwargs}
        self._logger.log(logging.ERROR, event, extra=extra, exc_info=exc_info)


def get_logger(agent_name: str) -> AgentLogger:
    """
    Returns a structured logger bound to the given agent name.
    Call once per module at import time.

    Args:
        agent_name: short identifier e.g. "writer", "validator", "sandbox"

    Returns:
        AgentLogger instance
    """
    return AgentLogger(agent_name)
