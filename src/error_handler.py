"""
src/error_handler.py

Centralised error handling for the Kids Storytelling Agent.

Every agent and tool calls these helpers instead of writing
inline try/except blocks. This keeps retry logic, fallback
decisions, and logging in one place.

Three public patterns:
    retry_once(fn, *args, delay=2.0, error_cls=StoryAgentError, **kwargs)
        Calls fn(*args, **kwargs). On failure waits `delay` seconds
        and tries once more. Raises error_cls if both attempts fail.

    with_fallback(fn, fallback_fn, *args, session_id=None, **kwargs)
        Calls fn(*args, **kwargs). On any StoryAgentError calls
        fallback_fn(*args, **kwargs) instead. Logs a warning.

    safe_run(fn, *args, default=None, session_id=None, **kwargs)
        Calls fn(*args, **kwargs). On failure logs the error and
        returns `default`. Never raises. Use for optional enrichments
        (facts, pronunciation score) where failure is acceptable.
"""

import time
import functools
from typing import Any, Callable, Optional, Type, TypeVar

from src.errors import (
    StoryAgentError,
    ModelArmorAPIError,
    ModelArmorMatchError,
    SandboxTimeoutError,
    WebFetchError,
    TTSError,
    STTError,
)
from src.logger import get_logger

log = get_logger("error_handler")

T = TypeVar("T")


# ── retry_once ────────────────────────────────────────────────────────────────

def retry_once(
    fn: Callable[..., T],
    *args,
    delay: float = 2.0,
    error_cls: Type[StoryAgentError] = StoryAgentError,
    session_id: str = None,
    **kwargs,
) -> T:
    """
    Calls fn(*args, **kwargs). On failure waits `delay` seconds and
    tries once more. Raises error_cls wrapping the original exception
    if both attempts fail.

    Only retries on recoverable errors — if the exception is a
    ModelArmorMatchError (injection detected) or SandboxTimeoutError
    (input too long), we do not retry because the same input will
    produce the same result.

    Args:
        fn:         The callable to attempt.
        *args:      Positional arguments forwarded to fn.
        delay:      Seconds to wait between attempts. Default 2.0.
        error_cls:  Exception class to raise on total failure.
        session_id: Attached to log entries for correlation.
        **kwargs:   Keyword arguments forwarded to fn.

    Returns:
        fn(*args, **kwargs) result on success.

    Raises:
        error_cls if both attempts fail.
    """
    _NON_RETRYABLE = (ModelArmorMatchError, SandboxTimeoutError)

    for attempt in range(1, 3):  # attempts 1 and 2
        try:
            result = fn(*args, **kwargs)
            if attempt == 2:
                log.info(
                    "retry_succeeded",
                    session_id=session_id,
                    fn=getattr(fn, '__name__', repr(fn)),
                    attempt=attempt,
                )
            return result

        except _NON_RETRYABLE as exc:
            # Non-retryable — log and re-raise immediately, no second attempt
            log.warning(
                "non_retryable_error",
                session_id=session_id,
                fn=getattr(fn, '__name__', repr(fn)),
                error=str(exc),
                reason="same input will fail again",
            )
            raise

        except Exception as exc:
            if attempt == 1:
                log.warning(
                    "attempt_failed_will_retry",
                    session_id=session_id,
                    fn=getattr(fn, '__name__', repr(fn)),
                    attempt=attempt,
                    delay_s=delay,
                    error=str(exc),
                )
                time.sleep(delay)
            else:
                # Both attempts failed
                log.error(
                    "both_attempts_failed",
                    session_id=session_id,
                    fn=getattr(fn, '__name__', repr(fn)),
                    error=str(exc),
                    exc_info=True,
                )
                raise error_cls(
                    f"{getattr(fn, '__name__', repr(fn))} failed after 2 attempts: {exc}",
                    session_id=session_id,
                    original_error=str(exc),
                ) from exc


# ── with_fallback ─────────────────────────────────────────────────────────────

def with_fallback(
    fn: Callable[..., T],
    fallback_fn: Callable[..., T],
    *args,
    session_id: str = None,
    **kwargs,
) -> T:
    """
    Calls fn(*args, **kwargs). On any StoryAgentError calls
    fallback_fn(*args, **kwargs) instead and logs a warning.

    If the fallback itself raises, that exception propagates normally
    (we do not silently swallow fallback failures).

    Designed for:
        - Kokoro TTS → pyttsx3
        - Whisper → re-ask child

    Args:
        fn:          Primary callable.
        fallback_fn: Fallback callable with the same signature.
        *args:       Forwarded to both callables.
        session_id:  Attached to log entries.
        **kwargs:    Forwarded to both callables.

    Returns:
        Result of fn or fallback_fn.
    """
    try:
        return fn(*args, **kwargs)
    except StoryAgentError as exc:
        log.warning(
            "primary_failed_using_fallback",
            session_id=session_id,
            primary=getattr(fn, '__name__', repr(fn)),
            fallback=getattr(fallback_fn, '__name__', repr(fallback_fn)),
            error=str(exc),
        )
        return fallback_fn(*args, **kwargs)


# ── safe_run ──────────────────────────────────────────────────────────────────

def safe_run(
    fn: Callable[..., T],
    *args,
    default: Any = None,
    session_id: str = None,
    **kwargs,
) -> Optional[T]:
    """
    Calls fn(*args, **kwargs). On any exception logs the error
    and returns `default`. Never raises.

    Use for optional enrichments where failure is acceptable:
        - Web facts fetch (story still works without them)
        - Pronunciation score (answer judged on text alone)
        - Model Armor (raw content passed through if API down)

    Args:
        fn:         The callable to attempt.
        *args:      Forwarded to fn.
        default:    Value returned on failure. Default None.
        session_id: Attached to log entries.
        **kwargs:   Forwarded to fn.

    Returns:
        fn(*args, **kwargs) result, or `default` on any failure.
    """
    try:
        return fn(*args, **kwargs)
    except ModelArmorMatchError as exc:
        # Injection detected — block content, return empty string not default
        log.warning(
            "model_armor_blocked_content",
            session_id=session_id,
            fn=getattr(fn, '__name__', repr(fn)),
            error=str(exc),
        )
        return ""  # empty string signals architect to skip external facts
    except Exception as exc:
        log.warning(
            "optional_step_failed_continuing",
            session_id=session_id,
            fn=getattr(fn, '__name__', repr(fn)),
            error=str(exc),
            default_used=repr(default),
        )
        return default


# ── timed ────────────────────────────────────────────────────────────────────

def timed(agent_name: str, event: str, session_id_key: str = "session_id"):
    """
    Decorator that logs duration_ms for any agent function.
    The decorated function must accept session_id as a kwarg
    OR it can be in **kwargs — we extract it for the log entry.

    Usage:
        @timed("writer", "story_generation")
        def writer_node(state: StoryState) -> StoryState:
            ...
    """
    agent_log = get_logger(agent_name)

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            session_id = kwargs.get(session_id_key) or (
                args[0].get(session_id_key) if args and isinstance(args[0], dict) else None
            )
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                duration_ms = round((time.perf_counter() - start) * 1000)
                agent_log.info(
                    event,
                    session_id=session_id,
                    duration_ms=duration_ms,
                )
                return result
            except Exception as exc:
                duration_ms = round((time.perf_counter() - start) * 1000)
                agent_log.error(
                    f"{event}_failed",
                    session_id=session_id,
                    duration_ms=duration_ms,
                    error=str(exc),
                    exc_info=True,
                )
                raise
        return wrapper
    return decorator
