"""
tests/test_error_handler.py

Tests for src/error_handler.py — retry, fallback, safe_run, timed.

Tests:
    - retry_once retries on recoverable errors
    - retry_once does NOT retry on non-retryable errors
    - retry_once raises wrapped error after two failures
    - retry_once succeeds on second attempt
    - with_fallback calls fallback on StoryAgentError
    - with_fallback propagates fallback exceptions
    - safe_run returns default on any exception
    - safe_run returns empty string on ModelArmorMatchError
    - safe_run returns result on success
    - timed decorator logs duration_ms
"""

import time
import pytest
import unittest.mock as mock

from src.errors import (
    StoryAgentError, ModelArmorMatchError, ModelArmorAPIError,
    SandboxTimeoutError, TTSError, STTError,
)
from src.error_handler import retry_once, with_fallback, safe_run, timed


# ── retry_once tests ──────────────────────────────────────────────────────────

class TestRetryOnce:
    def test_success_on_first_attempt(self):
        """Function succeeds on first call — no retry."""
        fn = mock.MagicMock(return_value="success")
        result = retry_once(fn, session_id="test")
        assert result == "success"
        assert fn.call_count == 1

    def test_success_on_second_attempt(self):
        """Function fails first, succeeds on second — retried once."""
        fn = mock.MagicMock(side_effect=[
            StoryAgentError("first fail"),
            "success",
        ])
        with mock.patch('src.error_handler.time.sleep'):
            result = retry_once(fn, session_id="test")
        assert result == "success"
        assert fn.call_count == 2

    def test_raises_after_two_failures(self):
        """Function fails twice — raises wrapped error."""
        fn = mock.MagicMock(side_effect=StoryAgentError("always fails"))
        with mock.patch('src.error_handler.time.sleep'):
            with pytest.raises(StoryAgentError):
                retry_once(fn, session_id="test")
        assert fn.call_count == 2

    def test_model_armor_match_not_retried(self):
        """ModelArmorMatchError is non-retryable — not retried."""
        fn = mock.MagicMock(side_effect=ModelArmorMatchError("injection"))
        with pytest.raises(ModelArmorMatchError):
            retry_once(fn, session_id="test")
        assert fn.call_count == 1   # called once only

    def test_sandbox_timeout_not_retried(self):
        """SandboxTimeoutError is non-retryable — not retried."""
        fn = mock.MagicMock(side_effect=SandboxTimeoutError("timeout"))
        with pytest.raises(SandboxTimeoutError):
            retry_once(fn, session_id="test")
        assert fn.call_count == 1

    def test_custom_error_class_raised_on_failure(self):
        """Custom error_cls is raised when both attempts fail."""
        fn = mock.MagicMock(side_effect=Exception("network error"))
        with mock.patch('src.error_handler.time.sleep'):
            with pytest.raises(TTSError):
                retry_once(fn, error_cls=TTSError, session_id="test")

    def test_delay_between_attempts(self):
        """2 second delay between retry attempts."""
        fn = mock.MagicMock(side_effect=[
            StoryAgentError("fail"), "success"
        ])
        with mock.patch('src.error_handler.time.sleep') as mock_sleep:
            retry_once(fn, delay=2.0, session_id="test")
        mock_sleep.assert_called_once_with(2.0)

    def test_args_and_kwargs_forwarded(self):
        """Arguments are forwarded to the function correctly."""
        fn = mock.MagicMock(return_value="ok")
        retry_once(fn, "arg1", "arg2", kwarg="value", session_id="test")
        fn.assert_called_with("arg1", "arg2", kwarg="value")


# ── with_fallback tests ───────────────────────────────────────────────────────

class TestWithFallback:
    def test_primary_success_fallback_not_called(self):
        """Primary succeeds — fallback never called."""
        primary  = mock.MagicMock(return_value="primary_result")
        fallback = mock.MagicMock(return_value="fallback_result")
        result = with_fallback(primary, fallback, session_id="test")
        assert result == "primary_result"
        fallback.assert_not_called()

    def test_primary_fails_fallback_called(self):
        """Primary raises StoryAgentError — fallback called."""
        primary  = mock.MagicMock(side_effect=TTSError("edge tts failed"))
        fallback = mock.MagicMock(return_value="fallback_result")
        result = with_fallback(primary, fallback, session_id="test")
        assert result == "fallback_result"
        fallback.assert_called_once()

    def test_fallback_exception_propagates(self):
        """If fallback also raises, that exception propagates."""
        primary  = mock.MagicMock(side_effect=TTSError("primary failed"))
        fallback = mock.MagicMock(side_effect=RuntimeError("fallback also failed"))
        with pytest.raises(RuntimeError):
            with_fallback(primary, fallback, session_id="test")

    def test_non_story_agent_error_not_caught(self):
        """Non-StoryAgentError from primary propagates — not caught."""
        primary  = mock.MagicMock(side_effect=ValueError("unexpected"))
        fallback = mock.MagicMock(return_value="fallback")
        with pytest.raises(ValueError):
            with_fallback(primary, fallback, session_id="test")

    def test_args_forwarded_to_both(self):
        """Arguments forwarded to primary and fallback correctly."""
        primary  = mock.MagicMock(side_effect=STTError("fail"))
        fallback = mock.MagicMock(return_value="ok")
        with_fallback(primary, fallback, "arg1", session_id="test")
        fallback.assert_called_with("arg1")


# ── safe_run tests ────────────────────────────────────────────────────────────

class TestSafeRun:
    def test_success_returns_result(self):
        """safe_run returns function result on success."""
        fn = mock.MagicMock(return_value="result")
        assert safe_run(fn, session_id="test") == "result"

    def test_exception_returns_default(self):
        """safe_run returns default on any exception."""
        fn = mock.MagicMock(side_effect=Exception("any error"))
        result = safe_run(fn, default="fallback", session_id="test")
        assert result == "fallback"

    def test_default_is_none_when_not_specified(self):
        """Default return value is None when not specified."""
        fn = mock.MagicMock(side_effect=Exception("error"))
        result = safe_run(fn, session_id="test")
        assert result is None

    def test_model_armor_match_returns_empty_string(self):
        """ModelArmorMatchError returns empty string specifically."""
        fn = mock.MagicMock(side_effect=ModelArmorMatchError("injection"))
        result = safe_run(fn, default=None, session_id="test")
        assert result == ""   # not None — empty string signals blocked content

    def test_never_raises(self):
        """safe_run never raises regardless of exception type."""
        for exc in [Exception, ValueError, RuntimeError, StoryAgentError]:
            fn = mock.MagicMock(side_effect=exc("error"))
            try:
                safe_run(fn, session_id="test")
            except Exception as e:
                pytest.fail(f"safe_run raised {type(e).__name__}: {e}")

    def test_args_forwarded(self):
        """safe_run forwards args and kwargs to function."""
        fn = mock.MagicMock(return_value="ok")
        safe_run(fn, "a", "b", key="val", session_id="test")
        fn.assert_called_with("a", "b", key="val")


# ── timed decorator tests ─────────────────────────────────────────────────────

class TestTimedDecorator:
    def test_decorated_function_returns_correct_result(self):
        """@timed preserves function return value."""
        @timed("test_agent", "test_event")
        def my_func(state):
            return {**state, "result": "done"}

        result = my_func({"session_id": "abc", "value": 42})
        assert result["result"] == "done"
        assert result["value"] == 42

    def test_decorated_function_called_with_correct_args(self):
        """@timed passes arguments through correctly."""
        call_args = []

        @timed("test_agent", "test_event")
        def my_func(state, extra="default"):
            call_args.append((state, extra))
            return state

        my_func({"session_id": "abc"}, extra="custom")
        assert call_args[0][1] == "custom"

    def test_exception_propagates_through_decorator(self):
        """@timed re-raises exceptions from the wrapped function."""
        @timed("test_agent", "test_event")
        def failing_func(state):
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func({"session_id": "abc"})

    def test_timing_is_reasonable(self):
        """@timed measures duration without adding significant overhead."""
        @timed("test_agent", "test_event")
        def fast_func(state):
            time.sleep(0.01)
            return state

        start = time.perf_counter()
        fast_func({"session_id": "abc"})
        elapsed = time.perf_counter() - start
        # Should complete in under 1 second despite timing overhead
        assert elapsed < 1.0
