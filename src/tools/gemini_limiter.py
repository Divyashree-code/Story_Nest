"""
src/tools/gemini_limiter.py

Thread-safe proactive rate limiter for Gemini API calls.

Free tier limits for gemini-2.5-flash:
    15 requests per minute (RPM)
    1,000,000 tokens per minute (TPM)
    1,500 requests per day (RPD)

Strategy:
    - Enforce a minimum gap of 60/14 ≈ 4.3 s between calls (14 RPM, giving
      one request of headroom under the 15 RPM cap).
    - On a 429 response, parse the suggested retry_delay from the error
      message and wait exactly that long rather than a fixed back-off.

Usage:
    from src.tools.gemini_limiter import gemini_limiter

    gemini_limiter.wait(session_id)          # call before every Gemini request
    gemini_limiter.backoff(exc, session_id)  # call on 429 — waits & returns True
                                             # returns False if not a rate-limit error
"""

import re
import threading
import time

from src.logger import get_logger

log = get_logger("gemini_limiter")

_RPM_LIMIT   = 14          # stay under the 15 RPM free-tier cap
_MIN_INTERVAL = 60.0 / _RPM_LIMIT   # ~4.3 s between calls
_DEFAULT_BACKOFF = 30.0    # fallback wait when retry_delay not in error


class GeminiRateLimiter:
    """
    Module-level singleton. Shared across all threads so the per-minute
    budget is enforced globally, not per-agent.
    """

    def __init__(self):
        self._lock      = threading.Lock()
        self._last_call = 0.0   # monotonic timestamp of last completed call

    def wait(self, session_id: str = "") -> None:
        """
        Block until it is safe to make the next Gemini call.
        Enforces at least _MIN_INTERVAL seconds between consecutive calls.
        """
        with self._lock:
            now     = time.monotonic()
            gap     = _MIN_INTERVAL - (now - self._last_call)
            if gap > 0:
                log.debug(
                    "gemini_rate_limit_wait",
                    session_id=session_id,
                    wait_s=round(gap, 2),
                )
                time.sleep(gap)
            self._last_call = time.monotonic()

    def backoff(self, exc: Exception, session_id: str = "") -> bool:
        """
        Call this when a Gemini call raises an exception.

        Returns True  — exception was a 429, waited the suggested delay.
        Returns False — exception was something else, caller should re-raise.
        """
        err = str(exc)
        is_rate_limit = (
            "429"        in err
            or "quota"   in err.lower()
            or "too many" in err.lower()
            or "resource_exhausted" in err.lower()
        )
        if not is_rate_limit:
            return False

        # Parse suggested retry delay from the error body, e.g. "retry in 13.15s"
        wait = _DEFAULT_BACKOFF
        match = re.search(r"retry[_\s]delay[^\d]*(\d+(?:\.\d+)?)", err, re.IGNORECASE)
        if not match:
            match = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", err, re.IGNORECASE)
        if match:
            wait = float(match.group(1)) + 2.0   # add 2 s buffer

        log.warning(
            "gemini_429_backing_off",
            session_id=session_id,
            wait_s=round(wait, 1),
        )
        time.sleep(wait)
        return True


# Singleton shared by all agents
gemini_limiter = GeminiRateLimiter()
