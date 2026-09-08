"""
Process-wide throttle for NVIDIA NIM API calls.

Every LLM call in this project (generation, claim decomposition, and the
claim verifier's LLM-judge escalation) goes through the same NVIDIA account
and the same single model, so they all draw on one shared 40 requests/minute
free-tier budget (see configs.models.NVIDIA_RPM_LIMIT). A sliding window over
call timestamps blocks the caller just long enough to stay under that budget.

``call()`` is the entry point callers should use, not ``acquire()`` directly:
the underlying OpenAI-like clients retry a 429 internally with their own
backoff, and those retries fire real HTTP requests that never pass back
through this module -- a limiter that only gates the *first* attempt still
lets retries burst past the budget. ``call()`` gates every attempt, including
retries, so callers must construct their client with ``max_retries=0`` and
let this module own all retry/backoff decisions instead.
"""

import logging
import threading
import time
from collections import deque

import configs.models as _models

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_call_times: deque = deque()

#: Module-level so the self-check below can shrink it instead of sleeping a
#: real 60s window to prove the throttle works.
WINDOW_SECONDS = 60.0

#: How many total attempts call() makes for one logical request (1 + retries).
MAX_ATTEMPTS = 4


def acquire() -> None:
    """Block until issuing another call would stay within the rolling window."""
    limit = _models.NVIDIA_RPM_LIMIT
    with _lock:
        now = time.monotonic()
        while _call_times and now - _call_times[0] >= WINDOW_SECONDS:
            _call_times.popleft()
        if len(_call_times) >= limit:
            sleep_for = WINDOW_SECONDS - (now - _call_times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            while _call_times and now - _call_times[0] >= WINDOW_SECONDS:
                _call_times.popleft()
        _call_times.append(now)


def _is_rate_limit_error(exc: Exception) -> bool:
    return "429" in str(exc) or "RateLimitError" in type(exc).__name__


def call(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), acquiring a slot before every attempt.

    Retries only on a rate-limit-shaped error, up to MAX_ATTEMPTS total
    attempts, with a short extra backoff on top of whatever acquire() already
    waited -- NVIDIA's own guidance is that the limit "varies with current
    overall traffic," so a fixed small buffer after a 429 is cheap insurance
    against immediately re-hitting the same transient congestion.
    """
    last_exc: Exception = RuntimeError("unreachable")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        acquire()
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_rate_limit_error(exc) or attempt == MAX_ATTEMPTS:
                raise
            backoff = 2.0 * attempt
            logger.warning(
                "Rate-limited (attempt %d/%d), backing off %.1fs: %s",
                attempt, MAX_ATTEMPTS, backoff, exc,
            )
            time.sleep(backoff)
    raise last_exc


def demo() -> None:
    """Self-check: the (n+1)th acquire at a tiny limit/window blocks measurably,
    and call() retries a rate-limit-shaped failure through to success."""
    global WINDOW_SECONDS
    original_limit = _models.NVIDIA_RPM_LIMIT
    original_window = WINDOW_SECONDS
    _models.NVIDIA_RPM_LIMIT = 3
    WINDOW_SECONDS = 1.0
    try:
        _call_times.clear()
        start = time.monotonic()
        for _ in range(4):
            acquire()
        elapsed = time.monotonic() - start
        assert elapsed > 0.5, f"expected a throttled wait near 1s, got {elapsed:.2f}s"
        assert elapsed < 2.0, f"self-check ran too long ({elapsed:.2f}s) -- window not shrinking?"
        print(f"rate_limiter self-check OK: 4 acquires at limit=3/window=1s took {elapsed:.2f}s")

        _call_times.clear()
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("Error code: 429 - Too Many Requests")
            return "ok"

        assert call(flaky) == "ok" and attempts["n"] == 3
        print("rate_limiter self-check OK: call() retried a 429 through to success")
    finally:
        _models.NVIDIA_RPM_LIMIT = original_limit
        WINDOW_SECONDS = original_window
        _call_times.clear()


if __name__ == "__main__":
    demo()
