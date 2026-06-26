"""
shaaru_retry.py — Resilient NVIDIA API caller with exponential backoff.

Wraps OpenAI-compatible chat completion calls with:
- 4 retries max
- Exponential backoff (1s, 2s, 4s, 8s) with jitter
- 45-second total timeout ceiling
- Graceful handling of rate limits (429) and server errors (5xx)
"""

import time
import random
import logging
from typing import Optional

log = logging.getLogger("shaaru.retry")

MAX_RETRIES = 4
MAX_TOTAL_SECONDS = 150
BASE_DELAY = 1.0  # seconds


def nvidia_call(
    client,
    model: str,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.7,
    **kwargs,
) -> str:
    import concurrent.futures

    HARD_KILL_SECONDS = 120  # absolute wall-clock ceiling, no exceptions

    start_time = time.time()
    last_error = None
    current_model = model

    for attempt in range(MAX_RETRIES + 1):
        elapsed = time.time() - start_time
        if elapsed > HARD_KILL_SECONDS:
            log.error(
                f"[RETRY] Hard kill at {HARD_KILL_SECONDS}s exceeded "
                f"after {attempt} attempts for {current_model}"
            )
            break

        remaining = max(HARD_KILL_SECONDS - (time.time() - start_time), 5.0)
        request_timeout = min(kwargs.pop('timeout', 25.0), remaining - 1)
        request_timeout = max(request_timeout, 5.0)

        # snapshot values for closure — prevents loop variable capture bug
        _model = current_model
        _timeout = request_timeout
        _kwargs = {k: v for k, v in kwargs.items()}

        def _make_call():
            return client.chat.completions.create(
                model=_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=_timeout,
                **_kwargs,
            )

        try:
            # ThreadPoolExecutor hard kill — bypasses SDK timeout issues on NVIDIA endpoint
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_make_call)
                try:
                    resp = future.result(timeout=request_timeout + 2)  # +2s grace
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    raise TimeoutError(
                        f"Hard kill: {_model} did not respond in {request_timeout:.0f}s"
                    )

            content = resp.choices[0].message.content
            if attempt > 0:
                log.info(f"[RETRY] Succeeded on attempt {attempt + 1} for {_model}")
            return content or ""

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # don't retry auth or 404 errors
            if "404" in error_str or "not found" in error_str:
                log.error(f"[RETRY] Model not found ({current_model}): {e}")
                raise
            if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
                log.error(f"[RETRY] Auth error for {current_model}: {e}")
                raise

            # fallback chain — vision AND text models both covered now
            if current_model == "meta/llama-3.2-90b-vision-instruct":
                log.warning(f"[FALLBACK] 90b-vision → 11b-vision: {e}")
                current_model = "meta/llama-3.2-11b-vision-instruct"
            elif current_model == "meta/llama-3.1-70b-instruct":
                log.warning(f"[FALLBACK] 70b → 8b: {e}")
                current_model = "meta/llama-3.1-8b-instruct"

            if attempt < MAX_RETRIES:
                remaining = HARD_KILL_SECONDS - (time.time() - start_time)
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                delay = min(delay, max(0.1, remaining - 2))

                is_rate_limit = "429" in error_str or "rate" in error_str
                log.warning(
                    f"[RETRY] Attempt {attempt + 1}/{MAX_RETRIES + 1} failed "
                    f"for {current_model} ({'rate limited' if is_rate_limit else 'error'}): "
                    f"{e}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                log.error(
                    f"[RETRY] All {MAX_RETRIES + 1} attempts exhausted "
                    f"for {current_model}: {e}"
                )

    raise last_error or RuntimeError(f"nvidia_call failed for {model}")


def nvidia_call_raw(
    client,
    model: str,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.7,
    **kwargs,
):
    """
    Same retry/fallback logic as nvidia_call(), but returns the raw
    ChatCompletionMessage object instead of just .content.

    Use this when you need to inspect tool_calls on the message.
    Callers must handle resp.tool_calls and resp.content themselves.
    """
    import concurrent.futures

    HARD_KILL_SECONDS = 120
    start_time = time.time()
    last_error = None
    current_model = model

    for attempt in range(MAX_RETRIES + 1):
        elapsed = time.time() - start_time
        if elapsed > HARD_KILL_SECONDS:
            log.error(
                f"[RETRY-RAW] Hard kill at {HARD_KILL_SECONDS}s exceeded "
                f"after {attempt} attempts for {current_model}"
            )
            break

        remaining = max(HARD_KILL_SECONDS - (time.time() - start_time), 5.0)
        request_timeout = min(kwargs.pop('timeout', 25.0), remaining - 1)
        request_timeout = max(request_timeout, 5.0)

        _model   = current_model
        _timeout = request_timeout
        _kwargs  = {k: v for k, v in kwargs.items()}

        def _make_call():
            return client.chat.completions.create(
                model=_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=_timeout,
                **_kwargs,
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_make_call)
                try:
                    resp = future.result(timeout=request_timeout + 2)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    raise TimeoutError(
                        f"Hard kill: {_model} did not respond in {request_timeout:.0f}s"
                    )

            # Return the full message object
            message = resp.choices[0].message
            if attempt > 0:
                log.info(f"[RETRY-RAW] Succeeded on attempt {attempt + 1} for {_model}")
            return message

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            if "404" in error_str or "not found" in error_str:
                log.error(f"[RETRY-RAW] Model not found ({current_model}): {e}")
                raise
            if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
                log.error(f"[RETRY-RAW] Auth error for {current_model}: {e}")
                raise

            if current_model == "meta/llama-3.2-90b-vision-instruct":
                log.warning(f"[FALLBACK-RAW] 90b-vision → 11b-vision: {e}")
                current_model = "meta/llama-3.2-11b-vision-instruct"
            elif current_model == "meta/llama-3.1-70b-instruct":
                log.warning(f"[FALLBACK-RAW] 70b → 8b: {e}")
                current_model = "meta/llama-3.1-8b-instruct"

            if attempt < MAX_RETRIES:
                remaining = HARD_KILL_SECONDS - (time.time() - start_time)
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                delay = min(delay, max(0.1, remaining - 2))
                is_rate_limit = "429" in error_str or "rate" in error_str
                log.warning(
                    f"[RETRY-RAW] Attempt {attempt + 1}/{MAX_RETRIES + 1} failed "
                    f"for {current_model} ({'rate limited' if is_rate_limit else 'error'}): "
                    f"{e}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                log.error(
                    f"[RETRY-RAW] All {MAX_RETRIES + 1} attempts exhausted "
                    f"for {current_model}: {e}"
                )

    raise last_error or RuntimeError(f"nvidia_call_raw failed for {model}")

