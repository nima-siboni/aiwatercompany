"""
Reference fix for PR #212 (python_patterns/exercises/01_decorators/L2_retry_backoff.py).
See the answer key for the reasoning.
"""
import functools
import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
import uuid

logger = logging.getLogger(__name__)

SIM_API_URL = os.environ.get("SIM_API_URL", "http://sim-api.internal:8080")

RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def is_transient(exc: Exception) -> bool:
    """Timeouts, dropped connections and 429/5xx responses. Never other 4xx, never bugs."""
    if isinstance(exc, urllib.error.HTTPError):  # checked first: HTTPError subclasses URLError
        return exc.code in RETRYABLE_HTTP_CODES
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError))


def retry(*, max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 30.0,
          retry_if=is_transient):
    """Retry with capped exponential backoff and full jitter.

    The arguments are keyword-only, so a bare `@retry` raises TypeError at import
    instead of silently replacing the function with the inner decorator.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_attempts - 1 or not retry_if(exc):
                        raise
                    # Local to this call, so no backoff state is shared between calls or threads.
                    delay = random.uniform(0, min(max_delay, base_delay * 2**attempt))
                    logger.warning(
                        "%s failed (attempt %d of %d), retrying in %.1f s: %s",
                        func.__name__, attempt + 1, max_attempts, delay, exc,
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        SIM_API_URL + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def submit_job(structure_id: str, calc: str = "dft_relax") -> str:
    """Queue a calculation and return its job id."""
    # One key per logical submission, reused by every retry of it. This assumes sim-api
    # deduplicates on the key (confirm with its owners). If it doesn't, don't retry the
    # POST on a timeout; look the job up by a client-side tag instead.
    payload = {"structure_id": structure_id, "calc": calc, "idempotency_key": str(uuid.uuid4())}
    return _post_job(payload)


@retry(max_attempts=5, base_delay=2.0)
def _post_job(payload: dict) -> str:
    return _request("POST", "/jobs", payload)["job_id"]


@retry()
def get_job_status(job_id: str) -> str:
    """One of PENDING, RUNNING, COMPLETED or FAILED."""
    return _request("GET", f"/jobs/{job_id}")["status"]


@retry(max_attempts=5)
def get_result(job_id: str) -> dict:
    return _request("GET", f"/jobs/{job_id}/result")


def relax(structure_id: str, poll_s: float = 60.0) -> dict:
    """Submit a relaxation, wait for it to finish and return the result.

    Not retried as a whole: each sim-api call retries itself, and a FAILED job is
    a result to report, not a transient error.
    """
    job_id = submit_job(structure_id)
    while True:
        status = get_job_status(job_id)
        if status == "COMPLETED":
            return get_result(job_id)
        if status == "FAILED":
            raise RuntimeError(f"relaxation {job_id} of {structure_id} failed")
        time.sleep(poll_s)
