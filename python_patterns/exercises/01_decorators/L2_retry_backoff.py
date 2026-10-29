"""
PR #212: Retry transient failures when talking to sim-api

sim-api is our internal service that queues DFT relaxations on the cluster and
serves their results. Under load it times out or returns 503s, and at the
moment a single failed request kills the whole active-learning round. This adds
a `retry` decorator with exponential backoff and puts it on every sim-api call.
(A relaxation moves a structure's atoms to their lowest-energy positions. One
takes 1-6 hours of cluster time, and we read the final energy from the result.)
"""
import functools
import json
import logging
import os
import time
import urllib.request

logger = logging.getLogger(__name__)

SIM_API_URL = os.environ.get("SIM_API_URL", "http://sim-api.internal:8080")


def retry(max_attempts: int = 3, base_delay: float = 1.0, exceptions=(Exception,)):
    """Retry the decorated function with exponential backoff."""

    def decorator(func):
        delay = base_delay

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal delay
            for attempt in range(1, max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    logger.warning(
                        "%s failed (attempt %d of %d): %s", func.__name__, attempt, max_attempts, exc
                    )
                    time.sleep(delay)
                    delay *= 2

        return wrapper

    return decorator


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        SIM_API_URL + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


@retry(max_attempts=5, base_delay=2.0)
def submit_job(structure_id: str, calc: str = "dft_relax") -> str:
    """Queue a calculation and return its job id."""
    return _request("POST", "/jobs", {"structure_id": structure_id, "calc": calc})["job_id"]


@retry
def get_job_status(job_id: str) -> str:
    """One of PENDING, RUNNING, COMPLETED or FAILED."""
    return _request("GET", f"/jobs/{job_id}")["status"]


@retry(max_attempts=5)
def get_result(job_id: str) -> dict:
    return _request("GET", f"/jobs/{job_id}/result")


@retry(max_attempts=3)
def relax(structure_id: str, poll_s: float = 60.0) -> dict:
    """Submit a relaxation, wait for it to finish and return the result."""
    job_id = submit_job(structure_id)
    while True:
        status = get_job_status(job_id)
        if status == "COMPLETED":
            return get_result(job_id)
        if status == "FAILED":
            raise RuntimeError(f"relaxation {job_id} of {structure_id} failed")
        time.sleep(poll_s)
