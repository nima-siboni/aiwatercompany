"""
Reference fix for PR #303 (python_patterns/exercises/10_concurrency/L3_async_predict_client.py).
See the answer key for the reasoning.
"""
import asyncio
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable

log = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_DELAY_S = 0.5
REQUEST_TIMEOUT_S = 60.0
PROGRESS_EVERY_S = 30.0

Transport = Callable[[str, str, dict], Awaitable[dict]]


class TransientError(Exception):
    """A 429 or 503 from the API: the request is worth retrying."""


@dataclass(frozen=True)
class Prediction:
    candidate_id: str
    uptake_mmol_g: float | None  # None when the prediction failed; see `error`
    model_version: str
    error: str | None = None


class PredictionClient:
    def __init__(self, transport: Transport, model: str = "uptake-gnn", max_concurrency: int = 32):
        self.transport = transport
        self.model = model
        self.max_concurrency = max_concurrency
        self.saved: list[str] = []

    async def _call(self, method: str, path: str, payload: dict) -> dict:
        """One API call with a timeout. Transient errors are retried with backoff and jitter."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    self.transport(method, path, payload), REQUEST_TIMEOUT_S
                )
            except TransientError:
                if attempt == MAX_RETRIES:
                    raise
                log.warning("%s %s: transient error on attempt %d", method, path, attempt)
                await asyncio.sleep(RETRY_DELAY_S * 2 ** (attempt - 1) * random.uniform(0.5, 1.5))

    async def _model_version(self) -> str:
        """The model version the API is serving, so each result can be traced to it."""
        body = await self._call("GET", f"/models/{self.model}", {})
        return body["version"]

    async def _save(self, prediction: Prediction) -> None:
        await self._call("POST", "/results", asdict(prediction))
        self.saved.append(prediction.candidate_id)

    async def predict_one(self, candidate: dict, version: str | None = None) -> Prediction:
        if version is None:
            version = await self._model_version()
        body = await self._call(
            "POST", "/predict", {"model": self.model, "version": version, "cif": candidate["cif"]}
        )
        prediction = Prediction(candidate["id"], body["uptake_mmol_g"], version)
        await self._save(prediction)  # part of the unit of work, not fire-and-forget
        return prediction

    async def _report_progress(self, results: list) -> None:
        while True:
            await asyncio.sleep(PROGRESS_EVERY_S)
            finished = sum(r is not None for r in results)
            log.info("%d/%d predictions finished", finished, len(results))

    async def predict_many(
        self, candidates: list[dict], deadline_s: float = 3600.0
    ) -> list[Prediction]:
        """One prediction per candidate, in the same order as `candidates`.

        A candidate that fails or misses the deadline comes back with
        `uptake_mmol_g=None` and an `error`, never with a made-up value.
        """
        version = await self._model_version()  # once per run: one model for every result
        results: list[Prediction | None] = [None] * len(candidates)
        todo = iter(enumerate(candidates))

        async def worker() -> None:
            # The workers share one iterator, so each candidate is taken once and at
            # most max_concurrency requests are in flight.
            for i, candidate in todo:
                try:
                    results[i] = await self.predict_one(candidate, version)
                except Exception as exc:
                    results[i] = Prediction(candidate["id"], None, version, error=repr(exc))

        workers = [asyncio.create_task(worker()) for _ in range(self.max_concurrency)]
        progress = asyncio.create_task(self._report_progress(results))
        try:
            await asyncio.wait(workers, timeout=deadline_s)
        finally:
            for task in (*workers, progress):
                task.cancel()  # no-op for tasks that already finished
            await asyncio.gather(*workers, progress, return_exceptions=True)
        missed = sum(r is None for r in results)
        if missed:
            log.warning("%d predictions missed the %.0f s deadline", missed, deadline_s)
        return [
            r if r is not None else Prediction(c["id"], None, version, error="missed the deadline")
            for c, r in zip(candidates, results)
        ]


def screen(
    candidates: list[dict], transport: Transport, deadline_s: float = 3600.0
) -> list[Prediction]:
    """Synchronous entry point for campaign scripts.

    In a notebook, prefer `await PredictionClient(transport).predict_many(...)`.
    """

    def run() -> list[Prediction]:
        return asyncio.run(PredictionClient(transport).predict_many(candidates, deadline_s))

    try:
        asyncio.get_running_loop()
    except RuntimeError:  # no loop running in this thread: the normal script case
        return run()
    # Called from inside a running loop (Jupyter): asyncio.run would raise there, so
    # use a fresh loop in a worker thread. The transport must not be tied to that loop.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(run).result()


def top_candidates(predictions: list[Prediction], n: int = 100) -> list[Prediction]:
    """The n candidates with the highest predicted uptake; failed predictions are skipped."""
    ok = [p for p in predictions if p.uptake_mmol_g is not None]
    return sorted(ok, key=lambda p: p.uptake_mmol_g, reverse=True)[:n]
