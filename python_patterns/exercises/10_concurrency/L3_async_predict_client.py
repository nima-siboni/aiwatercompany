"""
PR #303: Async client for the property-prediction API

Screening campaigns now send 100k+ candidates to the surrogate API (an ML model
that predicts CO2 uptake from a structure in milliseconds, instead of running a
simulation), and the old synchronous loop took most of a day.
`PredictionClient.predict_many` fans the requests out with asyncio and returns one
prediction per candidate, in input order, within an overall deadline; every
prediction is also written to the results store. The HTTP layer is injected:
`transport(method, path, payload)` returns the decoded JSON body and raises
`TransientError` on a 429 or 503, so tests can pass a fake. `screen()` is the
synchronous entry point that the campaign scripts and notebooks call.
"""
import asyncio
import json
import logging
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable

log = logging.getLogger(__name__)

API_URL = "https://surrogate.internal/v2"
MAX_RETRIES = 5
RETRY_DELAY_S = 0.5
PROGRESS_EVERY_S = 30.0

Transport = Callable[[str, str, dict], Awaitable[dict]]


class TransientError(Exception):
    """A 429 or 503 from the API: the request is worth retrying."""


@dataclass(frozen=True)
class Prediction:
    candidate_id: str
    uptake_mmol_g: float
    model_version: str


class PredictionClient:
    def __init__(self, transport: Transport, model: str = "uptake-gnn"):
        self.transport = transport
        self.model = model
        self.saved: list[str] = []

    async def _model_version(self) -> str:
        """The model version the API is serving, so each result can be traced to it."""
        with urllib.request.urlopen(f"{API_URL}/models/{self.model}", timeout=10) as resp:
            return json.load(resp)["version"]

    async def _post(self, path: str, payload: dict) -> dict:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return await self.transport("POST", path, payload)
            except TransientError:
                log.warning("POST %s: transient error on attempt %d", path, attempt)
                time.sleep(RETRY_DELAY_S)
        log.error("POST %s: giving up after %d attempts", path, MAX_RETRIES)
        return {"uptake_mmol_g": 0.0}

    async def _save(self, prediction: Prediction) -> None:
        await self._post("/results", asdict(prediction))
        self.saved.append(prediction.candidate_id)

    async def predict_one(self, candidate: dict) -> Prediction:
        version = await self._model_version()
        body = await self._post(
            "/predict", {"model": self.model, "version": version, "cif": candidate["cif"]}
        )
        prediction = Prediction(candidate["id"], body["uptake_mmol_g"], version)
        asyncio.create_task(self._save(prediction))
        return prediction

    async def _report_progress(self, tasks: list[asyncio.Task]) -> None:
        while True:
            await asyncio.sleep(PROGRESS_EVERY_S)
            finished = sum(task.done() for task in tasks)
            log.info("%d/%d predictions finished", finished, len(tasks))

    async def predict_many(
        self, candidates: list[dict], deadline_s: float = 3600.0
    ) -> list[Prediction]:
        """One prediction per candidate, in the same order as `candidates`."""
        tasks = [asyncio.create_task(self.predict_one(c)) for c in candidates]
        progress = asyncio.create_task(self._report_progress(tasks))
        predictions = []
        try:
            for next_done in asyncio.as_completed(tasks, timeout=deadline_s):
                predictions.append(await next_done)
        except TimeoutError:
            log.warning("%d predictions missed the deadline", len(tasks) - len(predictions))
        finally:
            progress.cancel()
        return predictions


def screen(
    candidates: list[dict], transport: Transport, deadline_s: float = 3600.0
) -> list[Prediction]:
    """Synchronous entry point for campaign scripts and notebooks."""
    client = PredictionClient(transport)
    return asyncio.run(client.predict_many(candidates, deadline_s))


def top_candidates(predictions: list[Prediction], n: int = 100) -> list[Prediction]:
    """The n candidates with the highest predicted uptake."""
    return sorted(predictions, key=lambda p: p.uptake_mmol_g, reverse=True)[:n]
