"""
Reference fix for PR #302 (python_patterns/exercises/10_concurrency/L2_dft_queue.py).
See the answer key for the reasoning.
"""
import logging
import queue
import threading
from typing import Any, Iterable

log = logging.getLogger(__name__)

_STOP = object()


class DFTCampaign:
    def __init__(self, client: Any, n_workers: int = 8, wait_timeout_s: float = 12 * 3600):
        self.client = client
        self.n_workers = n_workers
        self.wait_timeout_s = wait_timeout_s
        # Bounded: when the workers fall behind, put() blocks the producer (backpressure).
        self.tasks: queue.Queue = queue.Queue(maxsize=2 * n_workers)
        self.energies: dict[str, float] = {}
        self.failures: list[str] = []
        self._lock = threading.Lock()

    @property
    def done(self) -> int:
        """How many candidates have finished so far, successfully or not."""
        return len(self.energies) + len(self.failures)

    def _worker(self) -> None:
        """Take candidates off the queue until told to stop."""
        while True:
            candidate = self.tasks.get()
            try:
                if candidate is _STOP:
                    return
                self._relax(candidate)
            except Exception:
                log.exception("DFT job failed for %s", candidate["id"])
                self.failures.append(candidate["id"])
            finally:
                self.tasks.task_done()  # on every path, or tasks.join() never returns

    def _relax(self, candidate: dict) -> None:
        job_id = self.client.submit(candidate["cif"], functional="PBE")
        result = self.client.wait(job_id, timeout=self.wait_timeout_s)
        with self._lock:
            self.energies[candidate["id"]] = result["energy_ev"]  # IDs are unique; formulas aren't
        self.client.upload_artifacts(job_id)  # slow I/O stays outside the lock

    def run(self, candidates: Iterable[dict]) -> dict[str, float]:
        """Relax every candidate; returns final energies (eV) keyed by candidate ID."""
        workers = [
            threading.Thread(target=self._worker, name=f"dft-worker-{i}", daemon=True)
            for i in range(self.n_workers)
        ]
        for worker in workers:
            worker.start()
        try:
            for candidate in candidates:
                self.tasks.put(candidate)
        finally:
            # One sentinel per worker, even if the candidate stream raised.
            for _ in workers:
                self.tasks.put(_STOP)
            for worker in workers:
                worker.join()
        log.info("campaign done: %d energies, %d failures", len(self.energies), len(self.failures))
        return self.energies
