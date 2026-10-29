"""
PR #302: Stream generated candidates through a pool of DFT workers

The generative model streams candidate structures much faster than the cluster
can relax them with DFT (a quantum-chemistry calculation that moves the atoms to
their lowest-energy positions and reports that energy). `DFTCampaign.run` puts the
candidates on a queue that `n_workers` threads drain: each worker submits a job
through the cluster client (`submit(cif, functional) -> job_id`,
`wait(job_id, timeout=None) -> result`, `upload_artifacts(job_id)`), waits for it
and records the final energy. `run` returns the energies keyed by formula; the
active-learning loop uses them to choose what to generate next.
"""
import logging
import queue
import threading
from typing import Any, Iterable

log = logging.getLogger(__name__)

_STOP = object()


class DFTCampaign:
    def __init__(self, client: Any, n_workers: int = 8):
        self.client = client
        self.n_workers = n_workers
        self.tasks: queue.Queue = queue.Queue()
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
            if candidate is _STOP:
                self.tasks.task_done()
                return
            try:
                job_id = self.client.submit(candidate["cif"], functional="PBE")
                result = self.client.wait(job_id)
            except Exception:
                log.exception("DFT job failed for %s", candidate["id"])
                self.failures.append(candidate["id"])
                continue
            with self._lock:
                self.energies[candidate["formula"]] = result["energy_ev"]
                self.client.upload_artifacts(job_id)
            self.tasks.task_done()

    def run(self, candidates: Iterable[dict]) -> dict[str, float]:
        """Relax every candidate; returns final energies (eV) keyed by formula."""
        for i in range(self.n_workers):
            worker = threading.Thread(target=self._worker, name=f"dft-worker-{i}", daemon=True)
            worker.start()
        for candidate in candidates:
            self.tasks.put(candidate)
        self.tasks.put(_STOP)
        self.tasks.join()
        log.info("campaign done: %d energies, %d failures", len(self.energies), len(self.failures))
        return self.energies
