"""Reference fix for PR #233 (python_patterns/exercises/03_context_managers/L3_simulation_workspace.py).
See the answer key for the reasoning."""
import fcntl
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

# No fallback to the current directory: workspaces get deleted.
SCRATCH_ROOT = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
LOCK_DIR = Path("/tmp/sim-locks")
_JOB_ID = re.compile(r"[\w.-]+")


class WorkspaceBusy(RuntimeError):
    """Another worker is already running this job."""


class SimulationWorkspace:
    def __init__(self, job_id: str, inputs: dict[str, str], threads: int = 1, keep: bool = False):
        if not _JOB_ID.fullmatch(job_id):
            raise ValueError(f"job_id must be a plain name, got {job_id!r}")
        self.job_id = job_id
        self.inputs = inputs
        self.threads = threads
        self.keep = keep
        self.path: Path | None = None  # a fresh directory, created in __enter__
        self.lock_path = LOCK_DIR / f"{job_id}.lock"
        self._stack: ExitStack | None = None

    def __enter__(self) -> "SimulationWorkspace":
        if SCRATCH_ROOT is None:
            raise RuntimeError("$SCRATCH is not set; refusing to create workspaces in the cwd")
        with ExitStack() as stack:
            stack.enter_context(self._lock())
            SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
            self.path = Path(tempfile.mkdtemp(prefix=f"{self.job_id}-", dir=SCRATCH_ROOT)).resolve()
            stack.push(self._cleanup)
            for name, content in self.inputs.items():
                (self.path / name).write_text(content)
            # Setup succeeded: hand the registered cleanups to __exit__. If anything above
            # raised, the `with` has already undone whatever had been set up.
            self._stack = stack.pop_all()
        log.info("job %s: workspace ready at %s", self.job_id, self.path)
        return self

    @contextmanager
    def _lock(self):
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise WorkspaceBusy(f"job {self.job_id} is already running") from None
        try:
            yield
        finally:
            os.close(fd)  # releases the lock; the kernel also releases it if the process dies

    def _cleanup(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            log.warning("job %s failed; keeping %s for debugging", self.job_id, self.path)
        elif not self.keep:
            try:
                shutil.rmtree(self.path)
            except OSError:  # never let cleanup replace the job's own outcome
                log.warning("job %s: could not remove %s", self.job_id, self.path, exc_info=True)

    def run(self, cmd: list[str], timeout: float = 24 * 3600) -> dict:
        """Run one simulation in the workspace and return its parsed results.json."""
        env = {**os.environ, "OMP_NUM_THREADS": str(self.threads)}
        start = time.monotonic()
        with open(self.path / "stdout.log", "w") as out, open(self.path / "stderr.log", "w") as err:
            proc = subprocess.run(cmd, cwd=self.path, env=env, stdout=out, stderr=err, timeout=timeout)
        if proc.returncode != 0:
            tail = (self.path / "stderr.log").read_text()[-2000:]
            raise RuntimeError(f"{cmd[0]} exited with code {proc.returncode} in {self.path}:\n{tail}")
        result = json.loads((self.path / "results.json").read_text())
        result["wall_time_s"] = time.monotonic() - start
        return result

    def save_outputs(self, dest: Path, patterns: tuple[str, ...] = ("results.json", "*.log")) -> list[Path]:
        """Copy selected outputs out of scratch before the workspace is cleaned up."""
        dest.mkdir(parents=True, exist_ok=True)
        saved = []
        for pattern in patterns:
            for src in self.path.glob(pattern):
                saved.append(Path(shutil.copy2(src, dest / src.name)))
        return saved

    def __exit__(self, exc_type, exc, tb):
        stack, self._stack = self._stack, None
        return stack.__exit__(exc_type, exc, tb)


def run_job(job: dict) -> dict:
    with SimulationWorkspace(job["job_id"], job["inputs"], threads=job.get("threads", 1)) as ws:
        return ws.run(job["cmd"])


def run_batch(jobs: list[dict], max_workers: int = 8) -> dict[str, dict]:
    """Run jobs concurrently; returns {job_id: result} for the jobs that succeeded."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_job, job): job["job_id"] for job in jobs}
        for fut in as_completed(futures):
            job_id = futures[fut]
            try:
                results[job_id] = fut.result()
            except WorkspaceBusy:
                log.warning("skipping %s: already running elsewhere", job_id)
            except Exception:
                log.exception("job %s failed", job_id)
    log.info("batch finished: %d of %d jobs succeeded", len(results), len(jobs))
    return results
