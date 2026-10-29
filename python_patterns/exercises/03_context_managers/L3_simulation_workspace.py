"""
PR #233: SimulationWorkspace context manager for DFT/MD/GCMC jobs

Every simulation driver we have (LAMMPS for MD, VASP for DFT, RASPA for GCMC) repeats the
same setup: make a scratch directory, write the input files, set the thread count, make sure
no other worker is already running the same job, run the binary, and clean up afterwards.
This PR moves all of that into one context manager and uses it in `run_batch`, which the
orchestrator calls with a few hundred jobs at a time on a 64-core node:

    with SimulationWorkspace("mof-5_co2_298K", {"in.lammps": script}, threads=8) as ws:
        result = ws.run(["lmp", "-in", "in.lammps"])

Our thin wrapper scripts around each code write `results.json` (energies in eV, uptakes in
mmol/g) into the working directory when the run finishes.
"""
import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)

SCRATCH_ROOT = Path(os.environ.get("SCRATCH", "."))
LOCK_DIR = Path("/tmp/sim-locks")


class WorkspaceBusy(RuntimeError):
    """Another worker is already running this job."""


class SimulationWorkspace:
    def __init__(self, job_id: str, inputs: dict[str, str], threads: int = 1, keep: bool = False):
        self.job_id = job_id
        self.inputs = inputs
        self.threads = threads
        self.keep = keep
        self.path = SCRATCH_ROOT / job_id
        self.lock_path = LOCK_DIR / f"{job_id}.lock"

    def __enter__(self) -> "SimulationWorkspace":
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            raise WorkspaceBusy(f"job {self.job_id} is already running")
        self.lock_path.write_text(str(os.getpid()))

        self.path.mkdir(parents=True, exist_ok=True)
        for name, content in self.inputs.items():
            (self.path / name).write_text(content)

        self._prev_cwd = os.getcwd()
        self._prev_threads = os.environ.get("OMP_NUM_THREADS")
        os.chdir(self.path)
        os.environ["OMP_NUM_THREADS"] = str(self.threads)
        log.info("job %s: workspace ready at %s", self.job_id, self.path)
        return self

    def run(self, cmd: list[str], timeout: float = 24 * 3600) -> dict:
        """Run one simulation in the workspace and return its parsed results.json."""
        start = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        Path("stdout.log").write_text(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(f"{cmd[0]} exited with code {proc.returncode}")
        result = json.loads(Path("results.json").read_text())
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
        os.chdir(self._prev_cwd)
        if self._prev_threads is not None:
            os.environ["OMP_NUM_THREADS"] = self._prev_threads
        if not self.keep:
            shutil.rmtree(self.path)
        self.lock_path.unlink()

    def __del__(self):
        # reclaim scratch space even if the workspace was never exited cleanly
        if not self.keep and self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)


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
