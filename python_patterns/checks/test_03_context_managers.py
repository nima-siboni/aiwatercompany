"""Checks for the 03_context_managers drills: every planted bug fails on the PR code and passes on the fix."""
import errno
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import pytest

from harness import bug, fine


def _tracking(real, sink):
    """Wrap a factory (open, sqlite3.connect) so every object it returns is recorded in `sink`."""

    def wrapper(*args, **kwargs):
        obj = real(*args, **kwargs)
        sink.append(obj)
        return obj

    return wrapper


# --- L1: Timer, in_dir and read_energy for DFT post-processing ----------------------------

L1 = "03_context_managers/L1_dft_helpers"

TOTEN = "  free  energy   TOTEN  =      {:.8f} eV\n"


def _outcar(path, energies):
    path.write_text("".join([" running on 64 cores\n"] + [TOTEN.format(e) for e in energies]))
    return path


@bug(L1)
def test_L1_timer_swallows_exceptions(m):
    """Blocker 1: Timer.__exit__ returns True, so exceptions from the block are swallowed."""
    propagated = False
    try:
        with m.Timer("parse"):
            raise FileNotFoundError("OUTCAR")
    except FileNotFoundError:
        propagated = True
    assert propagated, "the exception never left the with-block"


@bug(L1)
def test_L1_in_dir_restores_cwd_on_error(m, tmp_path, monkeypatch):
    """Major 2: in_dir has no try/finally, so the cwd isn't restored when the block raises."""
    monkeypatch.chdir(tmp_path)
    job = tmp_path / "runs" / "mof-2"
    job.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        with m.in_dir(job):
            raise FileNotFoundError("OUTCAR")
    assert os.path.samefile(os.getcwd(), tmp_path)


@bug(L1)
def test_L1_read_energy_unfinished_job(m, tmp_path):
    """Major 3: an OUTCAR with no TOTEN line returns None instead of raising."""
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(" running on 64 cores\n POSCAR found: 108 atoms\n")
    try:
        energy = m.read_energy(str(outcar))
    except ValueError:
        return
    raise AssertionError(f"unfinished job returned {energy!r} instead of raising")


@bug(L1)
def test_L1_read_energy_closes_file(m, tmp_path, monkeypatch):
    """Minor 4: read_energy opens the OUTCAR without `with`, so nothing closes it."""
    opened = []
    monkeypatch.setattr(m, "open", _tracking(open, opened), raising=False)
    m.read_energy(str(_outcar(tmp_path / "OUTCAR", [-101.5])))
    assert opened and all(f.closed for f in opened)


class _SteppedClock:
    """The wall clock steps back an hour (an NTP correction) inside the block; the monotonic clocks don't."""

    def __init__(self):
        self._wall = iter([1_700_003_600.0, 1_700_000_000.5])
        self._mono = iter([100.0, 100.5])

    def time(self):
        return next(self._wall)

    def perf_counter(self):
        return next(self._mono)

    monotonic = perf_counter


@bug(L1)
def test_L1_timer_monotonic_clock(m, monkeypatch):
    """Minor 5: Timer uses time.time(), so a clock correction gives a negative duration."""
    monkeypatch.setattr(m, "time", _SteppedClock())
    with m.Timer("parse") as t:
        pass
    assert 0 <= t.elapsed < 1


@fine(L1)
def test_L1_last_toten_is_returned(m, tmp_path):
    """Red herring: split()[-2] is the number (the last token is 'eV'), and the last match wins."""
    outcar = _outcar(tmp_path / "OUTCAR", [-98.25, -101.5, -101.75])
    assert m.read_energy(str(outcar)) == -101.75


# --- L2: transaction(), atomic_write() and the results store --------------------------------

L2 = "03_context_managers/L2_results_store"


def _point(mof_id, pressure, uptake, status="ok"):
    return {"mof_id": mof_id, "pressure_bar": pressure, "uptake_mmol_g": uptake, "status": status}


ISOTHERM = [_point("MOF-5", 1.0, 2.1), _point("MOF-5", 10.0, 7.9), _point("MOF-5", 30.0, 9.4)]


def _count(db):
    with closing(sqlite3.connect(db)) as conn:
        return conn.execute("SELECT COUNT(*) FROM points").fetchone()[0]


@pytest.fixture
def node_tmp(tmp_path, monkeypatch):
    """Point tempfile.gettempdir() inside tmp_path, so the PR's temp files stay hermetic."""
    d = tmp_path / "node_tmp"
    d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    return d


@bug(L2)
def test_L2_partial_job_not_committed(m, tmp_path):
    """Blocker 1: transaction() commits in `finally`, so a non-sqlite error commits half an isotherm."""
    db = str(tmp_path / "results.db")
    store = m.ResultsStore(db)
    truncated = ISOTHERM[:2] + [{"mof_id": "MOF-5", "pressure_bar": 30.0}]  # parser lost two fields
    with pytest.raises(KeyError):
        store.add_job("job-17", truncated)
    assert _count(db) == 0


@bug(L2)
def test_L2_failed_write_keeps_old_snapshot(m, tmp_path, node_tmp):
    """Blocker 2: atomic_write renames in `finally`, so a failed write replaces the last good snapshot."""
    target = tmp_path / "snapshot.jsonl"
    target.write_text('{"mof_id": "MOF-5"}\n')
    with pytest.raises(RuntimeError):
        with m.atomic_write(str(target)) as f:
            f.write('{"mof_id": "MOF-')
            raise RuntimeError("export failed halfway")
    assert target.read_text() == '{"mof_id": "MOF-5"}\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["node_tmp", "snapshot.jsonl"]


@bug(L2, raises=TypeError)
def test_L2_export_with_failed_points(m, tmp_path, node_tmp):
    """Major 3: round(None) crashes the export as soon as any GCMC point has failed."""
    store = m.ResultsStore(str(tmp_path / "results.db"))
    store.add_job("job-17", ISOTHERM[:2] + [_point("MOF-5", 30.0, None, status="failed")])
    out = tmp_path / "snapshot.jsonl"
    assert store.export_snapshot(str(out)) == 3
    last = json.loads(out.read_text().splitlines()[-1])
    assert last["uptake_mmol_g"] is None and last["status"] == "failed"


def _same_directory_only(real):
    """os.rename/os.replace stand-in: a move between directories fails, as it does across filesystems."""

    def move(src, dst, *args, **kwargs):
        if os.path.dirname(os.path.abspath(src)) != os.path.dirname(os.path.abspath(dst)):
            raise OSError(errno.EXDEV, "Invalid cross-device link", str(src), None, str(dst))
        return real(src, dst, *args, **kwargs)

    return move


@bug(L2, raises=OSError)
def test_L2_temp_file_next_to_target(m, tmp_path, node_tmp, monkeypatch):
    """Major 4: the temp file lives in tempfile.gettempdir(), so the rename crosses filesystems."""
    monkeypatch.setattr(os, "rename", _same_directory_only(os.rename))
    monkeypatch.setattr(os, "replace", _same_directory_only(os.replace))
    target = tmp_path / "shared" / "snapshot.jsonl"
    target.parent.mkdir()
    with m.atomic_write(str(target)) as f:
        f.write("{}\n")
    assert target.read_text() == "{}\n"


@bug(L2, raises=FileNotFoundError)
def test_L2_concurrent_exports_dont_clobber(m, tmp_path, node_tmp):
    """Major 5: a fixed temp name means two exports of files with the same name clobber each other."""
    a = tmp_path / "campaign_a" / "snapshot.jsonl"
    b = tmp_path / "campaign_b" / "snapshot.jsonl"
    a.parent.mkdir()
    b.parent.mkdir()
    with m.atomic_write(str(a)) as fa:
        fa.write('{"campaign": "a"}\n')
        with m.atomic_write(str(b)) as fb:  # a second exporter starts while the first is mid-write
            fb.write('{"campaign": "b"}\n')
    assert a.read_text() == '{"campaign": "a"}\n'
    assert b.read_text() == '{"campaign": "b"}\n'


@bug(L2)
def test_L2_connections_closed(m, tmp_path, node_tmp, monkeypatch):
    """Minor 6: `with sqlite3.connect(...) as conn` commits or rolls back, but never closes."""
    opened = []
    monkeypatch.setattr(sqlite3, "connect", _tracking(sqlite3.connect, opened))
    store = m.ResultsStore(str(tmp_path / "results.db"))
    store.add_job("job-17", ISOTHERM)
    store.export_snapshot(str(tmp_path / "snapshot.jsonl"))
    still_open = []
    for conn in opened:
        try:
            conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:  # "Cannot operate on a closed database."
            continue
        still_open.append(conn)
    for conn in still_open:
        conn.close()
    assert opened and not still_open


@bug(L2)
def test_L2_retried_job_is_idempotent(m, tmp_path):
    """Minor 9: a job retried after a successful commit inserts its isotherm a second time."""
    db = str(tmp_path / "results.db")
    store = m.ResultsStore(db)
    store.add_job("job-17", ISOTHERM)
    store.add_job("job-17", ISOTHERM)
    assert _count(db) == len(ISOTHERM)


@fine(L2)
def test_L2_export_holds_no_lock_while_writing(m, tmp_path, node_tmp, monkeypatch):
    """Red herring: fetchall() before writing means no read lock is held during the slow file write."""
    db = str(tmp_path / "results.db")
    store = m.ResultsStore(db)
    store.add_job("job-17", ISOTHERM)
    worker_committed = []
    real_dumps = json.dumps

    def dumps_while_a_worker_writes(obj, *args, **kwargs):
        if not worker_committed:
            with closing(sqlite3.connect(db, timeout=0)) as worker:
                worker.execute("BEGIN EXCLUSIVE")  # raises "database is locked" if the export holds a lock
                worker.rollback()
            worker_committed.append(True)
        return real_dumps(obj, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", dumps_while_a_worker_writes)
    store.export_snapshot(str(tmp_path / "snapshot.jsonl"))
    assert worker_committed


# --- L3: SimulationWorkspace -------------------------------------------------------------------

L3 = "03_context_managers/L3_simulation_workspace"

INPUTS = {"in.json": '{"mof": "MOF-5"}'}
# Stand-in for a simulation binary: echoes its input and its thread count into results.json.
FAKE_SIM = [
    sys.executable,
    "-c",
    "import json, os; mof = json.load(open('in.json'))['mof']; "
    "json.dump({'mof': mof, 'omp': os.environ.get('OMP_NUM_THREADS')}, open('results.json', 'w'))",
]
FAILING_SIM = [sys.executable, "-c", "print('SCF did not converge after 60 steps'); raise SystemExit(3)"]


@pytest.fixture
def scratch(m, tmp_path, monkeypatch):
    """Hermetic scratch and lock roots; cwd and OMP_NUM_THREADS are restored after the test."""
    monkeypatch.setattr(m, "SCRATCH_ROOT", tmp_path / "scratch")
    monkeypatch.setattr(m, "LOCK_DIR", tmp_path / "locks")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")  # records the original value so teardown restores it
    monkeypatch.delenv("OMP_NUM_THREADS")
    return tmp_path / "scratch"


@bug(L3)
def test_L3_concurrent_workspaces_isolated(m, scratch):
    """Blocker 1: os.chdir and os.environ are process-wide, so concurrent jobs read each other's files."""
    a = m.SimulationWorkspace("mof-a", {"in.json": '{"mof": "A"}'}, threads=2)
    b = m.SimulationWorkspace("mof-b", {"in.json": '{"mof": "B"}'}, threads=8)
    with a:
        with b:  # what the thread pool does: job B starts while job A is still running
            result = a.run(FAKE_SIM)
    assert (result["mof"], result["omp"]) == ("A", "2")


@bug(L3)
def test_L3_losing_workspace_leaves_running_job_alone(m, scratch):
    """Blocker 2: __del__ of a workspace that lost the lock deletes the running job's directory."""
    running = m.SimulationWorkspace("mof-5", INPUTS, keep=True)
    with running:
        loser = m.SimulationWorkspace("mof-5", INPUTS)
        try:
            with loser:
                pass
        except m.WorkspaceBusy:
            pass
        else:
            raise AssertionError("a second workspace entered a job that is already running")
        del loser  # CPython's refcounting runs __del__ right here
        assert (running.path / "in.json").exists()


@bug(L3)
def test_L3_existing_directory_survives(m, scratch):
    """Blocker 3: the workspace adopts an existing directory (exist_ok=True) and rmtree()s it on exit."""
    old = scratch / "mof-5"
    old.mkdir(parents=True)
    (old / "isotherm_2024.csv").write_text("pressure_bar,uptake_mmol_g\n1.0,2.1\n")
    with m.SimulationWorkspace("mof-5", INPUTS):
        pass
    assert (old / "isotherm_2024.csv").exists()


@bug(L3)
def test_L3_lock_is_atomic(m, scratch, monkeypatch):
    """Major 4: exists()-then-write isn't atomic, so two workers can both take the lock."""
    real_exists = Path.exists

    def lock_not_there_yet(self, *args, **kwargs):  # both workers check before either writes
        return False if self.suffix == ".lock" else real_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", lock_not_there_yet)
    first = m.SimulationWorkspace("mof-5", INPUTS, keep=True)
    second = m.SimulationWorkspace("mof-5", INPUTS, keep=True)
    with first:
        try:
            second.__enter__()
        except m.WorkspaceBusy:
            return
    raise AssertionError("two workers entered the same job")


@bug(L3)
def test_L3_stale_lock_after_crash(m, scratch):
    """Major 5: a lock file left by a killed worker blocks the job forever."""
    m.LOCK_DIR.mkdir(parents=True)
    (m.LOCK_DIR / "mof-5.lock").write_text("48213")  # PID of a worker that was OOM-killed
    try:
        with m.SimulationWorkspace("mof-5", INPUTS):
            pass
    except m.WorkspaceBusy:
        raise AssertionError("a crashed worker's lock file blocks the job forever") from None


@bug(L3)
def test_L3_failed_setup_releases_lock(m, scratch):
    """Major 6: if __enter__ fails halfway, __exit__ never runs and the lock is never released."""
    with pytest.raises(TypeError):
        with m.SimulationWorkspace("mof-5", {"in.json": "{}", "restart.bin": b"\x00\x01"}):
            pass
    try:
        with m.SimulationWorkspace("mof-5", INPUTS):
            pass
    except m.WorkspaceBusy:
        raise AssertionError("the failed setup left the job locked") from None


@bug(L3, raises=OSError)
def test_L3_cleanup_errors_dont_mask_or_leak(m, scratch, monkeypatch):
    """Major 7: an rmtree failure in __exit__ hides the job's exception and leaves the lock behind."""

    def rmtree_busy(path, *args, **kwargs):  # e.g. .nfsXXXX files held open on a shared filesystem
        raise OSError(errno.ENOTEMPTY, "Directory not empty", str(path))

    monkeypatch.setattr(m.shutil, "rmtree", rmtree_busy)
    with pytest.raises(ValueError):
        with m.SimulationWorkspace("mof-5", INPUTS):
            raise ValueError("bad structure")
    with m.SimulationWorkspace("mof-6", INPUTS):  # success path: the cleanup failure is only logged
        pass
    with m.SimulationWorkspace("mof-6", INPUTS):  # ...and the lock was released
        pass


@bug(L3)
def test_L3_env_restored(m, scratch):
    """Minor 9: OMP_NUM_THREADS stays set after the job if it wasn't set before."""
    with m.SimulationWorkspace("mof-5", INPUTS, threads=4):
        pass
    assert "OMP_NUM_THREADS" not in os.environ


@bug(L3)
def test_L3_failed_job_keeps_its_directory(m, scratch):
    """Minor 10: a failed job's scratch directory, logs included, is deleted on exit."""
    ws = m.SimulationWorkspace("mof-5", INPUTS)
    with pytest.raises(RuntimeError):
        with ws:
            ws.run(FAILING_SIM)
    log_file = ws.path / "stdout.log"
    assert log_file.exists() and "SCF did not converge" in log_file.read_text()


@fine(L3)
def test_L3_exceptions_propagate(m, scratch):
    """Red herring: __exit__ returns None, so the job's own exception propagates, as it should."""
    with pytest.raises(KeyError):
        with m.SimulationWorkspace("mof-5", INPUTS):
            raise KeyError("uptake_mmol_g")
