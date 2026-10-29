"""
Reference fix for PR #292 (python_patterns/exercises/09_singleton_di/L2_results_db.py).
See the answer key for the reasoning.
"""
import atexit
import os
import sqlite3
import threading

DEFAULT_DB = "gcmc_results.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    job_id        TEXT PRIMARY KEY,
    mof_id        TEXT NOT NULL,
    temperature_k REAL NOT NULL,
    pressure_bar  REAL NOT NULL,
    uptake_mmol_g REAL NOT NULL
)
"""

_conn: sqlite3.Connection | None = None
_conn_path: str | None = None
_lock = threading.Lock()


def get_db(path: str | None = None) -> sqlite3.Connection:
    """Return the shared connection, opening it on first use.

    Asking for a different path while a connection is open is an error, not a
    silent reuse of the old database: call close_db() first.
    """
    global _conn, _conn_path
    wanted = os.path.abspath(path) if path is not None else None
    with _lock:  # the None check and the connect must be one atomic step
        if _conn is None:
            _conn_path = wanted or os.path.abspath(DEFAULT_DB)
            _conn = sqlite3.connect(_conn_path, check_same_thread=False)
            _conn.execute(SCHEMA)
        elif wanted is not None and wanted != _conn_path:
            raise ValueError(f"results DB already open at {_conn_path}; call close_db() first")
        return _conn


def record_result(
    job_id: str,
    mof_id: str,
    temperature_k: float,
    pressure_bar: float,
    uptake_mmol_g: float,
    conn: sqlite3.Connection | None = None,
) -> None:
    if conn is None:
        conn = get_db()
    # One short transaction per result. A failed insert can only roll back itself,
    # nothing is left uncommitted at exit, and SQLite's write lock is released at once.
    with _lock, conn:
        conn.execute(
            "INSERT INTO results VALUES (?, ?, ?, ?, ?)",
            (job_id, mof_id, temperature_k, pressure_bar, uptake_mmol_g),
        )


def best_candidates(
    temperature_k: float, pressure_bar: float, n: int = 20, conn: sqlite3.Connection | None = None
) -> list[tuple[str, float]]:
    """The n MOFs with the highest uptake at the given conditions."""
    if conn is None:
        conn = get_db()
    with _lock:
        rows = conn.execute(
            "SELECT mof_id, uptake_mmol_g FROM results "
            "WHERE temperature_k = ? AND pressure_bar = ? "
            "ORDER BY uptake_mmol_g DESC LIMIT ?",
            (temperature_k, pressure_bar, n),
        )
        return rows.fetchall()


def close_db() -> None:
    global _conn, _conn_path
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = _conn_path = None


atexit.register(close_db)
