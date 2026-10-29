"""
PR #292: Keep one SQLite connection for the GCMC results store

The runner writes one row for every finished GCMC job (a grand canonical Monte
Carlo adsorption simulation). Opening a connection and committing for every row
was slow on the shared filesystem, so this keeps one module-level connection,
opened lazily on first use, and commits every 200 rows. The runner calls
`get_db(path)` at start-up, `record_result` from its worker threads and
`close_db()` at exit. Analysis notebooks call `get_db(path)` for the campaign
they want to look at, then `best_candidates()`.
"""
import atexit
import sqlite3
import threading

DEFAULT_DB = "gcmc_results.sqlite"
COMMIT_EVERY = 200

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
_pending = 0
_lock = threading.Lock()


def get_db(path: str = DEFAULT_DB) -> sqlite3.Connection:
    """Return the shared connection, opening it on first use."""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(path, check_same_thread=False)
        _conn.execute(SCHEMA)
    return _conn


def record_result(
    job_id: str, mof_id: str, temperature_k: float, pressure_bar: float, uptake_mmol_g: float
) -> None:
    global _pending
    conn = get_db()
    with _lock:
        try:
            conn.execute(
                "INSERT INTO results VALUES (?, ?, ?, ?, ?)",
                (job_id, mof_id, temperature_k, pressure_bar, uptake_mmol_g),
            )
        except sqlite3.IntegrityError:
            conn.rollback()
            raise
        _pending += 1
        if _pending >= COMMIT_EVERY:
            conn.commit()
            _pending = 0


def best_candidates(
    temperature_k: float, pressure_bar: float, n: int = 20
) -> list[tuple[str, float]]:
    """The n MOFs with the highest uptake at the given conditions."""
    rows = get_db().execute(
        "SELECT mof_id, uptake_mmol_g FROM results "
        "WHERE temperature_k = ? AND pressure_bar = ? "
        "ORDER BY uptake_mmol_g DESC LIMIT ?",
        (temperature_k, pressure_bar, n),
    )
    return rows.fetchall()


def close_db() -> None:
    if _conn is not None:
        _conn.close()


atexit.register(close_db)
