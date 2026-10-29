"""Reference fix for PR #232 (python_patterns/exercises/03_context_managers/L2_results_store.py).
See the answer key for the reasoning."""
import json
import os
import sqlite3
import tempfile
from contextlib import closing, contextmanager


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Commit the block's writes together, or none of them."""
    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()


@contextmanager
def atomic_write(path: str):
    """Write to a temporary file, then move it over `path`, so readers never see a partial file."""
    directory = os.path.dirname(os.path.abspath(path))
    # Same directory (so same filesystem) and a unique name per writer.
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; the dashboard runs as another user
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


class ResultsStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        # `with conn` only commits or rolls back; closing() is what closes the connection.
        with closing(sqlite3.connect(db_path)) as conn, conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS points ("
                "job_id TEXT, mof_id TEXT, pressure_bar REAL, uptake_mmol_g REAL, status TEXT, "
                "PRIMARY KEY (job_id, mof_id, pressure_bar))"
            )

    def add_job(self, job_id: str, points: list[dict]) -> None:
        """Store every point of one finished GCMC job (failed points have uptake None)."""
        with closing(sqlite3.connect(self.db_path)) as conn, transaction(conn):
            for p in points:
                conn.execute(
                    "INSERT OR REPLACE INTO points VALUES (?, ?, ?, ?, ?)",
                    (job_id, p["mof_id"], p["pressure_bar"], p["uptake_mmol_g"], p["status"]),
                )

    def export_snapshot(self, path: str) -> int:
        """Write every point as JSON Lines for the dashboard; returns the number of rows."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT mof_id, pressure_bar, uptake_mmol_g, status FROM points "
                "ORDER BY mof_id, pressure_bar"
            ).fetchall()
        with atomic_write(path) as f:
            for mof_id, pressure, uptake, status in rows:
                record = {
                    "mof_id": mof_id,
                    "pressure_bar": pressure,
                    "uptake_mmol_g": None if uptake is None else round(uptake, 3),
                    "status": status,
                }
                f.write(json.dumps(record) + "\n")
        return len(rows)
