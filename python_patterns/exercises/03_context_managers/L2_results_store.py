"""
PR #232: Transactional writes and atomic snapshots for the results store

GCMC workers write isotherm points into a shared SQLite database, and the screening
dashboard reads a JSON Lines snapshot that we export every few minutes. (An isotherm is
the gas uptake of one MOF measured at several pressures; one GCMC job produces one.) Two
problems keep coming up: a worker that fails halfway through a job leaves half an isotherm
in the database, and the scheduler then retries the job, so we get duplicates; and the
dashboard sometimes reads a half-written snapshot. This adds a `transaction()` context
manager and an `atomic_write()` helper, and uses both in `ResultsStore`.
"""
import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Commit the block's writes together; roll back if the database reports an error."""
    try:
        yield conn
    except sqlite3.Error:
        conn.rollback()
        raise
    finally:
        conn.commit()


@contextmanager
def atomic_write(path: str):
    """Write to a temporary file, then move it over `path`, so readers never see a partial file."""
    tmp = os.path.join(tempfile.gettempdir(), os.path.basename(path) + ".tmp")
    f = open(tmp, "w")
    try:
        yield f
    finally:
        f.close()
        os.rename(tmp, path)


class ResultsStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS points ("
                "job_id TEXT, mof_id TEXT, pressure_bar REAL, uptake_mmol_g REAL, status TEXT)"
            )

    def add_job(self, job_id: str, points: list[dict]) -> None:
        """Store every point of one finished GCMC job (failed points have uptake None)."""
        with sqlite3.connect(self.db_path) as conn, transaction(conn):
            for p in points:
                conn.execute(
                    "INSERT INTO points VALUES (?, ?, ?, ?, ?)",
                    (job_id, p["mof_id"], p["pressure_bar"], p["uptake_mmol_g"], p["status"]),
                )

    def export_snapshot(self, path: str) -> int:
        """Write every point as JSON Lines for the dashboard; returns the number of rows."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT mof_id, pressure_bar, uptake_mmol_g, status FROM points "
                "ORDER BY mof_id, pressure_bar"
            ).fetchall()
        with atomic_write(path) as f:
            for mof_id, pressure, uptake, status in rows:
                record = {
                    "mof_id": mof_id,
                    "pressure_bar": pressure,
                    "uptake_mmol_g": round(uptake, 3),
                    "status": status,
                }
                f.write(json.dumps(record) + "\n")
        return len(rows)
