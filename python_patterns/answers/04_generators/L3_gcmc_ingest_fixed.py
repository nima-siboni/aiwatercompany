"""Reference fix for PR #243 (python_patterns/exercises/04_generators/L3_gcmc_ingest.py).
See the answer key for the reasoning."""
import itertools
import json
import logging
import sqlite3
from collections import Counter
from typing import Iterable, Iterator

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2
BATCH_SIZE = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS points (
    mof_id TEXT, temperature_k REAL, pressure_bar REAL, uptake_mmol_g REAL,
    PRIMARY KEY (mof_id, temperature_k, pressure_bar)
);
CREATE TABLE IF NOT EXISTS mof_summary (
    mof_id TEXT PRIMARY KEY, best_uptake_mmol_g REAL, n_points INTEGER
);
"""


class BadRecord(ValueError):
    """A point that parsed but has physically impossible values."""


def _check_header(line: str, path: str) -> None:
    header = json.loads(line)
    if header.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema {header.get('schema')}, expected {SCHEMA_VERSION}")


def check_headers(paths: Iterable[str]) -> None:
    """Fail before anything is written if any shard has the wrong schema."""
    for path in paths:
        with open(path) as f:
            first = next(f, None)
        if first is not None:
            _check_header(first, path)


def _validated(point: dict) -> dict:
    if point["uptake_mmol_g"] < 0 or point["pressure_bar"] <= 0:
        raise BadRecord(
            f"{point['mof_id']}: uptake {point['uptake_mmol_g']} at {point['pressure_bar']} bar"
        )
    return point


def parse_shard(path: str, stats: Counter | None = None) -> Iterator[dict]:
    """Yield the valid points in one shard; skip (log and count) the lines that aren't."""
    with open(path) as f:
        first = next(f, None)  # never a bare next() inside a generator (PEP 479)
        if first is None:  # the worker died before writing its header
            return
        _check_header(first, path)
        for lineno, line in enumerate(f, start=2):
            try:
                point = _validated(json.loads(line))
            except (json.JSONDecodeError, BadRecord) as e:  # handled here, so the generator lives on
                log.warning("%s:%d: skipping record: %s", path, lineno, e)
                if stats is not None:
                    stats["skipped"] += 1
                continue
            yield point


def read_shards(paths: Iterable[str], stats: Counter | None = None) -> Iterator[dict]:
    """All valid points from all shards."""
    for path in paths:
        yield from parse_shard(path, stats)


def write_points(conn: sqlite3.Connection, points: Iterable[dict], batch_size: int = BATCH_SIZE) -> int:
    """Insert points in batches; returns how many were written."""
    it = iter(points)
    n = 0
    while batch := list(itertools.islice(it, batch_size)):  # the last, short batch included
        conn.executemany(
            "INSERT OR REPLACE INTO points VALUES (?, ?, ?, ?)",
            [(p["mof_id"], p["temperature_k"], p["pressure_bar"], p["uptake_mmol_g"]) for p in batch],
        )
        conn.commit()
        n += len(batch)
    return n


def ingest(paths: Iterable[str], conn: sqlite3.Connection, batch_size: int = BATCH_SIZE) -> dict[str, int]:
    """Load converged points from the shards into `points` and refresh `mof_summary`."""
    paths = list(paths)
    check_headers(paths)
    conn.executescript(SCHEMA)
    stats = Counter()
    # The primary key dedupes retried points, so no in-memory `seen` set is needed.
    n_points = write_points(conn, (p for p in read_shards(paths, stats) if p["converged"]), batch_size)
    # Aggregate in the database: it holds every point, from any shard, in any order, from any night.
    n_mofs = conn.execute(
        "INSERT OR REPLACE INTO mof_summary "
        "SELECT mof_id, MAX(uptake_mmol_g), COUNT(*) FROM points GROUP BY mof_id"
    ).rowcount
    conn.commit()
    log.info("ingested %d points; %d MOFs summarised; %d records skipped", n_points, n_mofs, stats["skipped"])
    return {"points": n_points, "mofs": n_mofs, "skipped": stats["skipped"]}
