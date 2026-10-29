"""
PR #243: Streaming ingest of GCMC results into the results database

GCMC (grand canonical Monte Carlo) simulations predict how much gas a porous material
adsorbs at a given pressure. Our workers append one JSON line to their shard file every
time a pressure point finishes; each worker runs several MOFs at once, and a failed job
is retried, possibly on another worker. The first line of every shard is a header with
the schema version. The old nightly pandas script ran out of memory at ~50M points, so
this replaces it with a lazy pipeline: parse -> keep converged points -> drop duplicate
points (from retries) -> write points to SQLite in batches -> refresh the per-MOF summary
(best uptake, number of points) that the screening dashboard ranks candidates by.

    stats = ingest(sorted(glob("/shared/gcmc/shards/*.jsonl")), sqlite3.connect("results.db"))

A point looks like {"mof_id": "MOF-5", "temperature_k": 298, "pressure_bar": 1.0,
"uptake_mmol_g": 2.13, "converged": true}.
"""
import itertools
import json
import logging
import sqlite3
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


def parse_shard(path: str) -> Iterator[dict]:
    """Yield the points in one shard after checking its header."""
    with open(path) as f:
        header = json.loads(next(f))
        if header.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema {header.get('schema')}, expected {SCHEMA_VERSION}")
        for line in f:
            point = json.loads(line)
            if point["uptake_mmol_g"] < 0 or point["pressure_bar"] <= 0:
                raise BadRecord(
                    f"{point['mof_id']}: uptake {point['uptake_mmol_g']} at {point['pressure_bar']} bar"
                )
            yield point


def read_shards(paths: Iterable[str]) -> Iterator[dict]:
    """All points from all shards, skipping (and logging) bad records."""
    for path in paths:
        points = parse_shard(path)
        while True:
            try:
                point = next(points)
            except StopIteration:
                break
            except BadRecord as e:
                log.warning("%s: skipping bad record: %s", path, e)
                continue
            yield point


def dedupe(points: Iterable[dict]) -> Iterator[dict]:
    seen = set()
    for p in points:
        key = (p["mof_id"], p["temperature_k"], p["pressure_bar"])
        if key not in seen:
            seen.add(key)
            yield p


def summarise(points: Iterable[dict]) -> Iterator[tuple[str, float, int]]:
    """(mof_id, best uptake, number of points) for each MOF."""
    for mof_id, group in itertools.groupby(points, key=lambda p: p["mof_id"]):
        uptakes = [p["uptake_mmol_g"] for p in group]
        yield mof_id, max(uptakes), len(uptakes)


def point_writer(conn: sqlite3.Connection, batch_size: int = BATCH_SIZE):
    """Coroutine: send() points in; they are written in batches of `batch_size`."""
    batch = []
    while True:
        p = yield
        batch.append((p["mof_id"], p["temperature_k"], p["pressure_bar"], p["uptake_mmol_g"]))
        if len(batch) >= batch_size:
            conn.executemany("INSERT OR REPLACE INTO points VALUES (?, ?, ?, ?)", batch)
            conn.commit()
            batch = []


def ingest(paths: Iterable[str], conn: sqlite3.Connection, batch_size: int = BATCH_SIZE) -> dict[str, int]:
    """Load converged points from the shards into `points` and refresh `mof_summary`."""
    conn.executescript(SCHEMA)
    points = dedupe(p for p in read_shards(paths) if p["converged"])
    to_db, to_summary = itertools.tee(points)

    writer = point_writer(conn, batch_size)
    next(writer)
    n_points = 0
    for p in to_db:
        writer.send(p)
        n_points += 1
    writer.close()

    n_mofs = 0
    for mof_id, best, n in summarise(to_summary):
        conn.execute("INSERT OR REPLACE INTO mof_summary VALUES (?, ?, ?)", (mof_id, best, n))
        n_mofs += 1
    conn.commit()
    log.info("ingested %d points for %d MOFs", n_points, n_mofs)
    return {"points": n_points, "mofs": n_mofs}
