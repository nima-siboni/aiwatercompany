"""Checks for the 04_generators drills: every planted bug fails on the PR code and passes on the fix."""
import gzip
import json
import random
import sqlite3

import pytest

from harness import bug, fine


def _jsonl(path, records, tail=""):
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + tail)
    return path


# --- L1: batched() and predict_uptake() for the surrogate model --------------------------------

L1 = "04_generators/L1_batched_inference"

MOFS = [{"mof_id": f"MOF-{i}", "metal": "Zn"} for i in range(4)]


def fake_model(batch):
    """Stand-in surrogate: 'predicts' an uptake derived from the id, one per structure, in order."""
    return [float(s["mof_id"].split("-")[1]) + 0.5 for s in batch]


@bug(L1)
def test_L1_generator_input_consumed_once(m):
    """Blocker 1: predict_uptake iterates `structures` twice, so a generator gives {}."""
    preds = m.predict_uptake(fake_model, (s for s in MOFS), batch_size=2)
    assert preds == {"MOF-0": 0.5, "MOF-1": 1.5, "MOF-2": 2.5, "MOF-3": 3.5}


@bug(L1)
def test_L1_last_partial_batch_kept(m):
    """Blocker 2: batched() never yields the final partial batch."""
    assert list(m.batched(range(5), 2)) == [[0, 1], [2, 3], [4]]


@bug(L1)
def test_L1_length_mismatch_is_loud(m):
    """Major 3: zip() silently truncates, so a short model response shifts predictions onto the wrong MOFs."""

    def model_that_skips_one(batch):  # e.g. a featuriser that drops a structure it can't handle
        return fake_model([s for s in batch if s["mof_id"] != "MOF-1"])

    try:
        preds = m.predict_uptake(model_that_skips_one, MOFS[:3], batch_size=3)
    except ValueError:
        return
    raise AssertionError(f"a short model response went unnoticed: {preds}")


@bug(L1, raises=ValueError)
def test_L1_blank_lines_and_error_location(m, tmp_path):
    """Minor 4: a blank line kills the run, and a bad line's error doesn't say where it is."""
    path = _jsonl(tmp_path / "gen_run_42.jsonl", MOFS[:2], tail="\n")
    assert [s["mof_id"] for s in m.read_structures(str(path))] == ["MOF-0", "MOF-1"]
    path.write_text(path.read_text() + '{"mof_id": "MOF-9", "met\n')
    with pytest.raises(ValueError, match=r"gen_run_42\.jsonl:4"):
        list(m.read_structures(str(path)))


@bug(L1)
def test_L1_batch_size_validated(m):
    """Minor 5: size=0 silently consumes the input and yields nothing."""
    try:
        out = list(m.batched(iter(MOFS), 0))
    except ValueError:
        return
    raise AssertionError(f"batched(..., 0) silently returned {out}")


@fine(L1)
def test_L1_batches_are_independent_lists(m):
    """Red herring: rebinding `batch = []` (not batch.clear()) keeps already-yielded batches intact."""
    batches = list(m.batched(range(4), 2))
    assert batches == [[0, 1], [2, 3]]
    assert batches[0] is not batches[1]


# --- L2: RelaxationDataset, a streaming dataset over JSONL shards -------------------------------

L2 = "04_generators/L2_streaming_dataset"


def _relaxation(i, n_atoms=4):
    return {
        "mof_id": f"MOF-{i}",
        "structure": {"lattice": [10.0, 10.0, 10.0]},
        "energy": -5.0 * n_atoms,
        "n_atoms": n_atoms,
    }


def _shards(tmp_path, n_shards=2, per_shard=3, tail=""):
    """JSONL shards with distinct MOF ids; `tail` is appended to the last shard."""
    paths = []
    for k in range(n_shards):
        recs = [_relaxation(k * per_shard + i) for i in range(per_shard)]
        paths.append(str(_jsonl(tmp_path / f"shard_{k:03d}.jsonl", recs, tail if k == n_shards - 1 else "")))
    return paths


@bug(L2)
def test_L2_every_epoch_sees_all_data(m, tmp_path):
    """Blocker 1: __iter__ returns self, so after one pass (mean_energy) every later epoch is empty."""
    ds = m.RelaxationDataset(_shards(tmp_path), buffer_size=1)
    assert m.mean_energy(ds) == pytest.approx(-5.0)
    epoch_sizes = [sum(1 for _ in ds) for _ in range(2)]
    assert epoch_sizes == [6, 6]


@bug(L2)
def test_L2_shuffle_buffer_flushes_tail(m, tmp_path):
    """Blocker 2: the shuffle buffer never flushes, so the last buffer_size - 1 records vanish."""
    ds = m.RelaxationDataset(_shards(tmp_path), buffer_size=10)
    ids = [r["mof_id"] for r in ds]
    assert sorted(ids) == sorted(f"MOF-{i}" for i in range(6))


@bug(L2)
def test_L2_schema_change_is_loud(m, tmp_path):
    """Blocker 3: `except Exception: continue` silently drops every record of a renamed field."""
    renamed = [{**_relaxation(i), "energy_eV": -20.0} for i in range(3)]
    for r in renamed:
        del r["energy"]
    path = _jsonl(tmp_path / "shard_new.jsonl", renamed)
    ds = m.RelaxationDataset([str(path)], buffer_size=1)
    try:
        records = list(ds)
    except ValueError:
        return
    raise AssertionError(f"records with a renamed field vanished silently ({len(records)} of 3 yielded)")


@bug(L2, raises=ValueError)
def test_L2_gzip_shards_readable(m, tmp_path):
    """Major 4: open_shard returns a generator over a gzip file that `with` has already closed."""
    path = tmp_path / "shard_007.jsonl.gz"
    with gzip.open(path, "wt") as f:
        f.writelines(json.dumps(_relaxation(i)) + "\n" for i in range(3))
    ds = m.RelaxationDataset([str(path)], buffer_size=1)
    assert sorted(r["mof_id"] for r in ds) == ["MOF-0", "MOF-1", "MOF-2"]


@bug(L2)
def test_L2_len_does_not_rescan(m, tmp_path, monkeypatch):
    """Major 5: __len__ rereads every shard on every call."""
    opened = []

    def counting_open(*args, **kwargs):
        opened.append(args[0])
        return open(*args, **kwargs)

    monkeypatch.setattr(m, "open", counting_open, raising=False)
    paths = _shards(tmp_path)
    ds = m.RelaxationDataset(paths, buffer_size=1)
    assert len(ds) == 6
    assert len(ds) == 6
    assert len(opened) == len(paths)


@bug(L2)
def test_L2_len_counts_records(m, tmp_path):
    """Minor 7: __len__ counts lines, including the ones iteration skips."""
    paths = _shards(tmp_path, tail='{"mof_id": "MOF-99", "struct')  # killed mid-write
    ds = m.RelaxationDataset(paths, buffer_size=1)
    assert len(ds) == sum(1 for _ in ds) == 6


@fine(L2)
def test_L2_private_rng_is_reproducible(m, tmp_path):
    """Red herring: a private random.Random(seed) gives the same order whatever else uses `random`."""
    paths = _shards(tmp_path, n_shards=3, per_shard=4)
    first = [r["mof_id"] for r in m.RelaxationDataset(paths, buffer_size=3, seed=7)]
    random.seed(12345)
    random.random()
    second = [r["mof_id"] for r in m.RelaxationDataset(paths, buffer_size=3, seed=7)]
    assert first == second


# --- L3: streaming GCMC ingest -----------------------------------------------------------------

L3 = "04_generators/L3_gcmc_ingest"

HEADER = {"schema": 2, "worker": "node17-3"}


def _pt(mof_id, pressure, uptake, converged=True):
    return {
        "mof_id": mof_id,
        "temperature_k": 298,
        "pressure_bar": pressure,
        "uptake_mmol_g": uptake,
        "converged": converged,
    }


def _shard(path, points, header=HEADER, tail=""):
    return str(_jsonl(path, [header] + points, tail))


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


def _rows(conn, sql):
    return conn.execute(sql).fetchall()


@bug(L3)
def test_L3_summary_with_interleaved_points(m, tmp_path, db):
    """Blocker 1: groupby on points that aren't sorted by MOF splits each MOF into fragments."""
    # A worker runs MOF-A and MOF-B at once, so their points arrive interleaved.
    points = [_pt("MOF-A", 10.0, 3.0), _pt("MOF-B", 1.0, 0.5), _pt("MOF-A", 1.0, 1.0)]
    m.ingest([_shard(tmp_path / "w1.jsonl", points)], db)
    assert _rows(db, "SELECT * FROM mof_summary ORDER BY mof_id") == [("MOF-A", 3.0, 2), ("MOF-B", 0.5, 1)]


@bug(L3)
def test_L3_last_batch_written(m, tmp_path, db):
    """Blocker 2: closing the point_writer coroutine drops the final partial batch."""
    points = [_pt("MOF-A", p, 1.0 + p) for p in (1.0, 5.0, 10.0)]
    m.ingest([_shard(tmp_path / "w1.jsonl", points)], db, batch_size=2)
    assert _rows(db, "SELECT COUNT(*) FROM points") == [(3,)]


@bug(L3)
def test_L3_bad_record_skips_only_itself(m, tmp_path):
    """Blocker 3: a BadRecord finishes parse_shard, so read_shards silently drops the rest of the shard."""
    points = [
        _pt("MOF-A", 1.0, 1.0),
        _pt("MOF-A", 5.0, -0.2, converged=False),  # garbage from a run that didn't converge
        _pt("MOF-A", 10.0, 3.0),
        _pt("MOF-B", 1.0, 0.5),
    ]
    got = [(p["mof_id"], p["pressure_bar"]) for p in m.read_shards([_shard(tmp_path / "w1.jsonl", points)])]
    assert got == [("MOF-A", 1.0), ("MOF-A", 10.0), ("MOF-B", 1.0)]


@bug(L3, raises=RuntimeError)
def test_L3_empty_shard(m, tmp_path):
    """Major 4: next(f) on an empty shard raises StopIteration inside a generator (PEP 479: RuntimeError)."""
    empty = tmp_path / "w2.jsonl"
    empty.write_text("")  # the worker died before writing its header
    shard = _shard(tmp_path / "w1.jsonl", [_pt("MOF-A", 1.0, 1.0)])
    assert [p["mof_id"] for p in m.read_shards([str(empty), shard])] == ["MOF-A"]


@bug(L3, raises=ValueError)
def test_L3_truncated_line(m, tmp_path):
    """Major 5: a truncated last line raises JSONDecodeError out of the pipeline and aborts the ingest."""
    points = [_pt("MOF-A", 1.0, 1.0), _pt("MOF-A", 5.0, 2.0)]
    shard = _shard(tmp_path / "w1.jsonl", points, tail='{"mof_id": "MOF-A", "temperat')  # killed mid-write
    assert len(list(m.read_shards([shard]))) == 2


@bug(L3)
def test_L3_schema_error_before_any_write(m, tmp_path, db):
    """Minor 8: the schema check runs lazily, so earlier shards are already written when it fails."""
    good = _shard(tmp_path / "w1.jsonl", [_pt("MOF-A", p, 1.0) for p in (1.0, 5.0, 10.0)])
    old = _shard(tmp_path / "w2.jsonl", [_pt("MOF-B", 1.0, 0.5)], header={"schema": 1})
    with pytest.raises(ValueError, match="schema"):
        m.ingest([good, old], db, batch_size=2)
    tables = {name for (name,) in _rows(db, "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "points" not in tables or _rows(db, "SELECT COUNT(*) FROM points") == [(0,)]


@bug(L3)
def test_L3_skips_are_counted(m, tmp_path, db):
    """Minor 10: skipped records are only logged one at a time; the returned stats can't show data loss."""
    shard = _shard(tmp_path / "w1.jsonl", [_pt("MOF-A", 1.0, 1.0), _pt("MOF-A", 5.0, -1.0)])
    assert m.ingest([shard], db).get("skipped") == 1


@fine(L3)
def test_L3_parse_shard_closes_its_file(m, tmp_path, monkeypatch):
    """Red herring: yield inside `with open(...)` is fine; the file closes when the generator ends."""
    opened = []

    def tracking_open(*args, **kwargs):
        f = open(*args, **kwargs)
        opened.append(f)
        return f

    monkeypatch.setattr(m, "open", tracking_open, raising=False)
    shard = _shard(tmp_path / "w1.jsonl", [_pt("MOF-A", 1.0, 1.0), _pt("MOF-A", 5.0, 2.0)])
    list(m.parse_shard(shard))
    early_stop = m.parse_shard(shard)
    next(early_stop)
    early_stop.close()  # a consumer that stops after the first point
    assert len(opened) == 2 and all(f.closed for f in opened)


@fine(L3)
def test_L3_summary_correct_for_sorted_points(m, tmp_path, db):
    """Red herring: each groupby group is used before groupby advances, so sorted input works."""
    points = [_pt("MOF-A", 1.0, 1.0), _pt("MOF-A", 10.0, 3.0), _pt("MOF-B", 1.0, 0.5)]
    m.ingest([_shard(tmp_path / "w1.jsonl", points)], db)
    assert _rows(db, "SELECT * FROM mof_summary ORDER BY mof_id") == [("MOF-A", 3.0, 2), ("MOF-B", 0.5, 1)]
