# Answer key: 19_results_export.py

**Good clarifying questions:** How long do ingest transactions run, and are there several
ingesting workers? When a structure is recomputed, which energy should win? How many rows
arrive per night, and how big is the training set? Does "no +U" (NULL) mean something different
from U = 0? Is anything besides the training job reading the Parquet?

## Blockers

1. **Rows without +U are never upserted** (`UniqueConstraint` with a nullable `u_value`).
   Unique constraints treat NULLs as distinct (the Postgres default), so `ON CONFLICT` never
   fires when `u_value` is NULL. Ingesting MOF-5 twice gave 2 rows; CuBTC with U = 4 eV gave 1.
   Most MOF runs have no +U, so every recomputation adds a duplicate row with a different
   energy. *Fix:* `NOT NULL` with a documented sentinel (0.0 meaning no U),
   `UNIQUE NULLS NOT DISTINCT` (Postgres 15+), or a separate key column.

2. **Recomputed rows keep their old `updated_at`** (`ingest`). The ORM's `onupdate` doesn't run
   for an `INSERT ... ON CONFLICT DO UPDATE`. In the check, the energy changed to −905 eV but
   `updated_at` stayed the same. The nightly export selects on `updated_at`, so it never sees
   recomputed results, and the training set keeps the old energies forever. *Fix:* add
   `"updated_at": func.now()` to `set_`, or use a trigger.

3. **The watermark misses rows that commit late** (`export_new`). `server_default=func.now()`
   is the time the *transaction started*, not when it committed. A long ingest that starts
   before a short one, but commits after it, gets a timestamp older than the watermark the
   short one already moved. In the check, `slow-batch` was never exported: the first export
   wrote 1 row and the second wrote 0. `>` also skips rows that share the watermark's
   timestamp. *Fix:* re-read an overlap window (`updated_at > since - interval '1 hour'`) and
   make the write idempotent per key, or record changes in an outbox table or a sequence
   instead of timestamps.

4. **The fingerprint is Python's `hash()`** (`fingerprint`). It is salted per process: three
   processes gave three different values. Tonight's fingerprints never match last night's, so
   the deduplication never matches anything, and every re-exported structure is appended
   again. The training set fills up with duplicates, old and new energies side by side. With a
   stable fingerprint the logic would be wrong the other way: it would drop the recomputed row
   and keep the stale one. *Fix:* `hashlib.sha256` of a canonical structure representation,
   and replace by key (newest wins) instead of skipping.

5. **The watermark is saved before the data is written.** When the Parquet write failed
   ("disk full" in the check), the re-run exported 0 rows, and the `late` structure was lost
   for good. *Fix:* write the data first, then the watermark. Make the data write idempotent
   (deterministic file names, `existing_data_behavior="delete_matching"`), so that a crash
   between the two steps can't duplicate anything. Or keep the watermark in the database, in
   the same transaction as an export log.

## Major

6. **The same key twice in one batch aborts the whole batch on Postgres:**
   `CardinalityViolation: ON CONFLICT DO UPDATE command cannot affect row a second time`.
   SQLite silently keeps the last row instead, so SQLite tests pass. Workflow retries can
   easily send a structure twice. *Fix:* deduplicate the batch by key, keeping the latest,
   before the INSERT.
7. **The training job can't read the directory** (`load_training_set`). `pd.read_parquet(ROOT)`
   takes the schema from one file. The first night had no +U runs, so `u_value` was written
   with Arrow's `null` type. After a night with U = 4, reading failed with
   `ArrowNotImplementedError: Unsupported cast from double to null`. The file names are random,
   so it can work one night and fail the next. Columns added later get dropped silently, by the
   same mechanism. *Fix:* write with an explicit `pa.schema` (float64 for `u_value`), and read
   with `pyarrow.dataset.dataset(ROOT, schema=...)`.
8. **The export loads far too much.** It reads the whole training set every night just to get
   the fingerprints, and the first export (watermark 1970) loads every calculation, CIFs and
   forces included, into memory. Both grow without bound. *Fix:* read only the fingerprint
   column (`columns=[...]`), or keep a key index in the database, and stream rows with
   `yield_per`.

## Minor

9. **Re-running an export duplicates data.** `write_to_dataset` names files with random GUIDs.
   Use a deterministic `basename_template` per export.
10. **Small things:** the engine is created at import time; the forces are a JSON string inside
    Parquet, so the trainer has to parse JSON for every structure; nothing logs how many rows
    were read, skipped and written.

## Red herrings (say why they're fine)

- **`INSERT ... ON CONFLICT DO UPDATE`** is the right tool. It's atomic under concurrency,
  unlike select-then-insert or `session.merge`. The bugs are in the key and in `set_`.
- **`DateTime(timezone=True)`** gives `timestamptz` on Postgres, and comparing it with an aware
  watermark is fine.
- **`with Session(engine) as session: ... session.commit()`** is correct session handling.
- **`order_by(updated_at)`, then `rows[-1]`** gives the latest row, as intended.
- **Partitioning by functional** (a handful of values) is a sensible, low-cardinality choice.

## Model 1-minute summary
"Recomputed results never reach the training set, and duplicates do. The unique key includes a
nullable `u_value`, so for the usual no-+U runs the upsert never matches and every
recomputation adds a row. Where it does match, `updated_at` isn't touched, so the export never
sees the change. The watermark misses rows from transactions that commit late, and it's saved
before the data, so a failed write loses rows for good. Deduplication uses Python's `hash()`,
which differs between processes, so nothing ever matches. Operationally, the same key twice in
a batch aborts it on Postgres while passing on SQLite, and the training job's directory read
breaks once a null-typed `u_value` file meets a float one. I'd ask for tests that run on
Postgres, plus an end-to-end test that recomputes one structure and checks the training set
holds exactly the new energy."
