# 12. Services and data: model APIs, databases and Parquet

Code that serves models and stores results passes tests that send one request and run the job
once. Its bugs show up on a second request, a rerun or a second worker, and many give plausible
numbers rather than errors. In a review, ask what happens with two requests at once, a rerun and
a late row.

Outputs come from Python 3.14, FastAPI 0.162, pydantic 2.14, SQLAlchemy 2.1, SQLite 3.53, pandas
3.1, pyarrow 26, scikit-learn 1.9 and PyTorch 2.15; Postgres results from PostgreSQL 16.2.

## Quick table

| You see | What goes wrong | Fix |
|---|---|---|
| `session = Session(engine)` at module level | requests share one transaction; one failed commit breaks every later request | one session per request, from a `yield` dependency |
| `fit_transform` in the serving path | the scaler is refitted on the request; one row becomes all zeros | load the fitted pipeline and only call `transform` |
| `np.array(list(features.values()))` | features in the client's key order, not the training order | select by name, in the training order |
| `model.train()` for dropout at inference | BatchNorm updates its running statistics; the served model drifts | `eval()`, then `train()` on the dropout layers only |
| a pydantic model without `extra="forbid"` | misspelt fields are ignored and defaults used; NaN and inf pass | `extra="forbid"`, `allow_inf_nan=False`, bounds and length limits |
| a numpy value or NaN returned from an endpoint | a 500, or a silent `null` (FastAPI 0.130+ with a response model) | `float()`, `.tolist()`, `math.isfinite` |
| `torch.set_grad_enabled(False)` at startup | grad mode is per thread, so `def` endpoints still build graphs | `torch.inference_mode()` around each model call |
| `LIMIT n OFFSET k` in a loop that changes rows | rows are skipped or repeated | keyset pagination on a unique key |
| `on_conflict_do_update` | `onupdate` isn't applied; a NULL in the key never conflicts | set `updated_at` in `set_`; `NOT NULL` key columns |
| `INSERT OR REPLACE`, `to_sql(if_exists="replace")` | delete and insert: other columns, constraints and indexes are lost | `ON CONFLICT ... DO UPDATE`; append into a table you created |
| `DateTime` without `timezone=True`, `utcnow()` | instants shift by the session's or the machine's offset | aware UTC datetimes and `timestamptz` |
| `WHERE updated_at > :last_seen` | rows that commit late are never exported | an overlap window and idempotent upserts |
| `pd.read_parquet(directory)` | the schema comes from the first file; later columns are dropped | an explicit schema when writing and reading |

## Examples

### One `Session` for every request

```python
session = Session(engine)                   # created once, at import
@app.post("/predict/{sid}")
def predict(sid: str):                      # runs in a worker thread
    session.add(Prediction(sid=sid, value=predict_energy(sid)))
    session.commit()
    return {"ok": True}
# sid is unique. Requests A, B, A, C, D: 200, 200, 500 (IntegrityError), then 500, 500
# (PendingRollbackError). 400 concurrent requests: all 400 failed and no row was stored.
```

A `Session` is one transaction and is not thread-safe, and `def` endpoints run in a thread pool;
after a failed commit it waits for a `rollback()` that never comes. **Fix:** one session per
request from a `yield` dependency, with the engine global; return IDs or plain values, since an
ORM object read after its session has committed and closed raises `DetachedInstanceError`.

### Preprocessing at serving time

```python
features = {"density": 0.6, "pore_diam": 7.0, "void_frac": 0.7}      # from the request
x = pd.DataFrame([features])[FEATURES]           # FEATURES is the training order
StandardScaler().fit_transform(x)                # [[0. 0. 0.]]: refitted on one row
scaler.transform(x)                              # [[-0.45 -0.73  0.65]] with the training scaler
model.predict(np.array([list(features.values())]))           # [64.12], with only a UserWarning
model.predict(np.array([[features[f] for f in FEATURES]]))   # [17.9]
np.log10(1.0e5), np.log(1.0)                     # 5.0, 0.0: trained on log10(Pa), served ln(bar)
```

Each wrong line still returns a plausible number. A DataFrame with its columns in another order
raises (scikit-learn 1.2+), but a bare array is only checked for its width. **Fix:** save the
fitted `Pipeline` and only call `predict`; select features by name; share one `featurize()` with
training, and test that a training row gets the same prediction through the API.

### MC dropout with `model.train()`

```python
model.train()                          # to turn on dropout
with torch.no_grad():
    samples = torch.stack([model(x) for _ in range(20)])
model.eval()                           # the eval output for x has now moved by up to 2.39
model.train(); model(x[:1])
# ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 4])
```

`train()` also switches BatchNorm, which then normalises with batch statistics and updates its
running statistics on every pass, even under `no_grad`. **Fix:** stay in `eval()` and call
`.train()` only on the `nn.Dropout` modules; the eval output then doesn't move.

### Request and response models

```python
class Query(BaseModel):
    pressure_bar: float
    temperature_k: float = 298.0
# {"pressure_bar": 1, "temprature_k": 77}: 200, computed at 298.0
# {"pressure_bar": "NaN"} or the JSON literal -Infinity: 200, with nan or -inf
return {"energy": y}
# np.float32: 500, ValueError: [TypeError("'numpy.float32' object is not iterable"), ...]
# nan: 500, ValueError: Out of range float values are not JSON compliant: nan
# nan with response_model=Out: 200 {"energy":null} on FastAPI 0.130+, a 500 on 0.129
```

Unknown keys are ignored, floats accept NaN and infinity, and the body has no size limit (a
20 MB list was parsed). `np.float64` subclasses `float`, so a float64 test model hides the 500.
**Fix:** `ConfigDict(extra="forbid")`, `Field(gt=0, allow_inf_nan=False)` and `max_length` (each
gives a 422), a body limit in the proxy, and `float(y)` after a `math.isfinite` check.

### Threads, workers and the lifespan

```python
@asynccontextmanager
async def lifespan(app):
    torch.set_grad_enabled(False)          # meant for the whole server
    app.state.model = load_model()
    yield
@app.post("/predict")
def predict(request: Request):             # runs in a worker thread
    y = request.app.state.model(x)         # y.requires_grad is True; False in an async def
def get_model(): return load_model()       # as Depends(get_model): 3 requests, 3 loads
JOBS: dict[str, dict] = {}                 # job status, under uvicorn --workers 4 (not run here)
```

Grad mode is per thread, and `def` endpoints run in a 40-thread pool, so each request builds a
graph; behind 45 slow predictions, a `def` health check took 1.81 s, and 0.00 s as `async def`.
Each worker has its own model and `JOBS`. **Fix:** `inference_mode()` per call, `async def`
health checks, one load in the lifespan, jobs in a database, and `with TestClient(app)` in tests.

### OFFSET pagination over changing rows

```python
page = 0
while rows := db.execute("SELECT id FROM job WHERE status = 'pending' "
                         "ORDER BY id LIMIT 100 OFFSET ?", (page * 100,)).fetchall():
    db.executemany("UPDATE job SET status = 'done' WHERE id = ?", rows)
    page += 1
# 1000 pending jobs: 500 processed, 500 still pending, no error
```

Each update shrinks the filtered set, so every offset skips a page. On Postgres, a loop without
`ORDER BY` that updated each page saw only 600 distinct rows of 1,000. **Fix:** keyset
pagination on a unique key, `WHERE id > :last ORDER BY id LIMIT 100`; it processed all 1,000, and
took 0.02 ms per page where `OFFSET 900000` took 2.9 ms (SQLite).

### Upserts and their keys

```python
stmt = insert(Result).values(rows)            # the sqlite or postgresql insert()
stmt = stmt.on_conflict_do_update(
    index_elements=["sid", "method", "kpoints"],
    set_={"energy": stmt.excluded.energy})     # updated_at has onupdate=datetime.now
session.execute(stmt)
# energy changes, updated_at doesn't (an ORM update does change it); kpoints=None: 3 upserts, 3 rows
# the same key twice in rows: SQLite keeps the last; Postgres raises CardinalityViolation:
# ON CONFLICT DO UPDATE command cannot affect row a second time
```

`onupdate` fires for SQLAlchemy UPDATEs, not for `ON CONFLICT` (Postgres too), so readers of
`updated_at > :since` miss the change. NULLs are distinct in unique indexes, and float keys
rarely match: `0.3` isn't a key of a dict built from `np.linspace(0, 1, 11)`. **Fix:**
`updated_at` in `set_`, a deduplicated batch, `NOT NULL` keys or `NULLS NOT DISTINCT`
(PostgreSQL 15+), and integer or rounded keys that include the model version.

### "Replace" deletes

```python
db.execute("INSERT INTO results VALUES ('A', -100.0, 3.4)")     # sid, energy, band_gap
db.execute("INSERT OR REPLACE INTO results (sid, energy) VALUES ('A', -101.0)")
# [('A', -101.0, None)]: band_gap is gone
df.to_sql("results", db, if_exists="replace", index=False)
# the new table has no primary key, NOT NULL or index, and both duplicate rows of df went in
```

`OR REPLACE` deletes the conflicting row and inserts a new one; `if_exists="replace"` drops the
table and rebuilds it from the DataFrame's dtypes. **Fix:** `ON CONFLICT(sid) DO UPDATE SET
energy = excluded.energy`; for pandas, append into a table you created, or use
`if_exists="delete_rows"` (pandas 3).

### Timestamps and watermarks

```python
t = datetime(2024, 1, 1, 10, 0, tzinfo=timezone(timedelta(hours=2)))    # 08:00 UTC
session.add(Event(ts=t))     # SQLite, DateTime(timezone=True): stored as 10:00, read back naive
datetime.utcnow().timestamp() - time.time()     # -3600 with TZ=Europe/Berlin in winter
rows = db.execute("SELECT name, updated_at FROM calc WHERE updated_at > ?", (wm,)).fetchall()
wm = max(r[1] for r in rows) if rows else wm
# a row stamped 10:00:01 that commits after a run exported 10:00:05 is never exported
```

The naive value read back raises `TypeError` against `datetime.now(timezone.utc)`. On Postgres,
an aware value in a `timestamp` column is converted to the session's time zone (08:00 under UTC,
03:00 under New York), and `DEFAULT now()` is the transaction's start, so a long transaction
commits behind the watermark. **Fix:** aware UTC datetimes and `timestamptz`; an overlap window
with an idempotent upsert, and the watermark saved in the same transaction as the output.

### Deduplication

```python
merged = pd.concat([existing, new_batch]).drop_duplicates(subset=["sid"])
# A was recomputed as -1.2, but merged keeps -1.0
key = hash(cif_text)                  # stored as the dedup key; 3 processes gave 3 values
for chunk in iter(lambda: f.read(1 << 20), ""):    # f opened with "rb": never stops
    h.update(chunk)
```

`drop_duplicates` keeps the first occurrence, so the old row wins; `hash()` of a string is salted
per process; `""` never equals `b""`. **Fix:** sort by version and `keep="last"`, or upsert in the
database; `hashlib.sha256` of a canonical form; `b""` as the sentinel, or `hashlib.file_digest`.

### Parquet directories and storage dtypes

```python
pd.DataFrame({"sid": ["a"], "energy": [-1.0]}).to_parquet("out/part-0.parquet")
pd.DataFrame({"sid": ["b"], "energy": [-2.0], "unc": [0.1]}).to_parquet("out/part-1.parquet")
pd.read_parquet("out/").columns.tolist()         # ['sid', 'energy']: 'unc' is dropped
df.to_parquet("ds/", partition_cols=["batch_id"])          # batch_id "0012"
pd.read_parquet("ds/").batch_id.tolist()                   # [12], a category of int32
pq.write_to_dataset(table, "res/"); pq.write_to_dataset(table, "res/")   # 4 rows from 2
pd.to_numeric(energies, downcast="float")        # float32: -12345.678 - (-12345.679) = 0.0009765625
```

The schema comes from the first file: a later column is dropped, an all-None column there fails
once a later file has values (`ArrowNotImplementedError`), and so does an int column once one has
0.5 (`ArrowInvalid`). Partition values are re-typed from directory names, and random file names
make a rerun add a copy. **Fix:** an explicit `pa.schema`, a string partition schema, a fixed
`basename_template` with `existing_data_behavior="delete_matching"`, and float64 energies.

## Looks wrong, is fine

- A plain `def` endpoint calling a slow model. FastAPI runs it in its thread pool; the bug is
  blocking code inside `async def`.
- `connect_args={"check_same_thread": False}` with SQLite: fine with one session per request.
- `tags: list[str] = []` in a pydantic model: pydantic copies the default for each instance.
- `col.in_([])`. SQLAlchemy renders `IN (SELECT 1 FROM (SELECT 1) WHERE 1!=1)` and returns `[]`.
- A request's `yield` session passed to `BackgroundTasks` (version-dependent). The task sees it
  open on FastAPI 0.118+, and already closed on 0.106 to 0.117 (checked on 0.117.1).
- `session.add()` in a loop, then one `commit()`. On Postgres, SQLAlchemy 2.1 sent 5,000 rows as
  5 INSERT statements; a commit inside the loop is the real problem.
