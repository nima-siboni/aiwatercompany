# Answer key: L3_gcmc_ingest.py — Generators, level 3 (35 min + 5 min summary)

**Pattern focus:** a lazy pipeline where every stage is a generator. For each stage, check what
it assumes about its input (sorted? single pass?), where its errors go (a generator that raises
is finished for good), what it holds in memory (`tee`, `seen` sets), and whether the last partial
batch is written. Laziness also moves failures far away from their cause.

**Good clarifying questions:** Are one MOF's points ever contiguous in a shard? (No: workers run
several MOFs at once, and retries move jobs between workers.) When a point appears twice, which
one should win? Does the nightly run re-read every shard, and can a MOF's points arrive over
several nights? What do bad records look like in practice: non-converged garbage, truncated
lines from killed workers, empty shards? How much memory does the ingest machine have?

## Blockers (silently wrong results / data loss)

1. **`summarise` runs `itertools.groupby` on points that aren't sorted by MOF.** `groupby` only
   groups *consecutive* items. Points arrive as A, B, A, ..., so it yields A, B and A as separate
   groups, and `INSERT OR REPLACE` keeps whichever fragment came last. With A at 10 bar = 3.0,
   B at 1 bar = 0.5 and A at 1 bar = 1.0, the summary says MOF-A's best uptake is 1.0 from one
   point, instead of 3.0 from two, and the dashboard ranks candidates on fragments.
   `sorted(glob(...))` sorts file names, not points; and even sorted input would summarise only
   this run, while the table holds every night's points. *Fix:* aggregate in the database, which
   sees every point in any order: `INSERT OR REPLACE INTO mof_summary SELECT mof_id,
   MAX(uptake_mmol_g), COUNT(*) FROM points GROUP BY mof_id`. (Sorting 50M points in Python is the
   memory problem the PR set out to fix.)

2. **The writer coroutine never writes its final partial batch.** `point_writer` writes only when
   `batch` reaches `batch_size`. `writer.close()` raises `GeneratorExit` at the `yield`, and the
   generator exits with up to 499 points still in `batch`. Every run silently loses its tail, a
   run with fewer than 500 converged points writes nothing, and `n_points` counts those points as
   ingested. *Fix:* replace the coroutine with a plain function that takes the iterable and writes
   what's left after the loop (`write_points` in the reference fix). Pulling from an iterator has
   no priming and no close step to forget.

3. **A bad record finishes `parse_shard`, and `read_shards` keeps calling a dead generator.** Once
   a generator raises, it's done, and the next `next()` raises `StopIteration`. `read_shards`
   catches `BadRecord`, logs "skipping bad record" and `continue`s; the next `next(points)` raises
   `StopIteration` and the loop `break`s. Every point after the first bad record in that shard is
   dropped, while the log says one record was skipped. In my check, a shard of A@1, bad, A@10, B@1
   gave only A@1. It's common, too: validation runs *before* the convergence filter, so a
   non-converged point with a negative uptake (which you'd discard anyway) takes the rest of its
   shard with it. *Fix:* handle per-record problems *inside* the generator (log, count, `continue`
   there) so it keeps going; `read_shards` becomes `yield from parse_shard(path)`.

## Major

4. **An empty shard aborts the whole ingest with `RuntimeError: generator raised StopIteration`.**
   `json.loads(next(f))` on an empty file (a worker that died before writing its header) raises
   `StopIteration` inside a generator, which Python 3.7+ turns into `RuntimeError` (PEP 479).
   `read_shards`' `except StopIteration` never sees it. *Fix:* `first = next(f, None)` and return
   if it's `None`. Never call a bare `next()` inside a generator.

5. **A truncated last line aborts the whole ingest.** A worker killed mid-write (OOM, preemption,
   wall-time limit) leaves half a JSON line; `json.loads` raises `JSONDecodeError`, which nothing
   catches. With many workers that's routine. *Fix:* skip and count undecodable lines inside
   `parse_shard`, in the same place as #3.

6. **`itertools.tee` buffers every point.** `to_db` is consumed completely before `to_summary`
   starts, so `tee` has to keep every point for the second branch: about 580 bytes per point by
   my tracemalloc measurement, so roughly 30 GB at 50M points. That's the out-of-memory the PR was
   written to fix, and the `tee` docs warn about exactly this. *Fix:* don't fan out. With the
   summary computed in SQL (#1) there's only one consumer.

7. **`dedupe` keeps an unbounded `seen` set, and doesn't need to exist.** About 220 bytes per point
   (measured), ~11 GB at 50M points, on top of #6. And `points` already has
   `PRIMARY KEY (mof_id, temperature_k, pressure_bar)` with `INSERT OR REPLACE`, which dedupes
   retries (last write wins), across nights too, which the in-memory set can't do. *Fix:* delete
   `dedupe`, and confirm that last-wins is the right policy for a retried point.

## Minor

8. **Laziness moves the schema check into the middle of the run.** A header is only checked when
   the pipeline reaches that shard. If shard 250 of 300 still has schema 1, the run fails after
   the earlier batches are committed and before the summary step, so the database holds new
   points with stale summaries. *Fix:* check every header up front (one line per shard) before
   writing anything. The writes are idempotent, so rerunning after a mid-run I/O error is safe.

9. **The coroutine API is easy to misuse.** Prime with `next()`, `send()`, then `close()`: three
   steps to get right, and the forgotten flush (#2) is the classic mistake. A plain function over
   an iterable is easier to read and to test.

10. **The returned stats can't show data loss.** `n_points` counts points *sent* to the writer,
    not written (#2), and skipped records only show up as individual log lines. *Fix:* return the
    number written and a `skipped` count, so the nightly job can alert on them.

## Red herrings (say why they're fine)
- **`yield` inside `with open(...)` in `parse_shard` is fine.** The file stays open while the
  generator is suspended and closes when it finishes or is closed. I checked both, including a
  consumer that stops after one point. Shards are read one at a time, so one file is open at once.
- **The way `summarise` uses each `groupby` group is right.** A group becomes invalid once
  `groupby` advances; `summarise` turns it into a list before yielding, so that classic bug isn't
  here. The bug is the unsorted input (#1), not the group's lifetime.

## Model 1-minute summary
"The top three are silent data problems. One: `summarise` runs `groupby` on points that aren't
sorted by MOF, so each MOF is split into fragments and the summary keeps the last one; the
dashboard ranks candidates on partial data. Two: the writer coroutine never flushes its last
partial batch, so every run loses up to 499 points while reporting them as ingested. Three: once
`parse_shard` raises `BadRecord` it's finished, so the `continue` in `read_shards` silently drops
the rest of the shard. Then two crashes: an empty shard hits PEP 479's 'generator raised
StopIteration', and a truncated line kills the run. Then memory: `tee` and the `seen` set hold
tens of gigabytes at 50M points, the problem the PR set out to fix. I'd let SQLite dedupe and
aggregate, handle bad records inside the generator, and check headers up front."

## Reference fix
`L3_gcmc_ingest_fixed.py` (`parse_shard`, `read_shards` and `ingest` keep their signatures):
- `parse_shard` reads the header with `next(f, None)` and skips, logs and counts bad or
  undecodable lines itself; `read_shards` is a `yield from`.
- `check_headers` validates every shard before anything is written.
- `write_points` replaces the coroutine and writes the last short batch.
- `dedupe`, `summarise` and the `tee` are gone: the primary key dedupes, and `mof_summary` is
  rebuilt by one `GROUP BY` query over the whole table.
- `ingest` returns the points actually written and a `skipped` count.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_04_generators.py -k L3 -v`
