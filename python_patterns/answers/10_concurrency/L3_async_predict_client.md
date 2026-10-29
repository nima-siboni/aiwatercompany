# Answer key: L3_async_predict_client.py — Concurrency, level 3 (35 min + 5 min summary)

**Pattern focus:** asyncio only helps if nothing blocks the event loop, and a fan-out needs the
same care as a thread pool: bounded concurrency, timeouts, per-item failure handling, results
matched to their inputs, and no background task left without an owner. Prioritise the silent
data problems first, then the throughput killers.

**Good clarifying questions:**
- What rate limit and concurrency does the surrogate API allow? Does it accept batches?
- Do the campaign scripts line results up with their candidate list (do they rely on the order)?
- Is the results store the record of what was screened? Do missing rows matter to anyone?
- What should a failed or late prediction look like to the caller?
- Is `screen()` really called from notebooks, and which event loop is the transport tied to?
- Can the model be redeployed while a run is going?

## Blockers (silently wrong results / data loss)

1. **Results come back in completion order, not input order.** The docstring promises "in the
   same order as `candidates`", but `as_completed` yields whatever finishes first, and
   `predictions.append(await next_done)` keeps that order. Campaign code that lines the results
   up with its candidates (`np.array([p.uptake_mmol_g for p in preds])` next to the feature
   matrix) assigns uptakes to the wrong MOFs. Each `Prediction` carries its ID, so nothing *has*
   to break, but the contract says order, so callers will rely on it. *Fix:* store each result
   at its input index (or use `asyncio.gather`, which keeps input order), and test the contract.

2. **The saves are fire-and-forget, and they get lost.** `asyncio.create_task(self._save(...))`
   drops the task:
   - The event loop keeps only weak references to tasks; the asyncio docs warn that a task
     nobody references can be garbage-collected before it finishes.
   - `screen()` goes through `asyncio.run`, which cancels every task still pending when
     `predict_many` returns, so the last saves of every run are cancelled. With 10 ms saves in
     the checks, none of five predictions was stored.
   - Exceptions in `_save` are never retrieved; at best asyncio logs "Task exception was never
     retrieved" when the task is collected.

   The results store silently misses predictions, and because of #3 a save that failed every
   retry is still added to `self.saved`. *Fix:* await the save inside `predict_one`, or keep the
   tasks in a set and gather them before returning.

3. **Exhausted retries turn into a 0.0 mmol/g prediction.** After `MAX_RETRIES` transient
   errors, `_post` logs and returns `{"uptake_mmol_g": 0.0}`. Under load, which is exactly when
   429s happen, those candidates come back as plausible zero uptakes; `top_candidates` ranks them
   last, and they drop out of the screen without anyone knowing they were never predicted. The
   same fallback makes a failed `/results` save look successful. *Fix:* raise after the last
   attempt and return the candidate as failed (`uptake_mmol_g=None` plus an `error`), never as a
   made-up value.

## Major

4. **`_model_version` blocks the event loop.** `urllib.request.urlopen` is a synchronous network
   call inside `async def`, and `predict_one` makes it for every candidate. While it waits,
   nothing else runs: the 100k version fetches happen strictly one after another (at 20 ms each,
   over half an hour) and stall every in-flight request while they do. It also bypasses the
   injected transport, so tests that pass a fake still hit the network, and whatever the
   transport sets up (auth, base URL, proxies) is skipped. *Fix:* fetch the version through the
   transport, once per run (#10). `asyncio.to_thread` is the escape hatch for a blocking call you
   can't replace.

5. **`time.sleep` in the retry loop freezes everything.** Inside a coroutine,
   `time.sleep(RETRY_DELAY_S)` blocks the whole event loop: one 429 stops every in-flight request
   for half a second, and under a burst of 429s the loop spends most of its time asleep. *Fix:*
   `await asyncio.sleep(...)`.

6. **Unbounded fan-out.** `predict_many` creates a task per candidate and starts them all at
   once: 100k simultaneous requests. That floods the API (more 429s, so more retries), and unless
   the transport caps its connection pool, it opens sockets until the process runs out of file
   descriptors. Even with a capped pool, 100k tasks and their payloads sit in memory. *Fix:*
   bound the concurrency: an `asyncio.Semaphore`, or a fixed number of worker coroutines pulling
   from a shared iterator or an `asyncio.Queue` (the reference fix runs 32 workers).

7. **One failure loses every result.** A non-transient error (a 400 for an unparseable CIF, a
   response without `uptake_mmol_g`) propagates out of `await next_done`, out of `predict_many`
   and out of `screen()`. The predictions collected so far vanish with the local list, and the
   other tasks are cancelled when `asyncio.run` exits. One bad CIF in 100k throws the whole
   screen away. *Fix:* catch per candidate and return a failure entry, or use
   `gather(..., return_exceptions=True)`.

8. **At the deadline, unfinished work is dropped silently and left running.** When
   `as_completed` times out, the code logs a count and returns the partial list. The unfinished
   tasks aren't cancelled: through `screen()`, `asyncio.run` happens to cancel them on exit, but
   in a long-lived loop (a service, a notebook) they keep calling the API, retrying and saving
   after the caller has its answer. The caller gets a shorter list with no way to tell which
   candidates are missing, and because of #1 it can't even line the list up. *Fix:* cancel
   what's pending, wait for the cancellations, and return an explicit "missed the deadline"
   entry for each of those candidates.

9. **`screen()` crashes in notebooks.** Jupyter already runs an event loop, so `asyncio.run(...)`
   raises `RuntimeError: asyncio.run() cannot be called from a running event loop`, and the PR
   says notebooks call `screen()`. *Fix:* in a notebook, `await client.predict_many(...)`
   directly. If `screen()` must work there, detect the running loop and run the coroutine on a
   fresh loop in a worker thread; the reference fix does this, and it only works if the transport
   isn't tied to the notebook's loop.

## Minor

10. **The model version is fetched once per candidate.** That's 100k calls for a value that
    should be fixed for the run, and if the model is redeployed mid-run, one screen silently mixes
    predictions from two models. *Fix:* fetch it once at the start of `predict_many` and send it
    with every request (the payload already has a `version` field the server can check).

11. **No per-request timeout.** A hung connection keeps its task alive until the overall
    deadline, an hour by default. Once concurrency is bounded (#6), a few hung requests also hold
    worker slots. *Fix:* `asyncio.wait_for(...)` or `async with asyncio.timeout(...)` around each
    call.

12. **Fixed retry delay with no backoff or jitter.** Every task that gets a 429 sleeps the same
    0.5 s and retries in lockstep, so the retries hit the API as another synchronised burst.
    *Fix:* exponential backoff with jitter, and honour `Retry-After` if the API sends it.

## Red herrings (say why they're fine)
- **Tasks append to `self.saved` without a lock.** Coroutines on one event loop only switch at
  an `await`, and `list.append` doesn't await, so the appends can't interleave. In asyncio the
  races live in state that's shared *across* an `await`.
- **Only `TransientError` is retried.** That's right: a 400 for a malformed CIF fails the same way
  every time, so retrying it only adds load. The problem is what happens after the last retry
  (#3), not which errors get retried.
- **`_report_progress` is a `while True` task.** It looks like a leak, but `predict_many` keeps a
  reference and cancels it in `finally`. The prediction tasks are safe from garbage collection
  too, because the `tasks` list holds them; only the save tasks (#2) have no owner.

## Model 1-minute summary
"Three silent data problems first. Results come back in completion order, although the docstring
promises input order, so callers that line them up with their candidates mislabel uptakes. The
saves are fire-and-forget tasks, so `asyncio.run` cancels whatever is still pending at the end and
the results store silently misses predictions. And after the last retry, `_post` returns 0.0, so
failed candidates look like real, poor MOFs. Then throughput: urlopen and `time.sleep` block the
event loop, and nothing bounds the concurrency, so 100k requests hit the API at once. Then
robustness: one bad CIF throws away every result, the deadline drops unfinished candidates without
saying which, and `asyncio.run` fails in notebooks. I'd rebuild it around a bounded pool of worker
coroutines that returns a result or an explicit error per candidate, in input order, with timeouts
and jittered backoff."

## Reference fix
`L3_async_predict_client_fixed.py`:
- A fixed pool of `max_concurrency` worker coroutines shares one iterator and writes each result
  into its input slot. Failures and deadline misses come back as
  `Prediction(..., uptake_mmol_g=None, error=...)`.
- The save is awaited inside `predict_one`. At the deadline, unfinished workers are cancelled and
  awaited.
- Every call goes through the transport: the version is fetched once per run, each call has
  `asyncio.wait_for(..., REQUEST_TIMEOUT_S)`, and retries use `await asyncio.sleep` with
  exponential backoff and jitter.
- `screen()` uses a fresh loop in a worker thread when it's called inside a running loop, and
  `top_candidates` skips failed predictions.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_10_concurrency.py -k L3 -v`
