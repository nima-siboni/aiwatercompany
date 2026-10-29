# Answer key: L2_retry_backoff.py — Decorators, level 2 (20 min)

**Pattern focus:** decorator factories (`@retry(...)`) and the state they close over. Check
what happens on the last attempt, what lives in the closure and who shares it, what a bare
`@retry` binds to, and which calls are safe to retry at all.

**Good clarifying questions:** Does sim-api accept an idempotency key (or a client job tag)
on `POST /jobs`, so a resubmission returns the existing job? Which failures are transient:
does it send 503/429 under load, and a `Retry-After` header? What does a FAILED job mean:
deterministic (the calculation didn't converge) or infrastructure (a node died)? How many
workers call sim-api at once? What does the AL loop do with a missing result?

## Blockers (silently wrong results)

1. **When the attempts run out, the decorator returns None instead of raising.** The loop is
   `range(1, max_attempts)`, so `max_attempts=5` makes 4 attempts, and when they all fail
   the wrapper falls off the end and returns `None`. If sim-api is unavailable for about
   15 s, `get_result` gives up and returns `None`, and `relax` returns `None` for a
   relaxation that actually finished. The AL loop then records no energy, or crashes much
   later on `None["energy"]`, far from the cause. `@retry(max_attempts=1)` is worse: the
   range is empty, so the function is never called and `None` comes back at once. Combined
   with the retry on `relax` (issue 6), a FAILED relaxation also becomes `None`. *Fix:*
   `for attempt in range(max_attempts)` and re-raise the last exception (`raise` inside
   the `except` on the final attempt).

2. **A bare `@retry` on `get_job_status` replaces it with the inner decorator.** `@retry`
   without parentheses calls `retry(get_job_status)`, so `max_attempts` is the function and
   the name `get_job_status` is bound to `decorator`. Calling `get_job_status("job-1")`
   wraps the string and returns a function. No request is ever made, and the return value
   is never `"COMPLETED"` or `"FAILED"`, so `relax` polls forever while the cluster job
   finishes. Nothing raises. *Fix:* `@retry()`. Better, make the factory's parameters
   keyword-only (`def retry(*, max_attempts=3, ...)`), so the bare form raises `TypeError`
   at import time.

## Major

3. **The backoff state is shared by every call and never resets.** `delay` lives in
   `decorator`'s scope and is updated with `nonlocal`, so all calls (and all threads) to one
   decorated function share it, and it only ever doubles. After 15 transient failures of
   `get_result` over a long AL campaign, the next call's first retry sleeps 2^15 s, about
   9 hours. Concurrent threads also double it for each other. *Fix:* compute the delay
   locally from the attempt number, e.g. `base_delay * 2**attempt`.

4. **Every exception is retried, including client errors and bugs.** With
   `exceptions=(Exception,)`, a 404 (unknown job ID), a 422 (bad `calc`), a `KeyError` from
   a malformed response and the FAILED `RuntimeError` are all retried with sleeps, and then
   turned into `None` by issue 1. *Fix:* retry only transient errors: timeouts, connection
   errors, and HTTP 429/5xx. Watch out: `HTTPError` is a subclass of `URLError`, so
   `except URLError` alone still retries every 4xx.

5. **`submit_job` retries a POST that isn't idempotent.** If the request times out after
   sim-api has queued the job, the retry queues a second multi-hour DFT relaxation. With
   `max_attempts=5` you can get up to 4 copies of the job, and duplicate results downstream.
   *Fix:* depends on the clarifying question. Send a client-generated idempotency key,
   created once per logical submission and reused by its retries, so sim-api can
   deduplicate. If sim-api can't do that, don't retry the POST on a timeout; look the job up
   by a client-side tag first.

6. **`@retry` on `relax` retries the whole workflow.** Any exception after submission makes
   `relax` start again from `submit_job`. A transient error in `get_result` re-runs a 1–6 h
   job that has already finished, and a genuinely FAILED job (a deterministic failure) is
   resubmitted. Once issue 1 is fixed, the retries also multiply: as configured, up to
   3 × 5 = 15 attempts of `get_result` for one relaxation. *Fix:* no retry at the workflow
   level. Retry the individual idempotent calls only, and treat FAILED as a result to report.

## Minor

7. **It sleeps after the final attempt.** The last failure still sleeps (16 s for
   `submit_job`) before giving up. *Fix:* sleep only when another attempt follows.
8. **No jitter and no cap.** Every AL worker that hit the same 503 retries at the same
   moments (a thundering herd on a service that is trying to recover), and the delay grows
   without limit (1024 s by the 11th retry). *Fix:* full jitter,
   `random.uniform(0, min(max_delay, base_delay * 2**attempt))`.

## Nit
- `relax` has no overall deadline, so a job stuck in PENDING blocks the AL loop for ever. Add
  a `timeout_s` and raise `TimeoutError` with the job ID.

## Red herrings (say why they're fine)
- **Retrying the GETs (`get_job_status`, `get_result`) is fine.** Reads are idempotent, so
  repeating them is harmless. Only the POST and the workflow-level retry need care.
- **There's no status-code check after `urlopen`, and none is needed.** `urllib` raises
  `HTTPError` for 4xx and 5xx responses itself, unlike `requests`, which needs
  `raise_for_status()`.

## Model 1-minute summary
"Two blockers, both silent. First, the retry loop is off by one and has no final raise, so
once the attempts run out the decorated function returns None. A completed relaxation comes
back as None, and with `max_attempts=1` the function is never called at all. Second,
`get_job_status` uses a bare `@retry`, which replaces it with the inner decorator, so it
returns a function and `relax` polls forever. Then the retry policy: `nonlocal delay` is
shared across calls and threads and only ever doubles, so a long-running worker ends up
sleeping for hours. It retries every exception, including 404s. It retries the job
submission, which can queue duplicate multi-hour DFT runs, so I'd ask whether sim-api
supports idempotency keys. And it retries the whole `relax` workflow, which resubmits failed
jobs. Smaller things: don't sleep after the last attempt, and add jitter and a cap. I'd add
unit tests with a fake transport for each of these."

## Reference fix
`L2_retry_backoff_fixed.py`:
- `retry(*, max_attempts, base_delay, max_delay, retry_if)`: keyword-only, a loop over
  `range(max_attempts)` that re-raises on the last attempt, and a local delay with full
  jitter and a cap.
- `is_transient` retries timeouts, connection errors and 429/5xx, and nothing else.
- `submit_job` sends an `idempotency_key` created once per submission; the retried
  `_post_job` reuses it.
- `@retry()` on `get_job_status`; no retry on `relax`.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_01_decorators.py -k L2 -v`
