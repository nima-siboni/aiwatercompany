# Answer key: L3_surrogate_client.py — Decorators, level 3 (35 min + 5 min summary)

**Pattern focus:** stacked decorators on methods in a realistic service client. Check what
each wrapper does with its arguments (logging them, rewriting them), whether a sync wrapper
can work on an `async def`, where decorator state lives (one object per decorated method,
created when the class is defined), and the order the stack runs in.

**Good clarifying questions:** Is the 20 req/s limit per token, per endpoint or for the
whole service? Can descriptors be NaN (a failed pore-size calculation, say), and what should
happen to those candidates? How does the AL loop use `predict_many`'s output: zipped with
candidate IDs? Where do the logs go, and who can read them? How many worker threads share
one client?

## Blockers (security hole / silently wrong results)

1. **The API token is written to the logs on every call.** `log_call` logs `args`, and
   `args[0]` is `self`. `@dataclass` generates a `__repr__` that includes every field, so
   each call logs `SurrogateClient(base_url=..., api_token='sk-live-...', ...)` at INFO,
   from both `predict` and `predict_async`. Anyone with access to the log pipeline can call
   the service as us. *Fix:* `api_token: str = field(repr=False)`, and log the function
   name and argument shapes rather than the arguments. Rotate the token if this has shipped.

2. **`validate_features` doesn't validate; it silently imputes, in the caller's array.**
   `np.nan_to_num(X, copy=False)` turns NaN into 0.0 and ±inf into ±3.4e38, so a candidate
   whose pore-size descriptor failed gets a confident prediction for a framework with zero
   pore size. It also writes in place: `np.asarray(..., dtype=np.float32)` returns the
   caller's own array when it's already float32. In `predict_many` the batches are slices
   (views) of the round's feature matrix, so a float32 matrix gets zeros written into it,
   and they're used again later (for retraining, say). With float64 input a copy is made, so
   the behaviour depends on the dtype. *Fix:* reject non-finite rows with a `ValueError`
   that names them. If imputation is wanted, do it explicitly upstream with training-set
   statistics. Never write to the caller's array.

3. **`predict_many` drops failed batches, so predictions shift onto the wrong candidates.**
   It concatenates only the successful batches, so after one failed batch every later
   prediction belongs to a different MOF. `dict(zip(candidate_ids, preds))` stops at the
   shorter input without complaint, and the AL loop picks the wrong structures. Issues 4 and
   5 make failures common on this path. If every batch fails, `np.concatenate([])` raises
   `ValueError`. *Fix:* keep row *i* aligned with candidate *i*: fill failed batches with NaN
   (or raise) and log them.

## Major

4. **Sync `@retry` and `@timed` do nothing useful on `async def predict_async`.** Calling an
   async function only creates a coroutine, so the wrapper returns it straight away. The
   `try/except` never sees the 503; it is raised later, when the caller awaits. So the async
   path, which is the one scoring 50k candidates, has no retries, and `timed` records the
   microseconds it took to create the coroutine. *Fix:* the simplest is to make
   `predict_async` just `return await asyncio.to_thread(self.predict, features)`, which
   reuses the sync stack in a worker thread. Otherwise write `async def` wrappers that `await`. Make the sync
   decorators reject coroutine functions (`inspect.iscoroutinefunction`) at decoration time.

5. **On `predict_async` the rate limiter blocks the event loop and limits the wrong thing.**
   The limiter runs when `self.predict_async(batch)` is *called*, inside the generator that
   `gather(*...)` unpacks, so it calls `time.sleep` on the event-loop thread and freezes
   everything on the loop for seconds. All it throttles is coroutine creation. Once `gather`
   runs, the requests all start together, limited only by the default thread pool
   (`min(32, cores + 4)` at a time), far above 20 req/s. The result is 429s, which aren't
   retried (issue 4) and are then dropped (issue 3). *Fix:* as in issue 4, rate-limit inside
   the sync `predict` running in worker threads, or use an async limiter
   (`asyncio.Semaphore` plus `await asyncio.sleep`).

6. **The limiter records the time from before it slept.** After sleeping it appends `now`,
   which was read before the sleep, so the call is recorded up to `period` seconds earlier
   than it really went out. That stamp expires early and lets extra calls through. With
   `calls=2` and back-to-back calls, they go out at t = 0, 0, 1, 1, 1: three in one second.
   Under sustained load it allows up to twice the limit, and the service answers with 429s.
   *Fix:* take the timestamp after sleeping.

7. **The limiter isn't thread-safe, and the client is shared by the AL worker threads.**
   Rebinding `self._stamps`, the length check and the `append` are separate steps. Two
   threads can both see 19 stamps and both go, and a thread that rebinds `_stamps` from an
   old snapshot drops a stamp another thread has just appended. Both mean more than 20 req/s.
   *Fix:* a `threading.Lock` around check-and-record, and a `deque` for O(1) expiry.

8. **Stacking order: the retry is inside the rate limiter.** Decorators apply bottom-up, so
   `predict` is `log_call(RateLimiter(retry(timed(validate_features(f)))))`. A call takes one
   limiter slot, and then the retry makes up to 4 HTTP requests that never pass the limiter.
   When the service starts returning 429 because it's overloaded, every thread's retries add
   unthrottled traffic, at exactly the wrong moment. *Fix:* put `@retry` above
   `@RateLimiter`, so every attempt takes a slot. Related: `timed` inside `retry` records
   each attempt separately. That's probably what you want for latency, but the PR should say so.

## Minor

9. **One limiter per decorated method, shared by every client.** Decorators in a class body
   run once, when the class is defined. `predict` and `predict_async` each get their own
   `RateLimiter`, and every `SurrogateClient` instance shares them. If the limit is per token,
   two clients with different tokens throttle each other, while one client that uses both
   methods gets 40 req/s. *Fix:* depends on the clarifying question. Use one limiter per
   token (on the instance, or in a dict keyed by token), shared by both entry points.
10. **`log_call` formats eagerly and logs whole arguments.** The f-string is built even
    when INFO is disabled, including the repr of every argument on every call: about 55 µs
    for a 256×200 array (numpy summarises it), and about 3 ms if a caller passes lists (no
    summarising). *Fix:* lazy `%s` arguments at DEBUG, and log shapes and counts.
11. **`inspect.signature(func)` runs on every call.** It costs a few microseconds,
    negligible next to an HTTP request, but the answer never changes, so compute it once
    at decoration time.
12. **`time.time()` in the limiter.** If NTP steps the wall clock back an hour,
    `period - (now - stamps[0])` becomes 3601 s and the limiter sleeps for an hour. A step
    forward empties the window and allows a burst. *Fix:* `time.monotonic()`.
13. **`LATENCIES` grows forever**, one float per call per method, for the life of the
    process. *Fix:* a bounded `deque(maxlen=...)` per function, or a histogram.

## Red herrings (say why they're fine)
- **`retry` itself is correct for sync code.** It re-raises on the last attempt, doesn't
  sleep after it, and only catches `TransientError`, which `_post` raises for exactly the
  transient cases (`HTTPError` is checked before its base class `URLError`; 429/5xx only).
  The problems are where it's applied: on an async method, and inside the limiter.
- **`inspect.signature(func).bind(*args, **kwargs)` finds `features` correctly** whether it
  is passed by position or by keyword, with `self` first. Only the per-call cost is an issue.
- **`asyncio.to_thread(self._post, ...)`** is the right way to call blocking `urllib` from
  async code. The problem is the decorators around it.

## Model 1-minute summary
"Three blockers. First, `log_call` logs `self`, and the dataclass repr includes the API
token, so every request writes the token into our logs. Second, `validate_features`
doesn't validate. `nan_to_num` turns missing descriptors into zeros, so we get confident
predictions for nonsense inputs, and with float32 input it writes those zeros into the
caller's matrix. Third, `predict_many` drops failed batches and concatenates the rest, so
every prediction after a failure is attributed to the wrong MOF. Those failures will be
common, because the sync decorators don't work on `predict_async`: retry never retries,
timing reads zero, and the rate limiter blocks the event loop while throttling nothing
real. The limiter also records the pre-sleep time, isn't thread-safe and sits outside the
retry. I'd make `predict_async` a `to_thread` call to `predict`, fix the limiter, and add
tests for row alignment and for secrets in the logs."

## Reference fix
`L3_surrogate_client_fixed.py`:
- `api_token` has `repr=False`; `log_call` logs lazily at DEBUG, and only array shapes.
- `validate_features` rejects non-finite rows, never writes to its input, and builds the
  signature once.
- `predict_async` is `await asyncio.to_thread(self.predict, features)`. The sync decorators
  raise `TypeError` if they're applied to a coroutine function.
- `RateLimiter`: monotonic clock, a lock, a deque, and a timestamp taken after the sleep.
  It sits below `@retry`, so every attempt is limited.
- `predict_many` fills failed batches with NaN so rows stay aligned; `LATENCIES` keeps a
  bounded deque per function.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_01_decorators.py -k L3 -v`
(Not checked: the thread-safety race in issue 7, because it can't be forced
deterministically; the stacking order in issue 8, which needs concurrent traffic to show;
the shared limiter in issue 9, whose right fix depends on the clarifying question; and the
costs in issues 10 and 11.)
