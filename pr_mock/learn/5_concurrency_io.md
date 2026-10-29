# 5. Concurrency, I/O, retries, caching and basic security

Outputs come from Python 3.14. The learning nuggets for concurrency, caching and context
managers in `../../python_patterns/learn/` go deeper on each pattern.

## Pick the right tool

| Work | Use | Why |
|---|---|---|
| waiting on the network, disk or subprocesses | threads or asyncio | blocking I/O releases the GIL |
| pure-Python CPU work | processes, or vectorise | on the standard build, threads don't run Python code in parallel |
| NumPy or PyTorch kernels | threads can work | many kernels release the GIL; watch the total thread count |

Oversubscription is easy to miss. Sixteen worker processes, each running BLAS with 16 threads
on a 16-core machine, means 256 threads competing for 16 cores. Set `OMP_NUM_THREADS` (or the
library's equivalent) for each worker.

## Examples

### Which result belongs to which input

```python
paths = ["slow.cif", "fast.cif"]
futs = [pool.submit(run, p) for p in paths]
{p: f.result() for p, f in zip(paths, as_completed(futs))}
# {'slow.cif': 'result for fast.cif', 'fast.cif': 'result for slow.cif'}
```

`as_completed` yields futures in completion order. **Fix:** keep a map from each future to
its input:

```python
fut_to_path = {pool.submit(run, p): p for p in paths}
for f in as_completed(fut_to_path):
    results[fut_to_path[f]] = f.result()
```

`pool.map` keeps input order, but it re-raises the first exception when you reach it, and you
lose the results after it:

```python
for r in pool.map(relax, range(5)):     # item 2 raises
    got.append(r)                       # got == [0, 1], then ValueError
```

### Processes don't share your variables

```python
results = {}

def work(x):
    results[x] = x * x          # updates the child process's copy
    return x * x

with ProcessPoolExecutor() as ex:
    list(ex.map(work, range(4)))     # [0, 1, 4, 9]
results                              # {} in the parent
```

**Fix:** return values from workers and collect them in the parent.

### Read-modify-write races

Two workers each load `cache.json`, add their own key and write the file back. Whichever writes
last wins, and the other key is lost. Threads writing the same file at the same time can also
interleave into invalid JSON. **Fix:** a single writer (fed by a queue), a lock around the whole
read-modify-write, a database that handles concurrency (SQLite with one connection per thread),
or one output file per task, merged at the end.

### Atomic writes

```python
def save_json_atomic(obj, path: Path):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)        # an atomic rename on the same filesystem
```

Writing straight into `path` and crashing halfway leaves truncated JSON. If the loader then does
`except Exception: return {}`, the next save overwrites everything, and the data loss is
silent. Atomic replacement prevents torn files, but not lost updates; those need the fixes in
the previous section.

### Subprocesses

```python
cif = "x.cif; echo INJECTED"
subprocess.run(f"echo processing {cif}", shell=True)   # prints "processing x.cif", then "INJECTED"
subprocess.run(["echo", "processing", cif])            # prints "processing x.cif; echo INJECTED"
```

The exit code is easy to ignore:

```python
r = subprocess.run(cmd, capture_output=True, text=True)   # the run crashed with exit code 3
float(r.stdout.split()[-1])                                # 1.23: its partial output parses fine
```

**Fix:** an argument list (no shell), `check=True`, and `timeout=` (the child is killed and
`TimeoutExpired` is raised). Read the result from a file in a known format rather than from the
last line of stdout.

### Retries

```python
for attempt in range(3):
    try:
        job_id = submit(job)
        break
    except Exception:
        time.sleep(1)
return job_id    # UnboundLocalError: cannot access local variable 'job_id' ...
```

This has four problems:

1. It retries everything, including bugs and 4xx responses that will never succeed.
2. The fixed delay has no backoff or jitter, so many clients retry in lockstep.
3. `submit` isn't idempotent, and a timeout doesn't mean the job wasn't created, so a retry can
   create a duplicate job.
4. When every attempt fails, it falls through to an `UnboundLocalError`.

A version without those problems:

```python
def call_with_retries(fn, *, attempts=4, base_delay=0.5, retry_on=(ConnectionError, TimeoutError)):
    for i in range(attempts):
        try:
            return fn()
        except retry_on:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * 2**i * random.uniform(0.5, 1.5))
```

For side effects, add an idempotency key derived from the job's content, or check for an
existing job before resubmitting. Also check whether the client already retries. The Anthropic
and OpenAI SDKs retry connection errors, 408, 409, 429 and 5xx responses twice by default, so a
wrapper around them multiplies the attempts.

### Timeouts

`requests.get(url)` has no default timeout (`timeout=None`), so it can hang forever. The same
goes for `fut.result()`, `queue.get()`, `proc.wait()` and `lock.acquire()` without a timeout.
In a review, ask what happens if the other side never answers.

### Cache keys

```python
str({"T": 298, "P": 1}) == str({"P": 1, "T": 298})                    # False: order-dependent
json.dumps({"T": 298}) == json.dumps({"T": 298.0})                    # False: 298 vs 298.0
json.dumps({"T": 298, "P": 1}, sort_keys=True) == json.dumps({"P": 1, "T": 298}, sort_keys=True)  # True
```

Keys that aren't canonical cause cache misses. Incomplete keys are worse: a cache keyed on an
ID alone serves stale results after the input file, the parameters or the code version
changes. **Fix:** hash the content and everything the result depends on:

```python
def cache_key(cif_bytes: bytes, params: dict, code_version: str) -> str:
    meta = json.dumps({"params": params, "version": code_version}, sort_keys=True)
    return hashlib.sha256(cif_bytes + b"\0" + meta.encode()).hexdigest()
```

Don't cache failures or NaN. The caching cheat sheet covers the rest: `lru_cache` on methods,
mutable cached values, eviction.

### asyncio

```python
async def fetch_blocking(i):
    time.sleep(0.2)                              # blocks the event loop

async def fetch_offloaded(i):
    await asyncio.to_thread(time.sleep, 0.2)     # runs in a worker thread

# asyncio.gather over 5 calls: 1.02 s blocking, 0.21 s offloaded
```

Any blocking call inside `async def` serialises everything: `requests`, a synchronous SDK
client, file or database I/O, heavy CPU work. Also:

- bound the fan-out with a `Semaphore`;
- keep references to the tasks you create, because unreferenced tasks can be
  garbage-collected;
- don't call `asyncio.run` from library code, because it fails inside a running loop, such as
  Jupyter's.

## Security in ordinary Python code

| You see | Risk | Fix |
|---|---|---|
| `f"... WHERE id = '{x}'"` | SQL injection | parameters: `execute("... WHERE id = ?", (x,))` |
| `shell=True` with interpolated values | command injection (see above) | an argument list |
| `pickle.load`, `joblib.load` or `torch.load(weights_only=False)` on files from elsewhere | arbitrary code execution | trusted sources only; safetensors |
| `yaml.load(..., Loader=yaml.Loader)` or `eval(text)` | code execution | `yaml.safe_load`, `ast.literal_eval`, or JSON |
| `os.path.join(root, user_path)` | `../` escapes the root, and an absolute path replaces it: `os.path.join("/srv/ws", "/etc/passwd")` is `/etc/passwd` | `p = (root / user_path).resolve()`, then require `p.is_relative_to(root.resolve())` |
| secrets in code, URLs, logs or exception messages | leaked credentials | an environment variable or secret store; send keys in headers |
| a server calling `requests.get(url_from_user)` | SSRF: internal services, the cloud metadata endpoint (169.254.169.254) | allowlist hosts; block private addresses |

## Fail loudly

- Validate at the boundary (shapes, units, ranges, required columns), and raise with a message
  that names the bad value.
- Never use a plausible number such as `0.0` or `-1` to mean "failed" in numeric results; it
  gets averaged in. Use `None` or NaN with a status, or raise.
- `logger.exception(...)` inside `except` keeps the traceback, and `print(e)` doesn't. Include
  IDs such as `mof_id` and `job_id` in log lines.
