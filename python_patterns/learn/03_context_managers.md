# Context managers: learning nuggets

A context manager pairs setup with guaranteed teardown: files, locks, transactions, temporary
state. The bugs hide in what happens when the block raises, so read each one twice: once for
success, once for failure.

The examples are written for these notes, not taken from the drills, and the outputs in comments
come from running them on Python 3.14. After the drills, the
[cheat sheet](../answers/03_context_managers/CHECKLIST.md) is the full checklist.

## 1. `__exit__` returning True swallows the exception

```python
class step:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print("start", self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        print("end", self.name)
        return True                        # means "exception handled"

with step("relax"):
    1 / 0
print("carried on")                        # runs: the ZeroDivisionError is gone
```

A truthy return from `__exit__` suppresses every exception, Ctrl-C included. **Fix:** return
`None` (or nothing), unless suppressing one specific exception is the whole point.

## 2. `@contextmanager` without `try/finally` skips the cleanup on failure

```python
@contextmanager
def env_var(name, value):
    old = os.environ.get(name)
    os.environ[name] = value
    yield
    if old is None:                        # never reached if the block raises
        del os.environ[name]
    else:
        os.environ[name] = old

with env_var("OMP_NUM_THREADS", "1"):
    raise RuntimeError("job failed")
os.environ.get("OMP_NUM_THREADS")          # '1': the temporary value stayed
```

**Fix:** put the `yield` inside `try:`, and the restore in `finally:`.

## 3. Commit on success, not in `finally`

```python
@contextmanager
def transaction(conn):
    try:
        yield conn
    finally:
        conn.commit()                      # also commits when the block failed

with transaction(conn):
    conn.execute("insert into results values ('MOF-5', 4.1)")
    raise ValueError("second row failed validation")

conn.execute("select count(*) from results").fetchone()    # (1,): half a batch, made permanent
```

`os.replace(tmp, path)` in a `finally` has the same problem: a half-written file becomes the real
one. **Fix:**

```python
    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()
```

## 4. Several resources in `__enter__`: use an `ExitStack`

```python
class Scratch:
    def __enter__(self):
        self.lock = self.root / "LOCK"
        self.lock.touch()                                            # step 1 succeeds
        self.tmp = Path(tempfile.mkdtemp(dir=self.root / "missing"))  # step 2 raises
        return self

    def __exit__(self, *exc):
        self.lock.unlink()
```

If `__enter__` raises, `__exit__` never runs, so the lock file from step 1 stays behind and the
next job waits for it forever. **Fix:** register each undo step as you go, and keep the list only
once setup has finished:

```python
    def __enter__(self):
        with ExitStack() as stack:
            self.lock = self.root / "LOCK"
            self.lock.touch()
            stack.callback(self.lock.unlink)
            self.tmp = Path(tempfile.mkdtemp(dir=self.root / "missing"))
            stack.callback(os.rmdir, self.tmp)
            self._stack = stack.pop_all()        # setup succeeded: keep the undo list
        return self

    def __exit__(self, *exc):
        self._stack.close()
```

With this version, the lock file was gone after the failed `with`.

## 5. Library context managers that don't do what you'd guess

```python
with ThreadPoolExecutor() as ex:
    for _ in range(3):
        ex.submit(time.sleep, 0.5)
    raise KeyError("bad config")           # surfaced only after 0.5 s
```

Leaving the block waits for every submitted task before the exception propagates. With hour-long
jobs, a config error looks like a hang. Validate before submitting, or pass `cancel_futures=True`
to `shutdown` so queued tasks are dropped. Also, `with sqlite3.connect(...)` commits or rolls
back but never closes the connection (see
[Python pitfalls](../../pr_mock/learn/2_python_pitfalls.md)).

## 6. A `with` block can't be retried from inside the context manager

```python
@contextmanager
def retrying(attempts=3):
    for _ in range(attempts):
        try:
            yield
            return
        except ConnectionError:
            continue                       # tries to run the block again

with retrying():
    raise ConnectionError("flaky")
# RuntimeError: generator didn't stop after throw()
```

A `with` body runs exactly once. **Fix:** retry around the whole block, with a loop or a
retrying function.

## 7. Process-wide state behind a per-object API

`os.chdir`, `os.environ` and `sys.stdout` belong to the whole process. A context manager that
changes them is fine in a script. In threads or async code, every other task sees the change,
and relative paths in another thread resolve against the wrong directory. **Fix:** pass `cwd=`
and `env=` to `subprocess.run`, and use absolute paths.

## Usually fine (don't flag it)

- `__exit__` that ignores the exception and returns `None`: the cleanup runs and the exception
  propagates.
- `return` or `break` inside a `with` block: `__exit__` still runs. `with a, b:` exits in reverse
  order.
- `yield` inside `with open(...)` in a generator: the file closes when the generator finishes or is
  closed.
- `contextlib.suppress(FileNotFoundError)` around one specific call.
