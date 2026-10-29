# Context managers: review checklist

**What it's for:** pairing setup with guaranteed teardown (files, locks, transactions, temporary state),
so the release happens on every path. The bugs hide in what happens when the block raises, so read
each one twice: once for success, once for failure.

**When you see it in a PR, check:**
- **What `__exit__` returns**, or whether a generator CM catches without re-raising. Either way the error is "handled": a stray `return True` swallows every exception, Ctrl-C included → silently wrong results.
- **`@contextmanager` without `try/finally` around the `yield`** → cleanup is skipped exactly when the block raises (cwd or env not restored, lock never released).
- **Commit or publish in `finally`** (`commit()`, `os.replace(tmp, path)`) → a partial transaction or half-written file becomes permanent. Commit on success only; roll back or delete on failure.
- **A generator CM that catches the exception and yields again** (a "retry" CM) → `RuntimeError: generator didn't stop after throw()`. A `with` block can't be re-run; retry around the whole block.
- **Several resources acquired in `__enter__`** → if step 2 fails, `__exit__` never runs and step 1 leaks (stale lock, temp dir). Use an `ExitStack` and `pop_all()`.
- **Cleanup that can raise** (`rmtree`, `unlink`, `close` on a network FS) → replaces the original exception and skips the cleanups after it.
- **Process-global state behind a per-object API** (`os.chdir`, `os.environ`, `sys.stdout`) → wrong results under threads. Pass `cwd=`/`env=` instead.
- **Library CMs that don't do what you'd guess:** `with sqlite3.connect()` commits or rolls back but never closes; `with ThreadPoolExecutor()` waits for every submitted task before re-raising.
- **Atomic writes:** a unique temp file in the *target* directory (`mkstemp`), `flush` + `fsync`, `os.replace` only on success, `unlink` on failure → otherwise cross-device errors, clobbering, or truncated files.
- **Locks:** atomic acquisition (`O_CREAT | O_EXCL`, `flock`); what happens after a crash (stale lock → job stuck forever); re-entrancy (`Lock` deadlocks where `RLock` wouldn't).
- **Cleanup in `__del__`** → runs at arbitrary times, including for objects whose `__enter__` failed. Never put destructive cleanup there.
- **Destructive cleanup of something the CM didn't create** (`mkdir(exist_ok=True)` + `rmtree`, a root that defaults to `.`) → data loss.
- **Resources opened in `__init__` instead of `__enter__`** → leaked when the object is never used in a `with`.

**Usually fine (don't block on it):**
- `__exit__` that ignores `exc_type` and returns `None`: cleanup either way, and the exception propagates.
- `return` or `break` inside a `with` block (`__exit__` still runs); `with a, b:` (exits in reverse order).
- `yield` inside `with open(...)` in a generator (the file closes when the generator finishes or is closed); `contextlib.suppress(FileNotFoundError)` around one specific call.

**Idiomatic shape:**
```python
@contextmanager
def transaction(conn):
    try:
        yield conn
    except BaseException:
        conn.rollback()                  # undo on any failure, then re-raise
        raise
    else:
        conn.commit()                    # permanent only on success

def __enter__(self):                     # several resources: each step registers its own undo
    with ExitStack() as stack:
        stack.enter_context(self._lock())
        self.path = Path(mkdtemp(dir=root)); stack.callback(shutil.rmtree, self.path)
        self._stack = stack.pop_all()    # setup succeeded: __exit__ unwinds self._stack
    return self
```
