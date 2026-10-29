# Checklist: Observer, callbacks, hooks and pub/sub

The observer pattern lets a publisher announce events (job finished, epoch ended) without
knowing who listens. Subscribers register callbacks, and the publisher calls them on delivery.

**When you see it in a PR, check:**
- Callbacks defined in a loop that use the loop variable: late binding, every handler sees the last value.
- A registry stored as a class attribute or module global: every instance shares subscribers.
- Subscribing or unsubscribing during delivery: lists skip the next handler, dicts raise `RuntimeError`.
- No per-subscriber `try/except`: one failure stops delivery and crashes the publisher.
- `except Exception: pass` around callbacks: failures vanish. Isolate *and* log.
- `weakref.ref(obj.method)`: dead at once, so the handler never fires. Use `WeakMethod`.
- Unsubscribing with `is`: bound methods are new objects on each access, so it never matches.
- One mutable payload shared by all subscribers: edits leak to later ones, stored refs change later.
- Callbacks run on the emitter's thread: thread-affine resources (SQLite) break, slow ones block it.
- A registry or subscriber counter mutated from several threads without a lock.
- A callback that emits again: nested delivery reorders events. Queue re-entrant emits.
- Per-run state (best score, patience, stop flag) set in `__init__` instead of a start hook.
- Heavy or non-atomic I/O in a high-frequency hook (per batch, per heartbeat).
- `defaultdict` lookups in `publish`: every unknown topic leaves an entry behind.

**Usually fine (don't block on it):**
- Unsubscribing a bound method with `==` or `list.remove`: bound methods compare equal.
- No-op base-class hooks: an empty method call costs nanoseconds.
- `field(default_factory=time.time)` for an event timestamp.
- `deque(maxlen=N)` appended to from several threads: bounded and thread-safe.

**Idiomatic shape:**
```python
class EventBus:
    def __init__(self):
        self._subs: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
    def subscribe(self, topic, fn):
        with self._lock:
            self._subs[topic].append(fn)
    def publish(self, topic, event):  # event: a frozen dataclass
        with self._lock:
            handlers = list(self._subs.get(topic, ()))  # snapshot; call outside the lock
        for fn in handlers:
            try:
                fn(event)
            except Exception:
                logger.exception("handler %r failed on %s", fn, topic)
```
