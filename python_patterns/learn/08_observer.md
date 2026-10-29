# Observer, callbacks and pub/sub: learning nuggets

A publisher announces events (a job finished, an epoch ended) without knowing who listens.
Subscribers register callbacks, and the publisher calls them. The bugs sit in delivery: what
happens when a handler fails, changes the subscriber list, or runs on another thread.

The examples are written for these notes, not taken from the drills, and the outputs in comments
come from running them on Python 3.14. After the drills, the
[cheat sheet](../answers/08_observer/CHECKLIST.md) is the full checklist.

The examples use this minimal bus:

```python
class Bus:
    def __init__(self):
        self.subs = defaultdict(list)

    def subscribe(self, topic, fn):
        self.subs[topic].append(fn)

    def publish(self, topic, event):
        for fn in self.subs[topic]:
            fn(event)
```

## 1. Handlers defined in a loop see the last loop value

```python
bus = Bus()
for topic in ["started", "finished"]:
    bus.subscribe(topic, lambda e: print(f"[{topic}] {e}"))

bus.publish("started", "job-1")              # prints "[finished] job-1"
```

**Fix:** bind the value now, with `lambda e, topic=topic: ...` or `functools.partial`.

## 2. One failing handler stops delivery

```python
bus.subscribe("done", broken)                # raises KeyError
bus.subscribe("done", notify)
bus.publish("done", "job-1")                 # the KeyError reaches the publisher; notify never runs
```

**Fix:** wrap each handler call in `try/except` and log with `logger.exception`, so one bad
subscriber can neither silence the others nor crash the publisher. Not `except Exception: pass`:
that hides the failure.

## 3. Unsubscribing during delivery skips the next handler

```python
handlers = []

def once(e):
    print("once", e)
    handlers.remove(once)                    # a one-shot handler removes itself

def audit(e):
    print("audit", e)

handlers += [once, audit]
for fn in handlers:
    fn("job-1")                              # prints only "once job-1": audit was skipped
```

Removing an item shifts the list under the loop. With a dict of handlers you get
`RuntimeError: dictionary changed size during iteration` instead. **Fix:** iterate over a
snapshot, `for fn in list(handlers):`.

## 4. Weak references and bound methods

```python
r = weakref.ref(listener.on_event)
r()                                          # None: the bound method was a temporary object
weakref.WeakMethod(listener.on_event)()      # works while `listener` is alive

listener.on_event is listener.on_event       # False: every access creates a new bound method
listener.on_event == listener.on_event       # True
```

A subscriber list built with `weakref.ref` never calls anyone, and unsubscribing with `is` never
matches. **Fix:** `weakref.WeakMethod` for methods, and compare with `==` (which `list.remove`
already does).

## 5. One mutable payload shared by every subscriber

```python
event = {"job": "j1", "status": "done"}

def archive(e):
    e["status"] = "archived"

def report(e):
    print("report sees", e["status"])

for fn in (archive, report):
    fn(event)                                # report sees archived
```

**Fix:** frozen dataclass events, or a copy per subscriber.

## 6. Per-run state set in `__init__`

```python
class EarlyStop:
    def __init__(self, patience=2):
        self.patience, self.best, self.bad, self.stop = patience, float("inf"), 0, False

    def on_epoch_end(self, loss):
        if loss < self.best:
            self.best, self.bad = loss, 0
        else:
            self.bad += 1
            self.stop = self.bad >= self.patience
```

Reuse one instance for a second training run, and it compares the new losses (0.9, 0.7, 0.5,
…) with the first run's best (0.1). It stopped the second run after 2 epochs, while that run was
still improving. **Fix:** reset per-run state in a start hook (`on_train_begin`), not in
`__init__`.

## 7. Handlers run on the publisher's thread

A handler that uses a SQLite connection created on another thread fails with
`ProgrammingError: SQLite objects created in a thread can only be used in that same thread`, and a
slow handler blocks the publisher. **Fix:** keep handlers fast, and have them pass work to a
queue owned by the right thread.

Two more to look for: a subscriber list or counter changed from several threads needs a lock, and
a handler that publishes during delivery reorders events (queue such publishes instead).

## Usually fine (don't flag it)

- Unsubscribing a bound method with `==` or `list.remove`: bound methods compare equal.
- No-op hooks in a base class: an empty method call costs nanoseconds.
- `deque(maxlen=N)` appended to from several threads: bounded and thread-safe.
