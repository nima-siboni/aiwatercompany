# Answer key: L1_event_bus.py — Observer, level 1 (10 min)

**Pattern focus:** the canonical observer pitfalls. Closures in a loop bind late, a
subscriber registry stored as a class attribute is shared by every instance, removing a
subscriber while the bus is iterating skips the next one, and one failing subscriber must
not stop delivery to the rest.

**Good clarifying questions:** Is there one bus per process, or one per campaign (and a
fresh one in each test)? Should the job runner find out when a handler fails? Are topics ever
per job (`job.<id>.…`)? Does the order in which handlers run matter?

## Blockers (silently wrong results / lost data)

1. **Late binding in `JobStateCounter`.** `on_event` closes over the variable `state`, not its
   value at definition time. The closure reads it when an event arrives, after the loop has
   finished, so it is always `"failed"`. Publish one `job.queued` and one `job.running` and
   the counts are `Counter({"failed": 2})`: the dashboard shows every job as failed. A
   `lambda` in the loop would behave the same way. *Fix:* bind the value, e.g.
   `bus.subscribe(f"job.{state}", functools.partial(self._count, state))` (a default argument
   `state=state` or a small factory function also works).

2. **`_subscribers` is a class attribute, so every bus shares it.**
   `self._subscribers[topic].append(...)` mutates the class-level dict and never creates an
   instance attribute. Two buses (one per campaign, or a fresh bus in each test) deliver each
   other's events: a `ResultsWriter` on bus A stores uptakes published on bus B. Handlers also
   pile up across tests in the same process. *Fix:* create the dict in `__init__`.

3. **`once` changes the list while `publish` iterates it.** `publish` loops over
   `self._subscribers[topic]`, and `wrapper` removes itself from that same list. The list
   shifts left, and the loop skips the handler that came after it. Register a `once`
   notification before the `ResultsWriter` and the writer never sees that event, so the
   uptake is lost without an error. *Fix:* iterate over a snapshot,
   `for handler in list(...)`.

## Major

4. **One failing handler stops delivery and crashes the publisher.** Nothing isolates
   handlers. If the dashboard handler raises, every handler after it (for example the
   results writer) misses `job.finished`, and the exception propagates into the job runner
   in the middle of a state change. *Fix:* wrap each call in `try/except Exception`, log with
   `logger.exception` (handler and topic), and carry on. Ask whether the publisher needs to
   know about failures (a returned list, or a dead-letter log).

## Minor

5. **`publish` grows the topic table forever.** `self._subscribers[topic]` on a
   `defaultdict` inserts an empty list for every topic that is published without
   subscribers. With per-job topics (`job.<id>.heartbeat`) a long-running service leaks one
   entry per job. *Fix:* `self._subscribers.get(topic, ())`.

## Nit
- `unsubscribe` for a handler that isn't subscribed raises `ValueError` from `list.remove`.
  Decide whether that's an error or a no-op, and document it.

## Red herring (say why it's fine)
- **`unsubscribe("job.finished", self.on_finished)` in `ResultsWriter.close`.** Each
  attribute access creates a new bound-method object (`w.on_finished is w.on_finished` is
  False), so it looks as if `remove` can't find the original. But bound methods compare
  equal when they wrap the same function and the same instance, and `list.remove` uses `==`,
  so it works.

## Model 1-minute summary
"Three issues give silently wrong results. In `JobStateCounter`, the handler closes over the
loop variable `state`, so by the time events arrive every one is counted as 'failed'. I'd
bind the value with `functools.partial`. `_subscribers` is a class attribute, so every
`EventBus` shares one dict, and a second bus, for another campaign or in a test, delivers
its events to the first bus's subscribers. It should be created in `__init__`. And `once()`
removes its wrapper while `publish` is iterating the same list, so the next handler silently
misses the event, which can lose a result. Iterating over a copy fixes that. Next, a single
failing handler stops delivery to everyone after it and crashes the publisher, so each
handler needs its own try/except with logging. Minor: `publish` creates an empty entry for
every unknown topic. Unsubscribing a bound method is fine."

## Reference fix
`L1_event_bus_fixed.py`:
- `_subscribers` is created per instance in `__init__`.
- `publish` iterates over a snapshot, uses `.get` so unknown topics don't create entries,
  and isolates each handler with `try/except` and `logger.exception`.
- `JobStateCounter` binds each state with `functools.partial`.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_08_observer.py -k L1 -v`
