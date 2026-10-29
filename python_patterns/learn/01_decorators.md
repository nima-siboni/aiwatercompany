# Decorators: learning nuggets

A decorator replaces a function with a wrapper, so whatever the wrapper gets wrong, every call
site gets wrong. Read each wrapper twice: once for a call that succeeds, once for one that raises.

The examples are written for these notes, not taken from the drills, and the outputs in comments
come from running them on Python 3.14. After the drills, the
[cheat sheet](../answers/01_decorators/CHECKLIST.md) is the full checklist.

## 1. The wrapper must return the result

```python
def log_calls(func):
    def wrapper(*args, **kwargs):
        print(f"calling {func.__name__}")
        func(*args, **kwargs)                  # the result is computed, then dropped
    return wrapper

@log_calls
def cell_volume(a, b, c):
    return a * b * c

cell_volume(2.0, 3.0, 4.0)                              # None
np.array([cell_volume(2.0, 3.0, 4.0)], dtype=float)    # array([nan]): the None became a NaN
```

**Fix:** `return func(*args, **kwargs)`.

## 2. Without `functools.wraps`, every function is called "wrapper"

```python
@log_calls
def relax(): ...

@log_calls
def anneal(): ...

relax.__name__, anneal.__name__                 # ('wrapper', 'wrapper')
steps = {f.__name__: f for f in (relax, anneal)}
steps.keys()                                    # dict_keys(['wrapper']): relax silently disappeared
```

Registries, metrics and logs keyed by name collide. **Fix:** `@functools.wraps(func)` on the
wrapper.

## 3. Bookkeeping belongs in `finally`

```python
in_flight = 0

def track(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global in_flight
        in_flight += 1
        result = func(*args, **kwargs)
        in_flight -= 1                          # skipped when func raises
        return result
    return wrapper

@track
def submit(job):
    if job == "bad":
        raise ValueError("invalid job")
    return f"queued {job}"

submit("a")
submit("bad")          # raises, as it should
in_flight              # 1, although nothing is running
```

Counters, timers, released slots: anything that must happen after the call is skipped exactly
when the call fails. A concurrency limit built on this counter would eventually block everything.
**Fix:** `try: return func(*args, **kwargs)`, then `finally: in_flight -= 1`.

## 4. A decorator factory used without `()`

```python
def check_finite(label="result"):               # a factory: check_finite() returns the decorator
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            if not math.isfinite(out):
                raise ValueError(f"{label} is not finite: {out}")
            return out
        return wrapper
    return decorator

@check_finite                                   # should be @check_finite()
def per_atom(total_energy):
    return total_energy / 52

per_atom(-310.2)    # <function check_finite.<locals>.decorator.<locals>.wrapper at 0x...>
```

`per_atom` is now the inner `decorator`, so calling it returns a function instead of a number,
and nothing errors. **Fix:** make the factory's parameters keyword-only,
`def check_finite(*, label="result")`. Then the bare form fails at import:
`TypeError: check_finite() takes 0 positional arguments but 1 was given`.

## 5. State inside the decorator is shared by every caller

```python
def once(func):
    done = False
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal done
        if not done:
            done = True
            return func(*args, **kwargs)
    return wrapper

class Calculator:
    def __init__(self, name):
        self.name = name

    @once
    def setup(self):
        print("setting up", self.name)

Calculator("first").setup()       # setting up first
Calculator("second").setup()      # prints nothing: the second calculator is never set up
```

The decorator runs once, when the class body runs, so `done` belongs to the class, not to each
instance. Counters, backoff state and rate limits kept in a decorator behave the same way:
they're shared by every call, every thread and every instance. **Fix:** keep per-instance
state on `self`.

## 6. A plain wrapper around an `async def` does nothing

```python
def safe(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)        # only creates the coroutine; nothing has run yet
        except ConnectionError:
            return None
    return wrapper

@safe
async def fetch_props(mof_id):
    raise ConnectionError("service down")

asyncio.run(fetch_props("MOF-5"))               # ConnectionError: service down
```

The error happens later, when the coroutine is awaited, outside the `try`. Timing, retries and
limits written this way fail the same way, and `time.sleep` in such a wrapper blocks the event
loop. **Fix:** an `async def wrapper` that does `return await func(*args, **kwargs)`, chosen with
`inspect.iscoroutinefunction(func)`.

## 7. Stacking order

```python
# tag(name) prints "enter <name>" before the call and "exit <name>" after it
@tag("outer")
@tag("inner")
def run():
    print("body")

run()    # enter outer, enter inner, body, exit inner, exit outer
```

Decorators are applied bottom-up and run top-down. With `@retry` above `@rate_limited`, every
attempt goes through the limiter; the other way round, retries bypass it. Decide where timing
goes too: inside the retry it measures one attempt, outside it the whole sequence.

## Usually fine (don't flag it)

- A plain (non-factory) decorator used without parentheses.
- `@register` above another decorator: the registry then holds the wrapped function, which is
  usually what you want.
- Retrying idempotent reads on 429 or 5xx responses.
- Work done once at decoration time, such as inspecting a signature or compiling a regex. That's
  the right place for it; the bug is doing it on every call.
