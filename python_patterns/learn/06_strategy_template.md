# Strategy and template method: learning nuggets

Strategy swaps one algorithm for another behind a shared interface (acquisition functions,
optimisers, calculators). Template method fixes the skeleton of an algorithm in a base class and
lets subclasses fill in the steps. Both only work if every implementation keeps the same
contract, so the review is about that contract, and about who owns state.

The examples are written for these notes, not taken from the drills, and the outputs in comments
come from running them on Python 3.14. After the drills, the
[cheat sheet](../answers/06_strategy_template/CHECKLIST.md) is the full checklist.

## 1. Make the base class abstract

```python
class Scorer:
    def score(self, candidate):
        return NotImplemented

class UptakeScorer(Scorer):
    def scroe(self, candidate):               # misspelt override
        return candidate["uptake"]

UptakeScorer().score({"uptake": 4.1})        # NotImplemented, silently
```

Whatever uses that result fails later, far from the typo. On Python 3.14 even `if result:`
raises `TypeError: NotImplemented should not be used in a boolean context`. **Fix:**
`class Scorer(ABC)` with `@abstractmethod` on `score`. Then `UptakeScorer()` fails at once:
`TypeError: Can't instantiate abstract class UptakeScorer without an implementation for abstract
method 'score'`.

## 2. Every strategy must mean the same thing by its return value

```python
cands = [{"id": "A", "energy_ev": -1.0}, {"id": "B", "energy_ev": -2.0}]

class Stability:
    def score(self, c):
        return c["energy_ev"]                # lower energy is better...

max(cands, key=Stability().score)["id"]      # 'A': ...but the caller takes the highest score
```

Direction (is higher better?), units (eV or kJ/mol) and normalisation (per atom or per cell) must
match across strategies, or results are silently wrong. **Fix:** state the contract in the base
class ("higher is better, eV per atom") and convert inside each strategy:
`return -c["energy_ev"]` picks `'B'`.

## 3. Mutable state in the class body is shared

```python
class Optimiser:
    history = []                             # one list for every instance

    def step(self, x):
        self.history.append(x)

a, b = Optimiser(), Optimiser()
a.step(1.0)
b.history                                    # [1.0]
```

**Fix:** create state in `__init__`, and reset it at the start of each run if objects are
reused.

## 4. Don't call hooks from the base `__init__`

```python
class Base:
    def __init__(self):
        self.setup()                         # runs before Child.__init__ has finished

class Child(Base):
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff

    def setup(self):
        self.skin = self.cutoff * 0.1

Child(6.0)    # AttributeError: 'Child' object has no attribute 'cutoff'
```

It also runs the hook once per object instead of once per run. **Fix:** call hooks from the
template method (`run()`), not from `__init__`.

## 5. Hooks must not modify their inputs

```python
def step(positions, forces):
    positions += 0.01 * forces              # in place
    return positions

x0 = np.zeros(3)
step(x0, np.ones(3))
x0                                          # [0.01 0.01 0.01]: the caller's array changed
```

**Fix:** `positions = positions + 0.01 * forces`, or have the template copy its inputs once
(`np.array(x, dtype=float)`) before calling any hook.

## 6. Fallback chains that catch everything

```python
class NotAvailable(Exception): ...

def band_gap(mof_id):
    for method in (from_cache, from_db, estimate):
        try:
            return method(mof_id)
        except Exception:
            continue

def from_cache(mof_id):
    raise NotAvailable("not cached")

def from_db(mof_id):
    row = {"band_gap_ev": 1.2}
    return row["bandgap_ev"]                # a typo: KeyError

def estimate(mof_id):
    return 3.0

band_gap("MOF-5")       # 3.0, the crude estimate, instead of 1.2 from the database
```

`except Exception` turns a bug in one method into a silent fallback to a worse one. **Fix:**
catch only the "not available" error the methods are meant to raise (`except NotAvailable:`).
Then the `KeyError` surfaces. Catching `LookupError` wouldn't help, because `KeyError` is a
`LookupError`. Also log which method answered, and check that every method returns the same
units.

## 7. In a mixin chain, every `__init__` must call `super().__init__(**kwargs)`

```python
class LoggingMixin:
    def __init__(self, **kwargs):
        self.log = []                        # no super().__init__(**kwargs)

class Relaxer:
    def __init__(self, steps=100, **kwargs):
        self.steps = steps

class LoggedRelaxer(LoggingMixin, Relaxer):
    pass

LoggedRelaxer(steps=5).steps   # AttributeError: 'LoggedRelaxer' object has no attribute 'steps'
```

`Relaxer.__init__` never ran, because the mixin didn't pass control along the MRO. **Fix:**
`super().__init__(**kwargs)` in every `__init__` in the chain.

## Usually fine (don't flag it)

- `super().method()` inside a mixin with no base class: it follows the instance's MRO.
- Plain functions instead of strategy classes when there's no state or configuration.
- A retry loop whose last attempt re-raises: it can't fall through and return `None`.
