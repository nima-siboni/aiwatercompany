# Answer key: L2_frozen_structure.py — Dataclasses, level 2 (20 min)

**Pattern focus:** `frozen=True` is shallow. It blocks attribute assignment, but NumPy arrays
and dicts inside the object stay mutable and may be shared with the caller. The generated
`__eq__` also doesn't understand arrays, so a value object with array fields needs explicit
copies and explicit equality.

**Good clarifying questions:** What counts as a duplicate: bit-identical, within a tolerance,
or the same crystal up to atom order and symmetry? Do callers pass views into bigger arrays
(MD trajectories, generator batches) or reuse buffers? Should two structures with different
`energy` or `metadata` be equal? How many structures per batch, and how many atoms each?

## Blockers (silently wrong results / data corruption)

1. **`rattle` mutates the original structure.** `coords = s.frac_coords` is the same array
   object, and `coords += noise` writes into it. Frozen only stops `s.frac_coords = ...`, not
   writes into the array. So `augment([s], n_copies=5)` returns five structures that share
   one array holding the sum of all five noise draws, and `s` itself now carries that noise.
   Its hash changes, so it's also lost from any set or cache it was in, and a `cart_coords`
   value cached earlier is now stale. *Fix:* build a new array
   (`s.frac_coords + noise`) and make the stored arrays read-only (#2), so an in-place write
   raises instead of corrupting data.

2. **`__post_init__` writes into the caller's array and keeps a reference to it.**
   `self.frac_coords[:] = self.frac_coords % 1.0` wraps the caller's data in place.
   (`self.frac_coords %= 1.0` would raise `FrozenInstanceError`, because augmented assignment
   re-assigns the attribute; that's probably why it's written this way.) Build one
   `Structure` per frame of an MD trajectory stored as unwrapped coordinates, and
   `Structure(lat, sp, traj[t])` receives a view, so the trajectory itself gets wrapped and
   a diffusion coefficient computed from `traj` afterwards sees atoms jumping across the cell.
   It also works the other way: if the caller reuses a buffer for the next generator batch,
   every "frozen" structure built from it changes, and so do their hashes. *Fix:* take private
   float copies in `__post_init__`, set `flags.writeable = False` and assign with
   `object.__setattr__`, which is the standard idiom inside a frozen class's `__post_init__`.

## Major

3. **The generated `__eq__` can't compare arrays.** Comparing two `Structure`s either raises
   `ValueError: The truth value of an array ... is ambiguous` or returns an array instead of a
   bool, depending on the Python version and the fields, so `a == b`, `in` and dedupe logic
   break. Here it raises. `deduplicate` passes a unit test with distinct structures (their
   hashes differ, so `__eq__` never runs) and crashes on the first genuine repeat, which is
   the one case it exists for. *Fix:* write `__eq__` and `__hash__` from one key,
   `(species, lattice.tobytes(), frac_coords.tobytes())`, so they can't disagree, and leave
   `energy` and `metadata` out of identity (or the same crystal from two databases never
   matches). If "duplicate" means within a tolerance, that can't be an `__eq__`/`__hash__`
   pair at all; use a structure matcher inside `deduplicate`.

4. **Type hints aren't enforced: `species` from JSON is a list.** `from_dict` passes
   `d["species"]` straight through, so the first time a structure loaded from JSON is hashed
   (by `deduplicate` or a cache), it raises `TypeError: unhashable type: 'list'`.
   *Fix:* `tuple(self.species)` in `__post_init__`.

5. **`dataclasses.replace` is shallow, so the copy shares `metadata`.** `rattle` does
   `out.metadata["rattle_sigma"] = sigma`, and `out.metadata` is the original's dict, so the
   original gets tagged too. If the pipeline later keeps rattled copies out of the test set
   with `"rattle_sigma" not in s.metadata`, the originals disappear as well. *Fix:* pass
   `metadata={**s.metadata, "rattle_sigma": sigma}` to `replace`, and copy `metadata` in
   `__post_init__` so the caller's dict isn't shared either.

## Minor

6. **The hash depends on dtype.** `from_dict` keeps whatever dtype the JSON implies. A lattice
   written as integers (`[[10, 0, 0], ...]`) becomes int64, whose bytes differ from the
   float64 lattice of the same crystal, so `deduplicate` keeps both. *Fix:* convert with
   `dtype=float` in `__post_init__`.

7. **Validation assumes ndarrays.** `Structure([[10.0, 0, 0], ...], ("Zn",), [[0.1, 0.2, 0.3]])`
   raises `AttributeError: 'list' object has no attribute 'shape'` instead of converting.
   *Fix:* convert first, then validate the shapes.

8. **Hashing re-serialises the arrays on every call.** `__hash__` calls `tobytes()` on both
   arrays (a full copy) each time, and `deduplicate` hashes every structure twice (`in`, then
   `add`). It's microseconds per call, but it adds up over millions of generated samples.
   Once the arrays are read-only, compute the key once (`cached_property`). The defensive copy
   in the fix also has a cost: one array copy per construction, including every `replace()`.
   That's the right trade at these sizes.

## Red herrings (say why they're fine)
- **`with_energy` uses `dataclasses.replace`.** That's the right way to "modify" a frozen
  object: it goes through `__init__`, so `__post_init__` validation runs again, and sharing
  the arrays between the old and new objects is fine once they're read-only.
- **`@cached_property` on a frozen dataclass.** It works, because it stores the value straight
  into the instance `__dict__` rather than going through the frozen `__setattr__` (it does
  fail with `slots=True`). It's safe once the arrays can't change; under Blocker 1 it goes
  stale. Return a read-only array so callers can't edit the cached value.

## Model 1-minute summary
"The headline is that `frozen=True` is shallow. `rattle` adds noise to `s.frac_coords` in
place, so augmentation returns identical copies, corrupts the originals and changes their
hashes. And `__post_init__` writes the wrapped coordinates into the caller's array, so
building structures from an MD trajectory silently wraps the trajectory itself. Both are
fixed by taking read-only float copies in `__post_init__` and assigning them with
`object.__setattr__`. Next, the generated `__eq__` can't compare arrays: `deduplicate` passes
tests with distinct structures and raises ValueError on the first real duplicate. I'd write
`__eq__` and `__hash__` from one key and agree on what identity means. Then robustness:
`species` from JSON is a list, so hashing fails; `replace` shares the metadata dict, so
tagging a copy tags the original; and integer lattices hash differently. `with_energy` and
`cached_property` are fine."

## Reference fix
`L2_frozen_structure_fixed.py`:
- `__post_init__` converts inputs to read-only float64 copies (lists, ints and views are all
  handled), turns `species` into a tuple and copies `metadata`, all via `object.__setattr__`.
- `__eq__` and `__hash__` share one cached key (species plus array bytes); `energy` and
  `metadata` aren't part of identity.
- `rattle` builds a new coordinate array and a new metadata dict and passes both to `replace`.
- `cart_coords` returns a read-only array, so the cached value can't be edited.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_07_dataclasses.py -k L2 -v`
