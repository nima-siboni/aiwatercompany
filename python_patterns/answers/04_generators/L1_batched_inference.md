# Answer key: L1_batched_inference.py — Generators, level 1 (10 min)

**Pattern focus:** the canonical generator pitfalls. An iterator can be consumed only once, a
hand-written batcher must flush its tail, and `zip` stops at the shorter input without telling
you. All three fail silently, and all three pass a unit test that uses a short list.

**Good clarifying questions:** Does the model always return exactly one prediction per
structure, in order, or can it drop structures it can't featurise? Which Python version do we
run (`zip(strict=True)` needs 3.10, `itertools.batched` 3.12)? Are the JSONL files ever
concatenated or edited by hand?

## Blockers (silently wrong results / data loss)

1. **`structures` is iterated twice, so the documented usage returns `{}`.**
   `mof_ids = [s["mof_id"] for s in structures]` exhausts the generator from `read_structures`.
   `batched(structures, ...)` then sees an empty iterator, `preds` stays empty, and
   `dict(zip(mof_ids, []))` is `{}`. With a list, which is what a unit test usually passes, it
   works, so the tests pass and the real run silently scores nothing. *Fix:* one pass: pair the
   ids with the predictions inside the batch loop. Don't "fix" it with `list(structures)`, which
   throws away the streaming the PR exists for.

2. **`batched` drops the final partial batch.** Items left in `batch` when the loop ends are
   never yielded. 200,000 structures in batches of 256 is 781 full batches plus 64, so the last 64
   MOFs are never scored; an input shorter than one batch scores nothing at all. Nothing errors,
   because `zip` hides it (#3). *Fix:* `if batch: yield batch` after the loop, or
   `itertools.batched` on 3.12+ (it yields tuples).

## Major

3. **`zip` truncates silently, which hides #2 and any model-side mismatch.** `preds` is one flat
   list for the whole run, so if the model ever returns fewer predictions than inputs (it skips
   a structure it can't featurise), every later MOF in the run gets its neighbour's prediction:
   with `MOF-1` dropped, `MOF-1` is reported with `MOF-2`'s uptake. *Fix:* pair per batch with
   `zip(batch, model(batch), strict=True)`, which raises `ValueError` on a length mismatch.

## Minor

4. **One blank or malformed line kills the run, with an unhelpful error.** A blank line (a stray
   empty line from a hand edit or a concatenation) gives `JSONDecodeError: Expecting value: line 2 column 1`.
   That "line 2" counts lines inside the string being parsed, not the file, and the message has no
   file name. And because `predict_uptake` only returns at the end, every prediction made so far
   is lost too. *Fix:* skip blank lines, and re-raise decode errors as
   `ValueError(f"{path}:{lineno}: ...")`.

5. **`batched(items, 0)` silently returns nothing.** `len(batch) == 0` is never true after an
   append, so the whole input is consumed into one list, which #2 then drops; a negative size does
   the same. *Fix:* raise `ValueError` for `size < 1`, as `itertools.batched` does. Note that in a
   generator function the check runs at the first `next()`, not when `batched()` is called.

## Red herring (say why it's fine)
- **`batch = []` after `yield batch` isn't wasteful; it's required.** Reusing the list with
  `batch.clear()` would empty the batch the caller is still holding:
  `list(batched(range(4), 2))` would give `[[], []]` (I checked). A new list per batch is correct.

## Model 1-minute summary
"Two blockers, both silent. First, `structures` is consumed twice: building `mof_ids` exhausts
the generator from `read_structures`, so the documented call returns an empty dict, while tests
that pass a list still pass. Second, `batched` never yields its last partial batch, so with 200k
MOFs the last 64 are never scored. `zip` hides both, because it truncates to the shorter input,
and if the model ever drops a structure, every later MOF gets its neighbour's prediction. The fix
is one pass that pairs each batch with its predictions using `zip(strict=True)`, plus a tail
flush, or `itertools.batched` on 3.12. Smaller: validate the batch size, skip blank lines, and put
the file and line number in parse errors. I'd add a test that feeds a generator, not a list, with
a length that isn't a multiple of the batch size."

## Reference fix
`L1_batched_inference_fixed.py`:
- `batched` flushes the tail and rejects `size < 1`.
- `predict_uptake` makes a single pass and pairs each batch with its predictions using `zip(..., strict=True)`.
- `read_structures` skips blank lines and reports `path:line` for invalid JSON.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_04_generators.py -k L1 -v`
