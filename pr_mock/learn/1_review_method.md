# 1. Reviewing out loud: the method, with a worked example

Besides the bugs you find, the interview scores how you **rank** them and how you **explain**
them. This chapter covers the method; chapters 2–6 cover the bug patterns.

## The 45 minutes

| Minutes | Do | Say, for example |
|---|---|---|
| 0–3 | Ask 3–5 questions. If they don't answer, state your assumptions. | "Who calls this, and how often? How big is the input? Is this research code or shared infrastructure?" |
| 3–8 | Skim the whole file before commenting: entry points, data flow, side effects. | "It reads candidates, scores them and returns the top k. The cache and the sort look riskiest, so I'll start there." |
| 8–35 | Go through it in detail, riskiest parts first. One comment per issue. | "[Blocker] `top_candidates`: …" |
| 35–40 | Re-rank, and pick the three you'd insist on. | "If only three things change before this merges…" |
| 40–45 | Summarise: the top three, the tests you'd ask for, what you'd let go. | "…and I'd let the naming go; it isn't worth another round." |

If the interviewer asks "what about line 40?", that's a hint. Go there and think aloud.

## Where bugs hide

Ask each part of the code:

- **Boundaries.** What comes in: files, user input, API responses, model output? What if it's
  empty, NaN, huge, duplicated or malformed?
- **State.** What outlives one call: default arguments, module globals, caches, class
  attributes, files? Who else reads or writes it?
- **Transformations.** Shapes, units, ordering, joins, splits. Would a mistake raise, or quietly
  produce numbers?
- **Error paths.** Every `except`, retry and fallback value. What does the caller see when it fails?
- **Loops.** Is each one bounded? Is there work inside that could be done once, or vectorised?
- **Concurrency.** What's shared between threads, tasks or processes? Which result belongs to
  which input?
- **Side effects.** Writes, job submissions, emails, deletions. Are they atomic, idempotent and
  authorised?

## Ranking

Rank by how bad the failure is and how likely it is, and put **silent** failures above loud
ones: a crash gets noticed, but a wrong number gets published.

| Tier | Typical examples |
|---|---|
| Blocker | test data leaking into training; results attached to the wrong inputs; a cache serving stale values; a crash mid-write that destroys the results file |
| Major | a crash on realistic input (an empty batch, an unknown element); injection; a 100× slowdown at production size; unbounded memory, loops or cost |
| Minor | a missing timeout on a call that rarely fails; vague error messages; hard-coded paths; `print` instead of logging |
| Nit | naming, comments, style, small refactors. Mention them in one breath and move on. |

## The comment formula

`[severity] where: problem → concrete failure → fix`. Most candidates skip the concrete
failure, and it's the part that makes a comment convincing.

| Weak | Strong |
|---|---|
| "The error handling isn't great." | "[Blocker] `load_cache`: `except Exception: return {}` turns a corrupt file into an empty cache, so we rerun every simulation and then overwrite the file. I'd catch `FileNotFoundError` only and move a corrupt file aside." |
| "This could be vectorised." | "[Major] `featurise`: there's a Python-level lookup per element per row. At 100k structures this loop dominates the runtime. A merge plus a groupby does it in one vectorised pass." |
| "Use better names." | "[Nit] `d` holds distances in Å; `dist_ang` would save the next reader a lookup." |

## Saying "this is fine"

Calling out a red herring shows judgement. Say what it looks like, then why it's safe here:

- "This mutable default is never mutated, so it's a nit at most."
- "`except Exception` is fine here: it's the top-level handler of a worker, and it logs and re-raises."
- "`time.sleep` in this thread worker is fine. It would only be a problem inside `async def`."

When you aren't sure, say what it depends on: "If `predict` returns shape (N, 1), line 30
broadcasts to (N, N); if it returns (N,), it's fine. Either way I'd add a shape assert."

## Code quality and tests, in two minutes

Keep these short, put them after the correctness issues, and tie each one to a bug you found:

- "If load, transform and save were separate functions, the transforms could be unit-tested.
  That's why none of these bugs had a test."
- "Pass the client in instead of creating it at import time; then a test can use a fake."
- "Put units in names and docstrings: `pressure_bar`, `energy_ev`."

Ask for the tests that would have caught your top findings, for example:

- a three-row fixture with a known ranking;
- `assert set(train_ids).isdisjoint(test_ids)` for a split;
- a fake that finishes calls out of order, for anything concurrent;
- a writer killed halfway through, for anything that persists state.

## Worked example

Read the snippet and decide what you'd say before opening the model review. Every behaviour it
describes was checked by running the code with pandas 2.3 and 3.1.

```python
# PR #230: pick the next MOFs to simulate, ranked by predicted CO2 uptake
import pandas as pd

FEATURES = ["pore_diameter", "surface_area", "density"]

def top_candidates(csv_path, model, k=10, cache={}):
    df = pd.read_csv(csv_path)
    df = df[df.surface_area > 500]
    scores = []
    for _, row in df.iterrows():
        if row.mof_id not in cache:
            x = row[FEATURES].values.reshape(1, -1)
            cache[row.mof_id] = model.predict(x)[0]
        scores.append(cache[row.mof_id])
    df["score"] = scores
    return df.sort_values("score").head(k).mof_id.tolist()
```

<details>
<summary>Model review (open after you've tried)</summary>

**Questions.** Is a higher score better? Does `model` change while the process runs (in an
active-learning loop it will)? How many rows: thousands or millions? Can `mof_id` repeat, and
can `surface_area` be missing?

**Skim.** "It reads candidates, filters on surface area, scores each row through a memo
cache, sorts, and returns the top k IDs. The sort, the cache and the per-row predict are the
risky parts."

1. **[Blocker] The sort returns the worst candidates.** `sort_values` is ascending by
   default, so if higher uptake is better, `head(k)` takes the lowest scores. With a toy model
   the function returned `['B', 'C']` when the best two were `['E', 'C']`. Nothing errors, and
   simulation time goes to the worst MOFs. *Fix:* `df.nlargest(k, "score")`, plus a three-row
   test with a known order.
2. **[Blocker] The cache keeps serving the old model's scores.** `cache={}` is created once,
   when `def` runs, and shared by every call. It's keyed only on `mof_id`, so after a retrain
   the function still returns the first model's scores: in the toy run, a retrained model with
   the opposite ranking still produced `['B', 'C']`. It also grows without limit. *Fix:* drop
   the cache, or key it on `(model_version, mof_id)` and give it an explicit owner.
3. **[Major] One `predict` call per row.** Per-call overhead dominates: 100k rows means 100k
   model calls instead of one. *Fix:* `df["score"] = model.predict(df[FEATURES].to_numpy())`.
   A single batched call is fast enough that the cache isn't needed at all.
4. **[Minor] Rows with a missing surface area disappear silently**, because `NaN > 500` is
   False. That may be intended; make it explicit and log how many rows were dropped.
5. **[Minor] A repeated `mof_id` shows up twice in the top k.** Ask whether IDs are unique; if
   they aren't, deduplicate or validate on load.
6. **[Nit] `500` is a magic number with an implicit unit (m²/g).** Make it a parameter such as
   `min_surface_area_m2_g=500`.

**Red herrings.**
- `df = df[...]` followed by `df["score"] = ...` looks like the classic chained-assignment
  bug. It isn't: `df` now names the filtered frame, so the column lands where it should, and
  neither pandas 2.3 nor 3.1 warns.
- `head(k)` with fewer than k rows is fine; it returns what there is.

**Summary (about 40 seconds).** "There are two silent blockers. The ascending sort returns the
worst candidates, and the default-argument cache serves stale scores after every retrain.
Next is performance: one model call per row. Batching fixes the speed and makes the cache
unnecessary. The NaN filter and duplicate IDs are questions as much as bugs. For tests, I'd
want a three-row fixture with a known ranking, and a test that retraining changes the output."

</details>

## Habits that cost points

- **Narrating line by line without ranking.** Rank as you go: "this is the most serious so far".
- **Opening with style.** Spend the first ten minutes on correctness.
- **"I'd rewrite this."** A rewrite isn't a review comment. Name the specific failures.
- **Claiming a bug without a failing input.** If you can't describe one, raise it as a question.
- **Spending fifteen minutes on one issue.** Note it, move on, and come back if there's time.
- **Running out of time before the summary.** Keep the last five minutes for it.
