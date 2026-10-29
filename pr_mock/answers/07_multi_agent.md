# Answer key: 07_multi_agent.py

**Good clarifying questions:** How many candidates per round, and what's the budget? What
does `predict_properties` do with an unknown or invented MOF name, and what types does it
return (numpy floats?)? Is the critic meant to check against the brief? Do we have a ground
truth to verify against (simulation, database)?

## Blockers

1. **The approval check passes on rejections.** `"APPROVED" in verdict.upper()` is True for
   "NOT APPROVED", "would be approved if…" and "cannot be APPROVED until…". The
   campaign stops at the first critical review and returns a shortlist the critic rejected.
   *Fix:* a structured verdict via structured outputs (`{"approved": bool, "issues": [...]}`)
   and a check on the boolean. *(MAST FM-3.1 premature termination, FM-3.3 incorrect
   verification.)*

2. **The shortlist is sorted by the evaluation *text*, not the score.**
   `key=lambda e: e["evaluation"]` sorts prose alphabetically, so "This MOF…" ranks above
   "Score: 3", which ranks above "Excellent…" (checked). The "top 5" is effectively
   arbitrary. *Fix:* have workers return structured `{score: int, justification: str}`
   and sort by `score`.

3. **All workers share one `worker_history`, across candidates and across rounds.**
   - Every "independent" evaluation sees every earlier one, so scores anchor on previous
     candidates.
   - The history grows without bound: by round 3, each worker call carries hundreds of old
     evaluations, which costs money and latency and eventually overflows the context.
   - If the calls ever become truly concurrent (see #4), interleaved appends would also pair
     replies with the wrong candidates.

   *Fix:* fresh `messages` for each worker call. *(Bug study: "cognitive context
   mismanagement".)*

## Major

4. **The "parallel" evaluation is sequential.** `evaluate_candidate` is `async`, but it calls
   the *synchronous* client and `predict_properties` with no `await`. Each coroutine runs to
   completion while blocking the event loop, so `gather` adds nothing (5 workers at 0.2 s each
   took 1.0 s when I tested it). *Fix:* `anthropic.AsyncAnthropic()` and
   `await client.messages.create(...)`. Run the CPU-bound surrogate with `asyncio.to_thread`
   or batch it, and bound concurrency with an `asyncio.Semaphore` to respect rate limits.

5. **The number of rounds isn't bounded.** `round_num` is counted but never checked, so if the
   critic never approves, the loop runs forever, and each round costs more (#11). *Fix:*
   `max_rounds` and a budget, returning the best result so far with an "unapproved" status.
   *(Infinite-loops paper: multi-agent loop without a turn bound; MAST FM-1.5 unaware of
   termination conditions.)*

6. **`json.loads(plan)` runs on free-form model output.** It breaks on markdown fences or a
   sentence before the JSON. Nothing validates the schema (a missing `name` raises KeyError)
   or caps the number of candidates (200 candidates means 200 worker calls). *Fix:* structured
   outputs or `client.messages.parse` with a pydantic model, plus a cap and validation.

7. **The critic can't verify anything.** It sees only each candidate's name plus the first 200
   characters of the worker's prose. That's not the brief, not the predicted properties, and
   not the scores; the cut may even drop the score. It's grading a summary of a summary.
   *Fix:* give the critic the brief, the structured evaluations and the raw properties.
   Better, verify against ground truth (simulation, database lookup) rather than another LLM
   opinion. *(MAST FM-3.2 no or incomplete verification.)*

8. **The brief never reaches the workers or the critic.** Only the planner sees it, so workers
   score against generic criteria instead of the target application (humid flue gas, a cost
   limit), and the critic approves against nothing. *Fix:* pass the brief, or a structured
   spec, to every agent. *(MAST FM-2.4 information withholding, FM-1.1 disobey task
   specification.)*

9. **Candidates aren't grounded.** The planner can invent MOFs or repeat earlier rounds'
   candidates, and `predict_properties` on an invented name returns garbage or raises.
   *Fix:* resolve names to database IDs (CSD refcodes), reject unknown ones, and deduplicate
   across rounds. *(MAST FM-1.3 step repetition.)*

10. **One failure kills the whole round.** `asyncio.gather` without `return_exceptions=True`
    means a single worker exception (a rate limit after retries, a KeyError, numpy floats
    in `props` that `json.dumps` can't serialise) aborts the round and loses every other
    evaluation. *Fix:* per-task error handling that records failures.

## Minor

11. **The planner's context grows every round.** `json.dumps(evaluations)` appends every
    evaluation in full each time, so each round costs more. Send a compact structured summary
    of the latest round instead.
12. **`ask()` ignores `stop_reason`.** A `max_tokens` cut-off truncates the JSON plan, and a
    `refusal` returns empty text. Both surface later as confusing parse errors.
13. **Self-preference bias.** The critic and the planner are the same model with similar prompts.
    Consider a different rubric, a different model, or programmatic checks.

## Red herrings (say why they're fine)
- **Sharing one `client` across all agents is correct** (connection pooling, safe to share).
- **`"".join(b.text for b in response.content if b.type == "text")` is the right way to
  read text**: it handles thinking blocks and multiple text blocks. Compare exercise 05,
  where `content[0].text` is the bug.
- **The planner history alternates user and assistant correctly.**

## Model 1-minute summary
"Two bugs make the output meaningless without any error: the shortlist is sorted by the
evaluation prose instead of the score, and the approval check matches 'NOT APPROVED'. A
third contaminates everything: all workers share one growing conversation, so the evaluations
aren't independent. Then the design problems: workers and critic never see the brief, the
critic only sees 200-character summaries so it can't verify anything, there's no round
limit, and the 'parallel' workers actually run one at a time because the sync client blocks
the event loop. Fixes: structured outputs for the plan, scores and verdict; a fresh context per
worker that includes the brief; AsyncAnthropic with a semaphore; `max_rounds`; and checking
candidates against the database."
