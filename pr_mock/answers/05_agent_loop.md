# Answer key: 05_agent_loop.py

**Good clarifying questions:** Who calls `run_agent`: a web service with many users, or a
notebook? Is `materials.db` the production database? What does a GCMC job cost, and is
there an approval or budget process? When `submit_gcmc_job` times out, was the job not
submitted, or is its status unknown? Which model and thinking settings are used (this
changes what `response.content` looks like)?

## Blockers

1. **Tool results are attached to the wrong tool calls.** In
   `zip(tool_blocks, as_completed(futures))`, `as_completed` yields futures in *completion*
   order, while `tool_blocks` are in *request* order. When the model makes two parallel
   calls (a slow Zn query and a fast Cu query), the Cu rows come back labelled with the Zn
   call's `tool_use_id`, and job IDs get swapped between MOFs. Nothing errors.
   *Fix:* `[f.result() for f in futures]` (submission order), or a `{future: block}` map.
   It's easy to demonstrate: two futures with different sleep times swap their results.

2. **Model-written SQL runs with write access and no limits.** `sqlite3.connect(DB_PATH)` is
   read-write, so the model, or a prompt injection hidden in the question or in database text,
   can run `DELETE`, `UPDATE` or `DROP`. `SELECT * FROM mofs` on a 100k-row table also gets
   fully `fetchall()`'d and dumped into the context. *Fix:* open it read-only
   (`sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)`), allow only SELECT, or better,
   replace raw SQL with typed tools (`find_mofs(min_uptake, water_stable, limit)`); enforce a
   row cap and a timeout. *(OWASP Agentic ASI02, tool misuse / excessive privilege.)*

3. **Errors are disguised as empty results.** `except Exception: return "[]"` makes a SQL
   syntax error, a wrong column name or a cluster outage look like "no rows". The model then
   confidently reports "no MOFs match your criteria", or that the job was queued. Silent tool
   failures are the most-cited production failure mode in the practitioner literature.
   *Fix:* return a `tool_result` with `"is_error": True` and a short message, and log the
   exception. Also, an unknown tool name falls through and returns `None`, which is invalid
   content and triggers a 400 (see #6).

4. **One conversation is shared by every caller.** `messages=[]` is a mutable default. Every call
   appends to the same list, so user B's request carries user A's whole conversation (a data
   leak), the context grows forever, and concurrent requests interleave into an invalid
   history (a `tool_result` without a matching `tool_use` gives a 400). *Fix:* use `messages=None`
   and create a new list per call, keeping sessions explicitly by ID.

## Major

5. **There's no bound on the loop.** `while True` stops only when the model says so. Combined
   with #3, the model keeps rewriting a failing query and getting `"[]"`, and every iteration
   re-sends the full history. *Fix:* `max_iterations`, a token/cost budget and a wall-clock
   timeout, returning a partial result with a status. *(Infinite-agentic-loops paper:
   model-controlled termination, tool-call iteration without a bound.)*

6. **Only `end_turn` is handled; everything else is treated as tool use.**
   - `max_tokens`: if the turn was cut off with no `tool_use` blocks, the code appends a user
     message with empty content, which is a 400. If it was cut off inside a tool call, that
     call is incomplete and must not run.
   - `refusal`: never execute tools from a refused turn.

   The resulting 400 is retried 5 times by `call_model` and finally reported as "model
   unavailable", which hides the real cause. *Fix:* branch explicitly. `tool_use` runs
   the tools; `end_turn` finishes; `max_tokens` retries with more room or fails loudly;
   `refusal` stops and reports; anything else raises.

7. **A side-effecting tool is retried even though it isn't idempotent.** `submit_simulation`
   retries on `TimeoutError`, but a client timeout doesn't mean the scheduler didn't accept the
   job, so you get duplicate multi-hour GCMC runs. *Fix:* an idempotency key (e.g. a hash of
   run_id + mof_id + params) that the scheduler deduplicates on, or a check for an existing
   job before resubmitting.

8. **There are no guardrails on expensive actions.** The model can submit any number of jobs
   with any parameters (500 jobs; 10,000 bar). *Fix:* a per-run cap, range validation on
   parameters, and a plan-then-approve step for batches. *(OWASP: excessive agency.)*

9. **The timestamp defeats prompt caching.** `system_prompt()` runs on every call and embeds
   `datetime.now().isoformat()` (microsecond precision), so the system prompt changes each
   iteration. The prefix order is tools → system → messages, so the cached history is never
   reused. Since the loop re-sends a growing history every turn, total cost grows roughly
   quadratically with the number of turns, all at full price. *Fix:* keep the system prompt
   static (a date at day granularity, computed once per session) and confirm with
   `usage.cache_read_input_tokens`.

10. **The retry wrapper retries everything.** `except Exception` includes 400s (a malformed
    request, context too long) and 401s, which will never succeed, and it hides the real
    error. The SDK already retries 429/5xx with backoff (`max_retries=2` by default), so this
    nests into up to 15 attempts with a fixed 1 s sleep. *Fix:* rely on the SDK's retries
    (tune `max_retries`). If you need more, catch only connection/rate-limit/5xx errors and
    let 4xx raise.

11. **`response.content[0].text` assumes the first block is text.** With adaptive thinking
    (the default on current Claude models), the first block is often a `thinking` block,
    so you get an AttributeError *after* the whole run has finished and been paid for. It
    also drops any extra text blocks. *Fix:*
    `"".join(b.text for b in response.content if b.type == "text")`.

## Minor

12. **The tool output has no column names.** `json.dumps(rows)` of tuples leaves the model
    guessing which number is uptake and which is surface area. Return a list of dicts using
    `cursor.description`, with a size cap.
13. **The tool descriptions are thin.** There are no units (is `co2_uptake` in mmol/g?), no
    encoding for `water_stable` (0/1?), no SQL dialect, and no guidance on when to use each
    tool. Add `strict: True` with `additionalProperties: False` so `submit_simulation(**tool_input)`
    can't receive unexpected keys.
14. **No observability.** `print` instead of structured logs, and no trace of each tool call's
    input and output or tokens per iteration, so the silent failures in #3 stay invisible.
15. **Small cleanups.** The connection isn't closed on exceptions (use `contextlib.closing`),
    and a new thread pool is created every iteration.

## Red herrings (say why they're fine)
- **`messages.append({"role": "assistant", "content": response.content})`** passes SDK block
  objects rather than dicts. That's correct and required, because it keeps the `tool_use`
  (and thinking) blocks the next request needs. Don't "fix" it by appending only the text.
- **All tool results go back in one user message.** That's correct, and it's what keeps the model
  making parallel calls.
- **`block.input` is already a dict**, so no `json.loads` is needed. In OpenAI-style APIs,
  `tool_call.function.arguments` is a JSON *string* that can be invalid.
- **The thread pool is fine** for I/O-bound tools. The bug is how the results are paired, not
  the threading.

**Applying this to OpenAI-style code:** the same checks are `finish_reason == "length"` (the
equivalent of `max_tokens`) and `"content_filter"`. Tool results there are separate
`{"role": "tool", "tool_call_id": ...}` messages, one per call.

## Model 1-minute summary
"Four things before anything else. Tool results get attached to the wrong calls whenever the
model calls tools in parallel. Errors come back as `[]`, so the agent confidently reports
'no matches'. Model-written SQL runs on a writable connection with no row limit. And the
mutable default shares one conversation across all users. Next is loop safety: there's no
iteration or cost cap, and only `end_turn` is handled, so `max_tokens` and `refusal` become a
400 that the broad retry disguises as an outage. Job submissions are retried even though they
aren't idempotent, and nothing caps expensive submissions. On cost, the timestamp in the system
prompt defeats caching on every turn. For tests, I'd use a fake client returning two parallel
tool calls that complete out of order, a tool that raises, and a `max_tokens` response."
