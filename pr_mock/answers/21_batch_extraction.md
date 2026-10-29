# Answer key: 21_batch_extraction.py

**Good clarifying questions:** Do papers report several MOFs, temperatures and pressures? Is
the text extracted from PDFs, with tables flattened? What share of papers report all three
quantities? What error rate is acceptable, given that this feeds the surrogate's training set?
How will we audit a row? What are the budget and the deadline (a batch can take up to 24 hours)?

## Blockers

1. **Results are paired with papers by position** (`extract`). Batch results can come back in
   any order (the SDK's docstring: "results may be given out of request order"), and the
   `custom_id` that would fix it is set and then ignored. In the check, with three results out
   of order, paper a got MOF-C's numbers, b got MOF-A's and c got MOF-D's. The grounding check
   then rejects most of these mismatched records, so the run produced 0 rows from 4 papers, with
   no error. In a long paper a wrong record still passes the substring check a few percent of
   the time: 2% with 300 numbers in the text, 6% with 1,500. So the table comes out mostly
   empty, and the rows that survive can be wrong. *Fix:* index the results by `custom_id`, look
   up each paper's id, and assert that the ids sent equal the ids returned.

2. **The schema forces an answer** (`SCHEMA`). Every number is required and none can be null,
   so for a paper that doesn't report BET or Qst the model still has to produce a number.
   Constrained decoding guarantees the JSON matches the schema, not that the values exist. In
   ChatExtract (Nat. Commun. 2024), removing the verification step dropped precision from 90.8%
   to 42.7%, mostly through values invented where none was given. One record per paper also
   discards every other MOF in the paper. And there are no fields for temperature and pressure,
   so values measured at 273 K or 0.15 bar get stored as 298 K and 1 bar. *Fix:* allow null
   (`["number", "null"]`) and return a list of records, each with its temperature, pressure,
   unit and a verbatim evidence quote.

3. **Unknown unit spellings are multiplied by 1** (`to_mmol_g`). The model copies units the way
   the paper writes them: "cm3(STP)/g", "cm³/g", "cc/g", "mmol g-1", "wt.%". All of these miss
   the dictionary and fall back to 1.0, so 50 cm³(STP)/g is stored as 50 mmol/g instead of
   2.23, a factor of 22, in a table that trains the surrogate. *Fix:* normalise unit strings
   (superscripts, "(STP)", spaces), raise on unknown units, and store the raw value and unit
   next to the converted one. An enum of units in the schema helps too.

4. **The grounding check rejects true values and accepts invented ones** (`grounded`).
   `str(v) in text` is a substring match on Python's number formatting:
   - "1,200 m2/g" contains neither "1200" nor "1200.0", and "4 mmol/g" doesn't contain "4.0",
     so correct records are rejected.
   - An invented BET of 2.0 and Qst of 12.0 passed, because both strings occur inside "12.05".
   - A number appearing somewhere in a 50-page paper says nothing about whether it belongs to
     this MOF and these conditions.
   *Fix:* ask for a verbatim evidence quote per value, check that the quote is in the text,
   then parse the numbers from the quote (normalising thousands separators, minus signs and
   superscripts) and compare them numerically.

## Major

5. **Failures disappear without a trace.** Errored, expired and cancelled results, unparsable
   JSON and grounding failures all hit `continue`, with no count, log or resubmission. In the
   check, paper 1's `invalid_request` simply vanished. The batch docs say to fix and resubmit
   invalid requests, and to resubmit server errors and expired requests. *Fix:* count every
   outcome, write the failures to a retry file, and report coverage.
6. **`max_tokens=1024` while Haiku 5.5 thinks by default.** Thinking tokens count toward
   `max_tokens`, so long papers stop in the middle of the JSON. The code never checks
   `stop_reason`, so the JSON fails to parse and the paper is silently skipped (5). *Fix:* give
   it room (you pay only for tokens generated), and route `max_tokens` and `refusal` stops to a
   retry with a larger limit.
7. **One batch for the whole corpus.** A batch is limited to 100,000 requests or 256 MB, and
   20,000 papers at 15k–60k characters each is 300 MB–1.2 GB, so `create` fails. *Fix:* split
   into several batches under the limit, store the batch ids so that a crash doesn't submit
   (and pay) twice, and save the raw results, which expire after 29 days.

## Minor

8. **`cache_control` on the paper text.** The marker comes after per-paper content, so every
   request pays the 1.25× cache-write price on its paper, and nothing is ever read back: about
   25% more input cost for nothing. The shared instructions are below the 512-token minimum
   anyway. Remove the marker. If few-shot examples are added to the instructions later, mark
   those.
9. **No provenance.** A row stores neither the evidence nor its location (page, table) nor the
   model version, so it can't be audited or re-run selectively.
10. **Vague instructions.** "Main MOF" is undefined, "wt%" can mean per gram of adsorbent or per
    gram of loaded sample, and the prompt asks for 298 K and 1 bar without recording what the
    paper actually used.

## Red herrings (say why they're fine)

- **`custom_id=f"paper-{i}"`** is unique and fine. The bug is ignoring it when reading results.
- **`next(b.text for b in message.content if b.type == "text")`** is the right way to read the
  answer. With thinking on, the first block is a thinking block, so `content[0].text` would
  fail.
- **`json.loads` on a structured-output response** is fine when the response completed,
  because decoding is constrained. The risk is truncation (6).
- **Polling every 60 seconds** is fine for a job that can take up to 24 hours.
- **`"additionalProperties": False`** is required for strict schemas.
- **The mg/g and cm³/g factors** (1/44.01 and 1/22.414) are right. Only the lookup is broken.
- **The Batches API at half price** is the right tool for an offline job of 20,000 papers.

## Model 1-minute summary
"The rows aren't attached to the right papers. Results are paired by position although batches
return them in any order, and the `custom_id` that would fix it is ignored. Most mismatched
records then fail the grounding check, so the table comes out mostly empty, with no error, and
the rows that pass can be wrong. The schema makes every number mandatory, so the model has to
invent BET and Qst values when a paper has none. The substring 'grounding' rejects true values
like 1,200 and accepts invented ones like 2.0 inside 12.05, and unknown unit spellings are
treated as mmol/g, a factor of 22 for cm³(STP)/g. Then the operational issues: failures are
dropped silently, thinking eats the `max_tokens`, the batch is over the size limit, and the cache
marker only adds cost. Before this feeds a training set, I'd want it scored against 100
hand-labelled papers."
