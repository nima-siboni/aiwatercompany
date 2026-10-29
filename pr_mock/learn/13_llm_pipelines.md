# 13. LLM pipelines: retrieval, extraction and long-running assistants

Retrieval, extraction and long-running assistants rarely crash when they are wrong. A distance
read as a similarity, a citation index from the wrong list or a cache that never hits still
returns an answer, so the bugs show up as worse answers, wrong provenance or a larger bill. They
sit in the glue code around the model. The tool-use loop, stop reasons and prompt injection are
in chapter 6.

Outputs come from Python 3.14, NumPy 2.5 and anthropic 2.3 with stub HTTP transports; API behaviour is from the linked docs.

## Quick table

| You see | What goes wrong | Fix |
|---|---|---|
| `zip(requests, results)` over batch results | results come back in any order; failed rows crash or vanish | key by `custom_id`; count `result.type`; resubmit `errored` and `expired` |
| `vecs @ q` on raw model outputs; `score > t` on Chroma or FAISS `l2` results | long chunks win on length; those scores are distances | normalise every row; check the metric before writing a threshold |
| `0.5 * bm25 + 0.5 * cosine` | BM25 is unbounded, so it decides the ranking alone | reciprocal rank fusion |
| one `embed()` for chunks and queries, cached by text | wrong query and passage prefixes; old-model vectors after an upgrade | model, version and prefix in the cache key and the index metadata |
| a hand-written chunker with `start = end - overlap` | it hangs, or drops the tail and short documents | force progress; test that the chunks cover the text |
| `chunk_size=4000` (characters) with all-MiniLM-L6-v2 | 4,000-character chunks of paper text had 638 or more words; the model keeps 256 tokens: "Longer texts will be truncated" ([docs](https://sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html)) | measure chunks with the embedder's tokeniser |
| a metadata filter applied to the top-k list | 0 of 10 hits, although 1,018 of 10,000 chunks matched | filter inside the search, or over-fetch |
| `hash(text)` as a chunk ID; `add` on re-ingest | a new ID in every process; Chroma ignores IDs it already has ([docs](https://docs.trychroma.com/docs/collections/add-data)) | `sha256` of document ID and chunk number; `upsert` |
| `list(data.values())` written as a CSV row | Anthropic: "required properties appear first, followed by optional properties" ([docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)) | write by key |
| a retry that re-sends the same extraction after a JSON error | after `max_tokens` the output "might be incomplete" ([docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)): 3 identical requests, then `None` | check `stop_reason`; raise the limit or split; log failures |
| an assistant prefill, or `tool_choice={"type": "tool", ...}` | rejected: prefill since Opus 4.6 ([guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide)), forced tool choice with a 400 on Opus 5.5, Sonnet 5.5 and Fable 5.1 ([docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use)) | `output_config.format`, or `strict: true` with `tool_choice` auto |
| `messages[-40:]` on a tool-using history | it can start with a `tool_result` whose `tool_use` was cut, but "Tool result blocks must immediately follow their corresponding tool use blocks" ([docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)); OpenAI also loses the system prompt | cut before a user turn with no `tool_result`; keep the system prompt separate |
| `usage.input_tokens` as the context size | with caching it counts only "tokens after the last cache breakpoint" ([docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)) | add `cache_creation_input_tokens` and `cache_read_input_tokens` |
| a retry loop around `client.messages.create` | the SDK retries twice already: 5 attempts sent 15 requests; the spend-cap 429 has no `retry-after` ([docs](https://platform.claude.com/docs/en/api/rate-limits)) | one retry layer; stop on `enforced_spend_limit_reached` |

## Examples

### Batch results arrive in any order

```python
results = list(client.messages.batches.results(batch.id))
[(p, r.custom_id) for p, r in zip(paper_ids, results)][:2]
# [('paper-0', 'paper-3'), ('paper-1', 'paper-0')]
Counter(r.result.type for r in results)   # Counter({'succeeded': 4, 'expired': 1, 'errored': 1})
results[5].result.message
# AttributeError: 'MessageBatchErroredResult' object has no attribute 'message'
```

Results "can be returned in any order"
([batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing)), so
values land on the wrong paper without an error. Errored and expired requests are not billed and
need resubmitting, and reading `.message` from them raises.

**Fix:** key on `custom_id`, assert that every ID came back, and resubmit the failures. The same
page allows "only alphanumeric characters, hyphens, and underscores" in IDs, so map DOIs to short IDs.

### Inner product, distance or similarity

```python
docs = np.stack([0.5 * unit(q + 0.1 * n1),          # short chunk that answers the question
                 3.0 * unit(0.2 * q + n2)])         # long boilerplate chunk
docs @ q                                            # [0.498 0.757]: the boilerplate wins
normalise(docs) @ q                                 # [0.995 0.252]
dist = ((normalise(docs) - q) ** 2).sum(axis=1)     # what Chroma's default "l2" space returns
[i for i, s in enumerate(dist) if s > 0.75]         # [1]: keeps the boilerplate, drops the answer
```

`IndexFlatIP` and `vecs @ q` give cosine similarity only for unit vectors, and `axis=0`
normalises columns, not rows: only 5 of the true top 10 survived it. Chroma's `l2` and `cosine`
spaces both return distances ([Chroma](https://docs.trychroma.com/docs/collections/configure)).

**Fix:** normalise each row of the index and the query, and read the metric before writing a
threshold.

### One embedding space for the index and the queries

```python
def embed(text, model):
    key = hashlib.sha256(text.encode()).hexdigest()   # the model is not part of the key
    if key not in cache:
        cache[key] = model(text)
    return cache[key]
index = np.stack([embed(c, model_b) for c in chunks])  # "re-embedded" after the upgrade
# recall@1 on 500 questions: 1.0 before the upgrade, 0.008 after; shape (500, 768) both times
```

Every lookup hit the cache, so the index kept the old model's vectors, and equal dimensions meant
nothing failed. Prefixes are the same kind of mismatch: E5 needs "query: " and "passage: ", or
"you will see a performance degradation" ([model card](https://huggingface.co/intfloat/e5-large-v2)).

**Fix:** put the model, its version and the prefix in the cache key and the index metadata, and
check them at query time.

### Chunkers that loop or drop text

```python
def chunk(text, size=1000, overlap=200):
    out, start = [], 0
    while start < len(text):             # a caption, then a flattened table: start = 0, 408, 408, ...
        end = min(start + size, len(text))
        cut = text.rfind(". ", start, end)                   # snap to a sentence end
        end = cut + 1 if cut != -1 and end < len(text) else end
        out.append(text[start:end])
        start = len(text) if end == len(text) else end - overlap
    return out
```

When the only ". " in the window lies within `overlap` of `start`, `start` stops moving; plain
prose gave 4 chunks. The one-line version, `range(0, len(toks) - size, size - overlap)`, lost the
last 72 of 1,000 tokens and made no chunks at all for a 200-token abstract.

**Fix:** snap only when `cut > start + overlap`, and test that the chunks cover the whole text.

### Citation indices

```python
docs_this_turn = ["paper_C.pdf", "paper_D.pdf"]          # turn 1 sent paper_A and paper_B
for block in resp.content:
    for c in getattr(block, "citations", None) or []:
        print(docs_this_turn[c.document_index], list(range(c.start_page_number, c.end_page_number + 1)))
# paper_D.pdf [4, 5]: the citation is to paper_B.pdf, page 4
```

Indices count "all document content blocks in the request (spanning across all messages)", and
"Page numbers are 1-indexed with exclusive end page numbers"
([citations](https://platform.claude.com/docs/en/build-with-claude/citations)). Offsets found in
transformed text drift too: `"İ".lower()` has two characters, so a slice gave `'200 '` for `'1200'`.

**Fix:** keep one list of every document block in request order, use `range(start, end)`, and
compute offsets on the text you store.

### Values that the schema forces, and substring grounding

```python
def grounded(value: float, page_text: str) -> bool:
    return str(value) in page_text

grounded(1200.0, "a BET area of 1200 m2/g")      # False: str(1200.0) is '1200.0'
grounded(2.0, "an uptake of 12.05 mmol/g")       # True: '2.0' is inside '12.05'
```

The check rejects true values and passes invented ones. A schema that requires a `number` in
every field gives the model no way to say "not reported": in ChatExtract, removing the
verification questions cut precision from 90.8% to 42.7%
([Polak and Morgan, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10882009)).

**Fix:** allow `null`, require a verbatim evidence quote, and compare the numbers parsed from it.

### Numbers and units from PDF text

```python
FACTOR = {"mmol/g": 1.0, "cm3/g": 22.414, "mg/g": 1 / 44.01}
def to_mmol_g(value: str, unit: str) -> float:
    return float(value) * FACTOR.get(unit, 1.0)
to_mmol_g("50", "cm3/g")        # 1120.7; 50 cm3(STP)/g is 2.23 mmol/g, so divide
to_mmol_g("3.1", "mmol g−1")    # 3.1: an unknown spelling gets the factor 1.0
to_mmol_g("−35.2", "kJ/mol")    # ValueError: could not convert string to float: '−35.2'
```

PDF text uses U+2212 for minus, commas or thin spaces in thousands, `±` and `×10³`. `float`
rejects all of them, and NFKC normalisation leaves U+2212 as it is.

**Fix:** replace `−` and thin spaces before parsing, raise on unknown units, and store the raw
value and unit next to the converted one.

### Prompt-cache traps

```python
while count_tokens(history) > 150_000:
    history.pop(0)                         # the oldest turn goes, on every request once full
resp = client.messages.create(model=MODEL, max_tokens=4096, messages=history,
                              cache_control={"type": "ephemeral"})
# 200 turns of 3k tokens: 7.3x the input cost of cutting to 100k in one step (9.9x at Opus 5.5 prices)
```

The cache matches on prefix, so a new `messages[0]` every turn makes each request a cache write
(1.25 times the input price) instead of a read (0.1 times)
([prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)). A marker
after per-request text misses the same way, and "a cache entry only becomes available after the
first response starts", so parallel requests over one paper all pay in full.

**Fix:** trim in large steps, put the marker on the last block every request shares, and send one
request before fanning out.

### Assembling a stream by hand

```python
buf, out_tokens = "", 0
for ev in client.messages.create(model=MODEL, max_tokens=4096, tools=tools, messages=msgs, stream=True):
    if ev.type == "content_block_delta" and ev.delta.type == "input_json_delta":
        buf += ev.delta.partial_json            # one buffer for every tool_use block
    elif ev.type == "message_delta":
        out_tokens += ev.usage.output_tokens    # already cumulative: ends at 92, the response used 52
json.loads(buf)                                 # JSONDecodeError: Extra data: line 1 column 19 (char 18)
```

Two tool calls leave two JSON objects in one buffer, and `message_delta` token counts "are
cumulative" ([streaming](https://platform.claude.com/docs/en/build-with-claude/streaming)).

**Fix:** one buffer per `index`, parsed at `content_block_stop`; or `get_final_message()` from
`client.messages.stream(...)`, which returned both inputs and 52 tokens.

### stdio MCP servers

```python
@mcp.tool()
def run_gcmc(cif_path: str) -> dict:
    subprocess.run(["simulate", "simulation.input"], cwd=workdir)   # inherits the server's stdout
    return parse_output(workdir)
# MCP SDK 2.3 client: "Failed to parse JSONRPC message from server" for every line the child prints
```

"The server MUST NOT write anything to its stdout that is not a valid MCP message"
([transports](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)), and a
child process breaks that rule with no `print` in sight. The SDK 2.3 client logs and skips those
lines, so tests pass; other clients may not, and a line that lands inside a message corrupts it.
Tool names are a second trap: MCP allows
dots ([tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)), but the
Messages API pattern is `^[a-zA-Z0-9_-]{1,128}$` ([reference](https://platform.claude.com/docs/en/api/messages/create)).

**Fix:** log to stderr, run children with `capture_output=True`, and rename tools `server__tool`.

## Looks wrong, is fine

- **Ascending L2 distances on unit vectors.** They rank as cosine does, since ‖a − b‖² = 2 − 2cos(a, b);
  the top 100 matched exactly on 1,000 random vectors.
- **A dot product on OpenAI embeddings without normalising.** They are "normalized to length 1"
  ([FAQ](https://developers.openai.com/api/docs/guides/embeddings)), unless you cut dimensions.
- **A large `max_tokens` on an Anthropic call.** You pay for tokens generated, and it "does not
  factor into OTPM rate limit calculations" ([docs](https://platform.claude.com/docs/en/api/rate-limits)).
  On OpenAI it does count ([docs](https://developers.openai.com/api/docs/guides/rate-limits)).
- **`count_tokens` before every request.** It is free and has its own rate limit
  ([docs](https://platform.claude.com/docs/en/build-with-claude/token-counting)).
- **A cache marker that moves to the newest message each turn.** Automatic caching does exactly
  that, and earlier entries are still read ([docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)).
- **An MCP tool that raises instead of returning `isError: true`.** The Python SDK turns the
  exception into an `isError` result ([source](https://github.com/modelcontextprotocol/python-sdk));
  SDK v2 hides an unexpected exception's message, so raise `ToolError` for text the model should see.
