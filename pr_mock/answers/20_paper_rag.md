# Answer key: 20_paper_rag.py

**Good clarifying questions:** How many chunks will the corpus have? Which filters do chemists
use, and how selective are they? Will chemists act on these answers in the lab, with the
citations taken as authoritative? How were the PDFs extracted (tables, captions,
supplementary information)? What latency is acceptable?

## Blockers

1. **The vectors are normalised along the wrong axis** (`Index.__init__`).
   `np.linalg.norm(vecs, axis=0)` divides each *dimension* by its norm over all chunks, instead
   of each vector by its own length. The scores are then neither cosines nor anything stable,
   and long vectors win. On 5,000 synthetic vectors, the PR's top 8 shared on average 1 chunk
   with the true cosine top 8, and the PR's top 8 had norms 3–5× the median. *Fix:*
   `axis=1, keepdims=True` (or `normalize_embeddings=True` in `encode`), and assert that the
   rows have unit norm.

2. **The filters are applied after the top-k** (`Index.search`). It takes the best 8 chunks of
   the whole corpus and then filters them. With Mg in 5% of the chunks, 331 of 500 random
   queries had no Mg chunk left. The model then gets an empty context and answers from memory,
   while the UI suggests the answer is grounded in the papers. *Fix:* filter first (or use the
   vector store's metadata filter), or over-fetch until there are k filtered hits. When there
   are none, say so instead of calling the model.

3. **The citations are shifted by one** (`answer`). The excerpts are numbered from 1, but
   `hits[n]` indexes from 0. A citation of [1] linked to the second paper, and citing the last
   excerpt raised IndexError. Every source link in the UI points at the wrong paper. *Fix:*
   `hits[n - 1]` with a range check, or better, cite document IDs rather than positions.

## Major

4. **The chunker can loop forever** (`chunk`). `start = end - overlap` moves backwards when the
   only sentence end in the window is near its start. The next window then finds the same
   sentence end, and the loop never advances, appending the same chunk again and again. A
   page of prose followed by a flattened table (with no ". " in it) did not return within 10
   seconds. Ingest hangs on the first such page, with memory growing. *Fix:*
   `start = max(end - overlap, start + 1)`, assert `0 <= overlap < size`, and test that the
   chunks cover the whole text.
5. **Chunks are longer than E5's window.** E5 reads at most 512 tokens, and sentence-transformers
   truncates longer input without a warning. A 2,000-character chunk of MOF synthesis text came
   to 611 tokens, so the last sixth of each chunk is never embedded, and a little more once the
   prefix is added. *Fix:* size the chunks in tokens with the model's own tokenizer (say 400),
   leaving room for the prefix.
6. **The E5 query prefix is missing.** E5 was trained with "query: " on queries and "passage: "
   on passages, and its model card warns that leaving them out degrades performance. The PR
   adds "passage: " to the chunks but nothing to the question. *Fix:*
   `embed("query: " + question)`.
7. **The embedding cache ignores the model** (`embed`). The key is the SHA-256 of the text alone.
   After `EMBED_MODEL` was switched, `embed()` still returned the old model's vector for every
   cached text. e5-base and bge-base both produce 768 dimensions, so nothing errors, and the
   index mixes two embedding spaces. *Fix:* put the model name, revision and prefix in the key,
   store them with the index, and re-embed when they change.
8. **Indexing is slow, and the cache is a directory of tiny files.** Every chunk is a separate
   `encode` call, with no batching. Each chunk is its own `.npy` file, hundreds of thousands in
   one directory. The whole index is rebuilt in memory at every start. *Fix:* encode in
   batches, store the vectors in one array file or a vector store, and build the index once.

## Minor

9. **`np.argsort` over every chunk for every query** is O(n log n). Use `argpartition`, or an
   approximate-nearest-neighbour index once the corpus grows.
10. **The system prompt doesn't say what to do** when the excerpts don't contain the answer.
    Tell the model to say so.
11. **"Page" is the page's position in the PDF,** which can differ from the journal's printed
    page number. Make sure the UI says which one it shows.

## Red herrings (say why they're fine)

- **SHA-256 cache keys** are stable across processes, unlike `hash()`. The problem is what goes
  into the key, not the hash function.
- **Normalising the query vector** is right; it's the index normalisation that's wrong.
- **`next(b.text for b in response.content if b.type == "text")`** handles thinking blocks
  correctly.
- **`np.argsort(...)[::-1][:k]`** does give the top k, just not the fastest way.
- **Chunking each page separately** keeps page numbers for the citations, which is a good
  design.

## Model 1-minute summary
"The retrieval doesn't return the right chunks, and the citations don't point at the right
papers. The index is normalised along the wrong axis, so long vectors win: on synthetic data the
PR's top 8 shared about one chunk with the real cosine top 8. Filters run after the top-k, so a
5% filter left two thirds of queries with nothing, and the model then answers from memory. The
citations are 1-based in the prompt and 0-based in the lookup, so every link is shifted, and
citing the last excerpt crashes. Then ingest: the chunker loops forever on a page with a
flattened table, chunks are longer than E5's 512 tokens, the query prefix is missing, and the
embedding cache ignores the model name. I'd want a small labelled set of question-and-paper
pairs to measure recall@k before and after the fixes."
