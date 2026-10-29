"""
PR #294: Retrieval over MOF synthesis papers for the lab assistant

Lets chemists ask questions like "how is Mg-MOF-74 activated?" over our 12k-paper corpus
(text already extracted from the PDFs, page by page). Pages are split into overlapping chunks,
embedded with E5 (intfloat/e5-base-v2) and searched by cosine similarity, optionally filtered
by metal and publication year. Claude answers from the top chunks and cites them by number;
the UI links each citation to the paper and page.
"""
import hashlib
import os
import re

import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "intfloat/e5-base-v2"
CACHE_DIR = "embedding_cache"
client = anthropic.Anthropic()
model = SentenceTransformer(EMBED_MODEL)


def chunk(text, size=2000, overlap=300):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        cut = text.rfind(". ", start, end)  # end chunks at a sentence boundary
        if cut != -1 and end < len(text):
            end = cut + 1
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def embed(text):
    key = hashlib.sha256(text.encode()).hexdigest()
    path = os.path.join(CACHE_DIR, key + ".npy")
    if os.path.exists(path):
        return np.load(path)
    vec = model.encode(text)
    np.save(path, vec)
    return vec


class Index:
    def __init__(self, papers):
        """papers: dicts with doi, title, metal, year and pages (a list of page texts)."""
        self.meta, vecs = [], []
        for paper in papers:
            for page_no, page in enumerate(paper["pages"], start=1):
                for c in chunk(page):
                    self.meta.append({"doi": paper["doi"], "page": page_no, "metal": paper["metal"],
                                      "year": paper["year"], "text": c})
                    vecs.append(embed("passage: " + c))
        vecs = np.array(vecs)
        self.vecs = vecs / np.linalg.norm(vecs, axis=0)

    def search(self, question, k=8, metal=None, min_year=None):
        q = embed(question)
        q = q / np.linalg.norm(q)
        top = np.argsort(self.vecs @ q)[::-1][:k]
        hits = [self.meta[i] for i in top]
        if metal:
            hits = [h for h in hits if h["metal"] == metal]
        if min_year:
            hits = [h for h in hits if h["year"] >= min_year]
        return hits


def answer(index, question, **filters):
    hits = index.search(question, **filters)
    excerpts = "\n\n".join(f"[{n}] {h['text']}" for n, h in enumerate(hits, start=1))
    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=4000,
        system="Answer the chemist's question from the numbered excerpts. Cite excerpts as [n].",
        messages=[{"role": "user", "content": f"Excerpts:\n{excerpts}\n\nQuestion: {question}"}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    sources = [{"doi": hits[n]["doi"], "page": hits[n]["page"]} for n in cited]
    return {"answer": text, "sources": sources}
