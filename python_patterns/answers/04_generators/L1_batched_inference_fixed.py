"""Reference fix for PR #241 (python_patterns/exercises/04_generators/L1_batched_inference.py).
See the answer key for the reasoning."""
import json
from typing import Callable, Iterable, Iterator

Model = Callable[[list[dict]], list[float]]


def read_structures(path: str) -> Iterator[dict]:
    """Yield one structure dict per line of a JSONL file, skipping blank lines."""
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
            yield record


def batched(items: Iterable, size: int) -> Iterator[list]:
    """Group `items` into lists of `size` items; the last list may be shorter."""
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def predict_uptake(model: Model, structures: Iterable[dict], batch_size: int = 256) -> dict[str, float]:
    """Return {mof_id: predicted uptake} for every structure."""
    preds = {}
    for batch in batched(structures, batch_size):  # a single pass, so generators work
        for s, uptake in zip(batch, model(batch), strict=True):
            preds[s["mof_id"]] = uptake
    return preds
