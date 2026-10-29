"""
PR #241: Batched inference for the uptake surrogate model

Scoring the ~200k MOFs from the latest generative run one at a time takes most of a day,
almost all of it per-request overhead in the model server. This adds a small `batched`
helper and a `predict_uptake` function that sends structures to the surrogate model in
batches. Structures are streamed from the generator's JSONL output, so we never hold them
all in memory:

    preds = predict_uptake(model, read_structures("gen_run_42.jsonl"), batch_size=256)

`model` takes a list of structure dicts and returns one predicted CO2 uptake (mmol/g) per
structure, in the same order.
"""
import json
from typing import Callable, Iterable, Iterator

Model = Callable[[list[dict]], list[float]]


def read_structures(path: str) -> Iterator[dict]:
    """Yield one structure dict per line of a JSONL file."""
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def batched(items: Iterable, size: int) -> Iterator[list]:
    """Group `items` into lists of `size` items."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []


def predict_uptake(model: Model, structures: Iterable[dict], batch_size: int = 256) -> dict[str, float]:
    """Return {mof_id: predicted uptake} for every structure."""
    mof_ids = [s["mof_id"] for s in structures]
    preds = []
    for batch in batched(structures, batch_size):
        preds.extend(model(batch))
    return dict(zip(mof_ids, preds))
