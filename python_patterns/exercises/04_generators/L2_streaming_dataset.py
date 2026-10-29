"""
PR #242: Streaming dataset for the DFT relaxation data

The relaxation dataset no longer fits in memory (~40 GB of JSONL in ~300 shards, the newer
ones gzipped), so this replaces the in-memory loader with a streaming one. Records are read
lazily shard by shard, mixed with a fixed-size shuffle buffer, and converted to per-atom
energies. A few shards end in a truncated line from relaxations that were killed
mid-write, so the reader skips lines it can't parse. Used by the surrogate training loop:

    ds = RelaxationDataset(sorted(glob("shards/*.jsonl*")), buffer_size=10_000, seed=0)
    mu = mean_energy(ds)             # target normalisation
    for epoch in range(n_epochs):
        for record in ds:
            ...
"""
import gzip
import json
import random
from typing import Iterable, Iterator


def open_shard(path: str):
    """Iterate over the lines of a shard, transparently handling .gz files."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return (line for line in f)
    return open(path)


class RelaxationDataset:
    def __init__(self, shard_paths: list[str], buffer_size: int = 10_000, seed: int = 0):
        self.shard_paths = shard_paths
        self.buffer_size = buffer_size
        self.rng = random.Random(seed)
        self._stream = self._shuffle(self._records())

    def _records(self) -> Iterator[dict]:
        for path in self.shard_paths:
            for line in open_shard(path):
                try:
                    rec = json.loads(line)
                    yield {
                        "mof_id": rec["mof_id"],
                        "structure": rec["structure"],
                        "energy_per_atom": rec["energy"] / rec["n_atoms"],
                    }
                except Exception:
                    continue

    def _shuffle(self, records: Iterator[dict]) -> Iterator[dict]:
        """Approximate shuffle: keep a buffer and emit a random element of it."""
        buf = []
        for rec in records:
            buf.append(rec)
            if len(buf) >= self.buffer_size:
                yield buf.pop(self.rng.randrange(len(buf)))

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        return next(self._stream)

    def __len__(self) -> int:
        return sum(1 for path in self.shard_paths for _ in open_shard(path))


def mean_energy(records: Iterable[dict]) -> float:
    """Mean per-atom energy (eV/atom), used to normalise the training targets."""
    n, total = 0, 0.0
    for rec in records:
        n += 1
        total += rec["energy_per_atom"]
    return total / n
