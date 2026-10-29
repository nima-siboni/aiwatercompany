"""Reference fix for PR #242 (python_patterns/exercises/04_generators/L2_streaming_dataset.py).
See the answer key for the reasoning."""
import gzip
import json
import logging
import random
from typing import Iterable, Iterator

log = logging.getLogger(__name__)


def open_shard(path: str) -> Iterator[str]:
    """Iterate over the lines of a shard, transparently handling .gz files."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:  # a generator, so the file stays open while it's being read
        yield from f


class RelaxationDataset:
    def __init__(self, shard_paths: list[str], buffer_size: int = 10_000, seed: int = 0):
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be at least 1, got {buffer_size}")
        self.shard_paths = list(shard_paths)
        self.buffer_size = buffer_size
        self.rng = random.Random(seed)
        self._len: int | None = None

    def _records(self, paths: list[str]) -> Iterator[dict]:
        for path in paths:
            for lineno, line in enumerate(open_shard(path), start=1):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # The only expected damage: a truncated last line from a killed writer.
                    log.warning("%s:%d: skipping unparseable line", path, lineno)
                    continue
                try:
                    out = {
                        "mof_id": rec["mof_id"],
                        "structure": rec["structure"],
                        "energy_per_atom": rec["energy"] / rec["n_atoms"],
                    }
                except (KeyError, TypeError, ZeroDivisionError) as e:
                    raise ValueError(f"{path}:{lineno}: bad record: {e!r}") from e
                yield out

    def _shuffle(self, records: Iterator[dict]) -> Iterator[dict]:
        """Approximate shuffle: keep a buffer and emit a random element of it."""
        buf = []
        for rec in records:
            buf.append(rec)
            if len(buf) >= self.buffer_size:
                i = self.rng.randrange(len(buf))
                buf[i], buf[-1] = buf[-1], buf[i]  # O(1) removal instead of pop(i)
                yield buf.pop()
        self.rng.shuffle(buf)  # flush the tail
        yield from buf

    def __iter__(self) -> Iterator[dict]:
        # A fresh pipeline per call, so every epoch sees the whole dataset.
        paths = list(self.shard_paths)
        self.rng.shuffle(paths)  # the buffer only mixes nearby records, so vary the shard order too
        return self._shuffle(self._records(paths))

    def __len__(self) -> int:
        if self._len is None:  # one full pass, once; a manifest from the ETL job would be cheaper
            self._len = sum(1 for _ in self._records(self.shard_paths))
        return self._len


def mean_energy(records: Iterable[dict]) -> float:
    """Mean per-atom energy (eV/atom), used to normalise the training targets."""
    n, total = 0, 0.0
    for rec in records:
        n += 1
        total += rec["energy_per_atom"]
    if n == 0:
        raise ValueError("mean_energy of an empty dataset")
    return total / n
