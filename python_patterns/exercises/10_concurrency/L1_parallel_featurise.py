"""
PR #301: Featurise candidate structures in a thread pool

Featurising the 50k hypothetical MOFs one at a time took most of an hour, so this
spreads the work over a thread pool. The featuriser parses the CIF text (the
standard file format for crystal structures) and computes descriptors in pure
Python, about 70 ms of CPU per structure. `featurise_all` returns a dict from
structure ID to feature vector; structures that fail are logged and skipped.
Each vector is also written to the shared feature cache as soon as it's ready, so
other jobs can reuse it.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Protocol

import numpy as np

log = logging.getLogger(__name__)


class FeatureCache(Protocol):
    def put(self, structure_id: str, features: np.ndarray) -> None: ...


def featurise_all(
    structures: dict[str, str],
    featurise: Callable[[str], np.ndarray],
    cache: FeatureCache,
    n_workers: int = 32,
) -> dict[str, np.ndarray]:
    """Featurise CIF texts keyed by structure ID."""
    failed: list[str] = []

    def work(structure_id: str) -> np.ndarray | None:
        try:
            return featurise(structures[structure_id])
        except Exception:
            log.warning("featurisation failed for %s", structure_id)
            failed.append(structure_id)
            return None

    ids = list(structures)
    features = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(work, sid) for sid in ids]
        for sid, future in zip(ids, as_completed(futures)):
            vec = future.result()
            if vec is not None:
                features[sid] = vec
                pool.submit(cache.put, sid, vec)
    log.info("featurised %d of %d structures; %d failed", len(features), len(ids), len(failed))
    return features
