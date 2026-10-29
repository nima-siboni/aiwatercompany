"""
Reference fix for PR #301 (python_patterns/exercises/10_concurrency/L1_parallel_featurise.py).
See the answer key for the reasoning.
"""
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Protocol

import numpy as np

log = logging.getLogger(__name__)


class FeatureCache(Protocol):
    def put(self, structure_id: str, features: np.ndarray) -> None: ...


def featurise_all(
    structures: dict[str, str],
    featurise: Callable[[str], np.ndarray],
    cache: FeatureCache,
    n_workers: int | None = None,
) -> dict[str, np.ndarray]:
    """Featurise CIF texts keyed by structure ID.

    The work is pure-Python CPU, so it runs in processes (threads would share one
    GIL). `featurise` must therefore be picklable: a module-level function.
    """
    features: dict[str, np.ndarray] = {}
    failed: list[str] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_to_id = {pool.submit(featurise, cif): sid for sid, cif in structures.items()}
        for future in as_completed(future_to_id):
            sid = future_to_id[future]  # completion order != submission order
            try:
                vec = future.result()
            except Exception:
                log.warning("featurisation failed for %s", sid, exc_info=True)
                failed.append(sid)
                continue
            features[sid] = vec
            try:
                cache.put(sid, vec)  # in this process, as each result arrives
            except Exception:
                log.warning("could not cache features for %s", sid, exc_info=True)
    log.info(
        "featurised %d of %d structures; %d failed", len(features), len(structures), len(failed)
    )
    return features
