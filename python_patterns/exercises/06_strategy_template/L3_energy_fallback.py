"""
PR #263: Formation energies with a DFT -> ML -> surrogate fallback

The active-learning service needs a formation energy (eV per unit cell,
relative to the pure elements) for every structure it labels. DFT
(quantum-mechanical simulation) is the most accurate but fails or times out on
some frameworks, so compute_energy() now tries a list of calculators in order
and records which one produced each value. The ML potential is retrained every
round and swapped in with update_model(). The surrogate is a cheap linear
model on element fractions, used as a last resort. run_dft(structure,
functional) is injected by the service: it submits a cluster job, waits for
it, and raises TransientError if the node is lost.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from aiwc.structures import Structure

log = logging.getLogger(__name__)

MAX_DFT_ATOMS = 400


class TransientError(Exception):
    """A failure worth retrying, e.g. the cluster lost the node."""


@runtime_checkable
class EnergyModel(Protocol):
    name: str

    def energy(self, structure: Structure) -> float: ...


class Calculator:
    """Base class: energy() caches results per structure; subclasses implement _compute()."""

    name = "base"

    def __init__(self) -> None:
        self._cache: dict[str, float] = {}

    def energy(self, structure: Structure) -> float:
        """Formation energy in eV per unit cell."""
        if structure.id not in self._cache:
            self._cache[structure.id] = self._compute(structure)
        return self._cache[structure.id]

    def _compute(self, structure: Structure) -> float:
        raise NotImplementedError


class TimingMixin:
    """Records the wall time of every call; the service dashboard reads .timings."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.timings: list[float] = []

    def energy(self, structure: Structure) -> float:
        start = time.perf_counter()
        try:
            return super().energy(structure)
        finally:
            self.timings.append(time.perf_counter() - start)


class RetryMixin:
    """Retries calls that fail with TransientError."""

    def __init__(self, retries: int = 2, **kwargs) -> None:
        self.retries = retries

    def energy(self, structure: Structure) -> float:
        for attempt in range(self.retries + 1):
            try:
                return super().energy(structure)
            except TransientError:
                if attempt == self.retries:
                    raise
                log.warning("transient failure on %s, retrying", structure.id)


class DFTCalculator(RetryMixin, TimingMixin, Calculator):
    name = "dft"

    def __init__(self, run_dft: Callable[[Structure, str], float], functional: str = "PBE", **kwargs) -> None:
        super().__init__(**kwargs)
        self.run_dft = run_dft
        self.functional = functional

    def _compute(self, structure: Structure) -> float:
        return self.run_dft(structure, self.functional)


class MLPotential(TimingMixin, Calculator):
    """Predicts one energy contribution per atom; they sum to the cell energy."""

    name = "ml"

    def __init__(self, model: Callable[[Structure], np.ndarray], **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.version = 0

    def update_model(self, model: Callable[[Structure], np.ndarray]) -> None:
        """Swap in the potential retrained this round."""
        self.model = model
        self.version += 1

    def _compute(self, structure: Structure) -> float:
        return float(np.sum(self.model(structure)))


class CompositionSurrogate(TimingMixin, Calculator):
    """Linear model on element fractions, fitted to our DFT formation energies per atom."""

    name = "surrogate"

    def __init__(self, coef: dict[str, float], intercept: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.coef = coef
        self.intercept = intercept

    def _compute(self, structure: Structure) -> float:
        n = len(structure.species)
        return self.intercept + sum(
            self.coef.get(el, 0.0) * structure.species.count(el) / n for el in set(structure.species)
        )


@dataclass
class EnergyResult:
    structure_id: str
    energy: float  # eV per unit cell
    source: str


def compute_energy(structure: Structure, calculators: list[EnergyModel]) -> EnergyResult:
    """Return the energy from the first calculator in the list that succeeds."""
    for calc in calculators:
        if not isinstance(calc, EnergyModel):
            raise TypeError(f"{calc!r} does not implement EnergyModel")
        if isinstance(calc, DFTCalculator) and len(structure.species) > MAX_DFT_ATOMS:
            continue
        try:
            return EnergyResult(structure.id, calc.energy(structure), calc.name)
        except Exception as exc:
            log.warning("%s failed on %s: %s", calc.name, structure.id, exc)
    raise RuntimeError(f"all calculators failed for {structure.id}")
