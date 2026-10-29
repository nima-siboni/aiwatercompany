"""
Reference fix for PR #263 (python_patterns/exercises/06_strategy_template/L3_energy_fallback.py).
See the answer key for the reasoning.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from aiwc.structures import Structure

log = logging.getLogger(__name__)

MAX_DFT_ATOMS = 400


class CalculationError(Exception):
    """An expected failure of one calculator on one structure; compute_energy() falls back."""


class TransientError(CalculationError):
    """A failure worth retrying, e.g. the cluster lost the node."""


@runtime_checkable
class EnergyModel(Protocol):
    name: str

    def supports(self, structure: Structure) -> bool: ...

    def energy(self, structure: Structure) -> float: ...


class Calculator:
    """Base class: energy() caches results; subclasses implement _compute().

    Contract for every subclass: _compute() returns the formation energy in eV per unit cell,
    and raises CalculationError when it can't. A subclass whose results change for the same
    structure (e.g. a retrained model) must bump self.version.
    """

    name = "base"
    max_atoms: int | None = None

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int], float] = {}
        self.version = 0

    def supports(self, structure: Structure) -> bool:
        return self.max_atoms is None or len(structure.species) <= self.max_atoms

    def energy(self, structure: Structure) -> float:
        """Formation energy in eV per unit cell."""
        key = (structure.id, self.version)
        if key not in self._cache:
            value = self._compute(structure)
            if not math.isfinite(value):
                raise CalculationError(f"{self.name} returned {value} for {structure.id}")
            self._cache[key] = value
        return self._cache[key]

    def _compute(self, structure: Structure) -> float:
        raise NotImplementedError


class TimingMixin:
    """Records the wall time of recent calls; the service dashboard reads .timings."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.timings: deque[float] = deque(maxlen=10_000)

    def energy(self, structure: Structure) -> float:
        start = time.perf_counter()
        try:
            return super().energy(structure)
        finally:
            self.timings.append(time.perf_counter() - start)


class RetryMixin:
    """Retries calls that fail with TransientError. `retries` counts the extra attempts after the first."""

    def __init__(self, retries: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
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
    max_atoms = MAX_DFT_ATOMS

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

    def update_model(self, model: Callable[[Structure], np.ndarray]) -> None:
        """Swap in the potential retrained this round."""
        self.model = model
        self.version += 1  # part of the cache key, so old predictions are not reused

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
        unseen = set(structure.species) - self.coef.keys()
        if unseen:
            raise CalculationError(f"surrogate has never seen {sorted(unseen)}")
        per_atom = self.intercept + sum(
            self.coef[el] * structure.species.count(el) / n for el in set(structure.species)
        )
        return per_atom * n  # the model is per atom; the contract is per unit cell


@dataclass
class EnergyResult:
    structure_id: str
    energy: float  # eV per unit cell
    source: str


def compute_energy(structure: Structure, calculators: list[EnergyModel]) -> EnergyResult:
    """Return the energy from the first calculator in the list that succeeds."""
    for calc in calculators:
        if not isinstance(calc, EnergyModel):  # only checks that the methods exist
            raise TypeError(f"{calc!r} does not implement EnergyModel")
    errors = []
    for calc in calculators:
        if not calc.supports(structure):
            errors.append(f"{calc.name}: skipped ({len(structure.species)} atoms)")
            continue
        try:
            return EnergyResult(structure.id, calc.energy(structure), calc.name)
        except CalculationError as exc:  # programming errors propagate instead of falling back
            log.warning("%s failed on %s: %s", calc.name, structure.id, exc)
            errors.append(f"{calc.name}: {exc}")
    raise RuntimeError(f"all calculators failed for {structure.id}: " + "; ".join(errors))
