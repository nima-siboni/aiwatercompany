"""
Reference fix for PR #262 (python_patterns/exercises/06_strategy_template/L2_relaxer_template.py).
See the answer key for the reasoning.
"""
from abc import ABC, abstractmethod

import numpy as np


class RelaxationNotConverged(RuntimeError):
    """run() hit max_steps before the largest force dropped below fmax."""


class BaseRelaxer(ABC):
    def __init__(self, calculator, fmax: float = 0.05, max_steps: int = 1000):
        self.calculator = calculator
        self.fmax = fmax
        self.max_steps = max_steps
        self.history: list[float] = []

    def setup(self) -> None:
        """Called at the start of every run(), once the subclass is fully constructed."""
        self.calculator.start()

    def teardown(self) -> None:
        """Called at the end of every run(), including when it fails."""
        self.calculator.stop()

    @abstractmethod
    def step(self, positions: np.ndarray, forces: np.ndarray) -> np.ndarray:
        """Return new positions. Must not modify `positions` in place."""

    def converged(self, forces: np.ndarray) -> bool:
        """The contract for every subclass: the largest per-atom force is below fmax (eV/Å)."""
        return np.linalg.norm(forces, axis=1).max() < self.fmax

    def run(self, positions: np.ndarray) -> tuple[np.ndarray, float]:
        """Relax one structure. Returns the relaxed positions and E_initial - E_final (eV per cell)."""
        positions = np.array(positions, dtype=float)  # a copy: the caller's array is never touched
        self.history = []
        self.setup()
        try:
            energy, forces = self.calculator.energy_and_forces(positions)
            self.history.append(energy)
            n_steps = 0
            while not self.converged(forces):
                if n_steps == self.max_steps:
                    worst = np.linalg.norm(forces, axis=1).max()
                    raise RelaxationNotConverged(f"max force {worst:.3f} eV/Å after {n_steps} steps")
                positions = self.step(positions, forces)
                energy, forces = self.calculator.energy_and_forces(positions)
                self.history.append(energy)
                n_steps += 1
        finally:
            self.teardown()
        return positions, self.history[0] - energy


class SteepestDescent(BaseRelaxer):
    def __init__(self, calculator, fmax: float = 0.05, step_size: float = 0.01, **kwargs):
        super().__init__(calculator, fmax, **kwargs)
        self.step_size = step_size

    def step(self, positions, forces):
        return positions + self.step_size * forces


class DampedDynamics(BaseRelaxer):
    """Steps with velocity and friction; much faster than steepest descent on soft modes."""

    def __init__(self, calculator, fmax: float = 0.05, dt: float = 0.1, friction: float = 0.1, **kwargs):
        super().__init__(calculator, fmax, **kwargs)
        self.dt = dt
        self.friction = friction

    def setup(self) -> None:
        super().setup()
        self.velocities = None  # fresh for every structure

    def step(self, positions, forces):
        if self.velocities is None:
            self.velocities = np.zeros_like(positions)
        self.velocities = (1.0 - self.friction) * self.velocities + self.dt * forces
        return positions + self.dt * self.velocities


class MLRelaxer(DampedDynamics):
    """For the ML potential. Its forces carry ~0.05 eV/Å of noise on single atoms, so it uses a
    looser fmax by default. The criterion itself (largest per-atom force) stays the same."""

    def __init__(self, calculator, fmax: float = 0.1, **kwargs):
        super().__init__(calculator, fmax, **kwargs)
