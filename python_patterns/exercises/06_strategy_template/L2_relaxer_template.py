"""
PR #262: Common base class for structure relaxations

A relaxation moves the atoms downhill in energy until the largest force on any
atom is below fmax (eV/Å). We had three copies of this loop, so
BaseRelaxer.run() now owns it and subclasses only supply the step rule. The
calculator is either our ML potential or a DFT code (slow, quantum-mechanical)
in interactive mode: start() launches it, energy_and_forces(positions)
evaluates one geometry, and stop() shuts it down. Each screening worker builds
one relaxer and calls run() for every structure. run() returns the relaxed positions and the relaxation
energy (eV), which we use as a stability descriptor for generated MOFs.
"""
import numpy as np


class BaseRelaxer:
    history = []

    def __init__(self, calculator, fmax: float = 0.05):
        self.calculator = calculator
        self.fmax = fmax
        self.setup()

    def setup(self) -> None:
        """Called before the first step."""
        self.calculator.start()

    def teardown(self) -> None:
        """Called after the last step."""
        self.calculator.stop()

    def step(self, positions: np.ndarray, forces: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def converged(self, forces: np.ndarray) -> bool:
        return np.linalg.norm(forces, axis=1).max() < self.fmax

    def run(self, positions: np.ndarray) -> tuple[np.ndarray, float]:
        energy, forces = self.calculator.energy_and_forces(positions)
        self.history.append(energy)
        while not self.converged(forces):
            positions = self.step(positions, forces)
            energy, forces = self.calculator.energy_and_forces(positions)
            self.history.append(energy)
        self.teardown()
        return positions, self.history[0] - energy


class SteepestDescent(BaseRelaxer):
    def __init__(self, calculator, fmax: float = 0.05, step_size: float = 0.01):
        super().__init__(calculator, fmax)
        self.step_size = step_size

    def step(self, positions, forces):
        positions += self.step_size * forces
        return positions


class DampedDynamics(BaseRelaxer):
    """Steps with velocity and friction; much faster than steepest descent on soft modes."""

    def __init__(self, calculator, fmax: float = 0.05, dt: float = 0.1, friction: float = 0.1):
        super().__init__(calculator, fmax)
        self.dt = dt
        self.friction = friction

    def setup(self) -> None:
        super().setup()
        self.velocities = None

    def step(self, positions, forces):
        if self.velocities is None:
            self.velocities = np.zeros_like(positions)
        self.velocities = (1.0 - self.friction) * self.velocities + self.dt * forces
        return positions + self.dt * self.velocities


class MLRelaxer(DampedDynamics):
    """For the ML potential. Its forces carry ~0.05 eV/Å of noise on single atoms,
    so the max-force test rarely passes; the mean over atoms is much more stable."""

    def converged(self, forces):
        return np.linalg.norm(forces, axis=1).mean() < self.fmax
