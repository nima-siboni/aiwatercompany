"""Checks for the 06_strategy_template drills: every planted bug fails on the PR code and passes on the fix."""
from dataclasses import dataclass

import numpy as np
import pytest

from harness import bug, fine


def _raised(exc_type, fn, *args, **kwargs) -> bool:
    """True if fn(*args, **kwargs) raises exc_type (other exceptions propagate)."""
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    return False


# --- L1: acquisition-function strategies -----------------------------------------------------

L1 = "06_strategy_template/L1_acquisition"

MEAN = np.array([1.0, 5.0, 9.0, 4.0])
STD = np.array([0.5, 0.1, 0.2, 2.0])


@bug(L1)
def test_L1_target_window_direction(m):
    """Blocker 1: TargetUptake returns a distance (lower is better), so select() picks the furthest."""
    assert list(m.select(m.TargetUptake(target=5.0), MEAN, STD, batch_size=1)) == [1]


@bug(L1)
def test_L1_unimplemented_score_fails_loudly(m):
    """Major 2: a strategy without a working score() silently selects candidate 0 (argsort of NotImplemented)."""

    class Misnamed(m.Acquisition):
        def scores(self, mean, std):  # typo: never called
            return mean

    assert _raised((TypeError, NotImplementedError), lambda: m.select(Misnamed(), MEAN, STD, batch_size=2))


@bug(L1)
def test_L1_zero_batch_selects_nothing(m):
    """Major 3: batch_size=0 (budget used up) gives [-0:], which selects every candidate."""
    assert len(m.select(m.Greedy(), MEAN, STD, batch_size=0)) == 0


@bug(L1, raises=TypeError)
def test_L1_new_cost_aware_strategy_needs_no_select_change(m):
    """Minor 4: select() special-cases CostAwareUCB, so another strategy that needs costs never gets them."""

    class CostAwareGreedy(m.Acquisition):
        def score(self, mean, std, cost=None):
            return mean / cost

    cost = np.array([1.0, 1.0, 10.0, 1.0])
    assert list(m.select(CostAwareGreedy(), MEAN, STD, batch_size=1, cost=cost)) == [1]


@bug(L1)
def test_L1_nan_scores_not_selected(m):
    """Minor 5: NaN sorts last in argsort, so a NaN prediction is treated as the best candidate."""
    mean = np.array([1.0, np.nan, 3.0])
    assert list(m.select(m.Greedy(), mean, np.zeros(3), batch_size=1)) == [2]


@bug(L1)
def test_L1_best_candidate_first(m):
    """Nit 6: the selected indices come back worst first."""
    assert list(m.select(m.Greedy(), MEAN, STD, batch_size=2)) == [2, 1]


@fine(L1)
def test_L1_ucb_beta_scales_std(m):
    """Red herring: UCB uses beta * std (beta is the multiplier, not its square root); it ranks as intended."""
    assert list(m.select(m.UCB(beta=2.0), MEAN, STD, batch_size=2)) in ([2, 3], [3, 2])
    assert list(m.select(m.UCB(beta=0.0), MEAN, STD, batch_size=1)) == [2]


# --- L2: template-method relaxer -------------------------------------------------------------

L2 = "06_strategy_template/L2_relaxer_template"


class TooManyCalls(Exception):
    pass


class CalculatorCrash(Exception):
    pass


class Harmonic:
    """E = k/2 |x|^2, forces -k x; start()/stop() tracked like the DFT driver's."""

    def __init__(self, k=10.0, require_started=True, fail_after=None, max_calls=5000, constant_force=None):
        self.k = k
        self.require_started = require_started
        self.fail_after = fail_after
        self.max_calls = max_calls
        self.constant_force = constant_force
        self.running = False
        self.calls = 0

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def energy_and_forces(self, positions):
        if self.require_started and not self.running:
            raise RuntimeError("calculator is not running")
        self.calls += 1
        if self.calls > self.max_calls:
            raise TooManyCalls
        if self.fail_after is not None and self.calls > self.fail_after:
            raise CalculatorCrash("SCF blew up")
        if self.constant_force is not None:  # a slope that never flattens out
            return -float(positions.sum()) * self.constant_force, np.full_like(positions, self.constant_force)
        return 0.5 * self.k * float((positions**2).sum()), -self.k * positions


START_A = np.full((2, 3), 0.5)  # E = 7.5 eV
START_B = np.full((3, 3), 0.2)  # E = 1.8 eV


@bug(L2)
def test_L2_relaxation_energy_per_run(m):
    """Blocker 1: history is a class attribute, so later runs subtract the first structure's energy."""
    calc = Harmonic(require_started=False)
    _, e_a = m.SteepestDescent(calc).run(START_A.copy())
    _, e_b = m.SteepestDescent(calc).run(START_B.copy())
    reused = m.SteepestDescent(calc)
    reused.run(START_A.copy())
    _, e_b_again = reused.run(START_B.copy())
    assert e_a == pytest.approx(7.5, abs=0.01)
    assert e_b == pytest.approx(1.8, abs=0.01)
    assert e_b_again == pytest.approx(1.8, abs=0.01)


@bug(L2)
def test_L2_input_positions_untouched(m):
    """Blocker 2: SteepestDescent updates positions in place, overwriting the caller's starting geometry."""
    given = START_A.copy()
    m.SteepestDescent(Harmonic(require_started=False)).run(given)
    assert np.array_equal(given, START_A)


@bug(L2)
def test_L2_ml_relaxer_keeps_max_force_contract(m):
    """Blocker 3: MLRelaxer's mean-force test calls a structure with one 1 eV/Å atom converged."""
    forces = np.zeros((400, 3))
    forces[0] = [1.0, 0.0, 0.0]
    assert not m.MLRelaxer(Harmonic(require_started=False)).converged(forces)


@bug(L2)
def test_L2_step_limit(m):
    """Major 4: no max_steps, so a relaxation that never converges loops forever."""
    calc = Harmonic(require_started=False, constant_force=1.0)
    try:
        m.SteepestDescent(calc).run(START_A.copy())
    except TooManyCalls:
        raise AssertionError("run() kept stepping until the test's call limit") from None
    except Exception:
        pass  # a clear "did not converge" error is fine


@bug(L2)
def test_L2_teardown_on_failure(m):
    """Major 5: teardown() isn't in a finally, so a crash mid-run leaves the DFT process running."""
    calc = Harmonic(fail_after=3)
    relaxer = m.SteepestDescent(calc)
    with pytest.raises(CalculatorCrash):
        relaxer.run(START_A.copy())
    assert not calc.running


@bug(L2, raises=RuntimeError)
def test_L2_setup_runs_for_every_structure(m):
    """Major 6: setup() runs once in __init__, so the worker's second run() uses a stopped calculator."""
    relaxer = m.DampedDynamics(Harmonic())
    relaxer.run(START_A.copy())
    positions, _ = relaxer.run(START_B.copy())
    assert positions.shape == (3, 3)


@bug(L2)
def test_L2_base_is_abstract(m):
    """Minor 7: BaseRelaxer can be built (and starts the calculator) although it has no step()."""
    calc = Harmonic()
    assert _raised(TypeError, m.BaseRelaxer, calc)


@fine(L2)
def test_L2_force_step_sign(m):
    """Red herring: positions + step_size * forces goes downhill, because forces are minus the gradient."""
    positions, relaxation_energy = m.SteepestDescent(Harmonic(require_started=False)).run(START_A.copy())
    assert np.abs(positions).max() < 0.01
    assert relaxation_energy > 7.4


@fine(L2)
def test_L2_damped_dynamics_returns_new_array(m):
    """Red herring: DampedDynamics returns positions + ..., a new array, so the caller's input is safe."""
    given = START_A.copy()
    positions, _ = m.DampedDynamics(Harmonic()).run(given)
    assert np.array_equal(given, START_A)
    assert np.abs(positions).max() < 0.01


# --- L3: calculator hierarchy with a fallback chain ------------------------------------------

L3 = "06_strategy_template/L3_energy_fallback"


@dataclass
class Structure:
    id: str
    species: tuple[str, ...]


MOF = Structure("mof-1", ("Zn",) * 2 + ("O",) * 6)  # 8 atoms


def per_atom(value):
    return lambda structure: np.full(len(structure.species), value)


class Broken:
    """Satisfies the protocol but has a programming error inside."""

    name = "broken"

    def supports(self, structure):
        return True

    def energy(self, structure):
        raise TypeError("unsupported operand type(s) for +: 'float' and 'NoneType'")


@bug(L3)
def test_L3_surrogate_returns_cell_energy(m):
    """Blocker 1: the surrogate returns eV per atom while every other calculator returns eV per cell."""
    surrogate = m.CompositionSurrogate(coef={"Zn": -1.0, "O": -2.0})
    assert surrogate.energy(MOF) == pytest.approx(-14.0)  # 8 atoms x -1.75 eV/atom


@bug(L3)
def test_L3_dft_actually_used(m):
    """Blocker 2: RetryMixin.__init__ skips super(), so every DFT call fails and the chain silently uses ML."""
    calls = []

    def run_dft(structure, functional):
        calls.append(functional)
        return -42.0

    result = m.compute_energy(MOF, [m.DFTCalculator(run_dft), m.MLPotential(per_atom(-1.0))])
    assert (result.source, result.energy, calls) == ("dft", -42.0, ["PBE"])


@bug(L3)
def test_L3_cache_sees_model_update(m):
    """Blocker 3: the cache is keyed on structure.id only, so update_model() leaves stale energies."""
    ml = m.MLPotential(per_atom(-1.0))
    assert ml.energy(MOF) == -8.0
    ml.update_model(per_atom(-2.0))
    assert ml.energy(MOF) == -16.0


@bug(L3)
def test_L3_programming_errors_propagate(m):
    """Major 4: except Exception turns a bug in a calculator into a silent fallback."""
    assert _raised(TypeError, m.compute_energy, MOF, [Broken(), m.MLPotential(per_atom(-1.0))])


@bug(L3)
def test_L3_non_finite_energy_falls_back(m):
    """Major 5: a NaN from the ML potential is cached and returned instead of falling back."""
    surrogate = m.CompositionSurrogate(coef={"Zn": -1.0, "O": -2.0})
    result = m.compute_energy(MOF, [m.MLPotential(per_atom(np.nan)), surrogate])
    assert result.source == "surrogate"


@bug(L3)
def test_L3_surrogate_rejects_unseen_elements(m):
    """Major 6: elements the surrogate never saw silently get a zero coefficient."""
    surrogate = m.CompositionSurrogate(coef={"Zn": -1.0, "O": -2.0})
    exotic = Structure("mof-2", ("Pu",) * 2 + ("O",) * 6)
    assert _raised(Exception, surrogate.energy, exotic)


@bug(L3)
def test_L3_size_limit_belongs_to_the_calculator(m):
    """Minor 7: compute_energy only knows DFTCalculator's limit; another calculator can't declare its own."""

    class SmallCellCode:
        name = "small_cell_dft"

        def supports(self, structure):
            return len(structure.species) <= 4

        def energy(self, structure):
            return -1.0

    result = m.compute_energy(MOF, [SmallCellCode(), m.MLPotential(per_atom(-1.0))])
    assert result.source == "ml"


@bug(L3)
def test_L3_timings_bounded(m):
    """Minor 9: timings gains one entry per call, forever."""
    ml = m.MLPotential(per_atom(-1.0))
    for _ in range(20_000):
        ml.energy(MOF)
    assert len(ml.timings) < 20_000


@bug(L3)
def test_L3_final_error_keeps_causes(m):
    """Minor 8: when every calculator fails, the error doesn't say why."""

    def lost_node(structure):
        raise m.TransientError("GPU node lost")

    with pytest.raises(RuntimeError) as excinfo:
        m.compute_energy(MOF, [m.MLPotential(lost_node)])
    assert "GPU node lost" in str(excinfo.value)


@fine(L3)
def test_L3_retry_loop_never_returns_none(m):
    """Red herring: RetryMixin re-raises on the last attempt, so it can't fall through and return None."""

    class Flaky(m.RetryMixin, m.Calculator):
        name = "flaky"

        def __init__(self, failures):
            m.Calculator.__init__(self)
            self.retries = 2
            self.failures = failures

        def _compute(self, structure):
            if self.failures:
                self.failures -= 1
                raise m.TransientError("node lost")
            return -3.0

    assert Flaky(failures=2).energy(MOF) == -3.0
    assert _raised(m.TransientError, Flaky(failures=3).energy, MOF)


@fine(L3)
def test_L3_super_in_parentless_mixin(m):
    """Red herring: super().energy() in TimingMixin follows the instance's MRO to Calculator.energy."""
    ml = m.MLPotential(per_atom(-1.0))
    assert ml.energy(MOF) == -8.0
    assert len(ml.timings) == 1


@fine(L3)
def test_L3_protocol_check_only_checks_names(m):
    """Red herring: the runtime_checkable isinstance passes the per-atom surrogate; it isn't a units check."""
    assert isinstance(m.CompositionSurrogate(coef={"Zn": -1.0}), m.EnergyModel)
