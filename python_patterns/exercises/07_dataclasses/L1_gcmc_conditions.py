"""
PR #271: Value objects for GCMC conditions and job requests

GCMC (grand canonical Monte Carlo) simulations predict how much gas a MOF
adsorbs at a given temperature and pressure. This replaces the loose dicts we
pass to the job scheduler with small dataclasses, so conditions can be
deduplicated, used as cache keys and sorted. `isotherm_jobs` builds the jobs
for one isotherm (the same MOF and temperature at several pressures), and the
scheduler submits jobs in the order `schedule()` returns them.
"""
from dataclasses import dataclass, field


@dataclass(order=True, unsafe_hash=True)
class Condition:
    temperature: float  # K
    pressure: float  # bar
    gas: str = "CO2"

    @classmethod
    def from_celsius(cls, temp_c: float, pressure: float, gas: str = "CO2") -> "Condition":
        return cls(temp_c + 273.15, pressure, gas)


@dataclass(order=True)
class JobRequest:
    mof_id: str
    condition: Condition
    priority: int = 0
    tags: list[str] = field(default_factory=list)


def pressure_sweep(base: Condition, pressures: list[float]) -> list[Condition]:
    """One condition per pressure, at the temperature and gas of `base`."""
    sweep = []
    for p in pressures:
        base.pressure = p
        sweep.append(base)
    return sweep


def isotherm_jobs(mof_id: str, base: Condition, pressures: list[float], priority: int = 0) -> list[JobRequest]:
    return [JobRequest(mof_id, cond, priority) for cond in pressure_sweep(base, pressures)]


def unique_conditions(jobs: list[JobRequest]) -> set[Condition]:
    """The distinct conditions in a batch, so each force-field setup is prepared once."""
    return {job.condition for job in jobs}


def schedule(jobs: list[JobRequest]) -> list[JobRequest]:
    """Highest priority first; jobs with equal priority keep their submission order."""
    return sorted(jobs, reverse=True)
