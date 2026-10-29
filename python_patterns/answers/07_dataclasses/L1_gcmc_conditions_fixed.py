"""
Reference fix for PR #271 (python_patterns/exercises/07_dataclasses/L1_gcmc_conditions.py).
See the answer key for the reasoning.
"""
from dataclasses import dataclass, field, replace


@dataclass(frozen=True, order=True)
class Condition:
    temperature: float  # K
    pressure: float  # bar
    gas: str = "CO2"

    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError(f"temperature is in K and must be > 0, got {self.temperature}")
        if self.pressure <= 0:
            raise ValueError(f"pressure is in bar and must be > 0, got {self.pressure}")
        # One canonical float per temperature, so -196.15 °C and 77 K are the same key.
        object.__setattr__(self, "temperature", round(self.temperature, 6))

    @classmethod
    def from_celsius(cls, temp_c: float, pressure: float, gas: str = "CO2") -> "Condition":
        return cls(temp_c + 273.15, pressure, gas)


@dataclass
class JobRequest:
    mof_id: str
    condition: Condition
    priority: int = 0
    tags: list[str] = field(default_factory=list)


def pressure_sweep(base: Condition, pressures: list[float]) -> list[Condition]:
    """One condition per pressure, at the temperature and gas of `base`."""
    return [replace(base, pressure=p) for p in pressures]


def isotherm_jobs(mof_id: str, base: Condition, pressures: list[float], priority: int = 0) -> list[JobRequest]:
    return [JobRequest(mof_id, cond, priority) for cond in pressure_sweep(base, pressures)]


def unique_conditions(jobs: list[JobRequest]) -> set[Condition]:
    """The distinct conditions in a batch, so each force-field setup is prepared once."""
    return {job.condition for job in jobs}


def schedule(jobs: list[JobRequest]) -> list[JobRequest]:
    """Highest priority first; jobs with equal priority keep their submission order."""
    return sorted(jobs, key=lambda job: job.priority, reverse=True)
