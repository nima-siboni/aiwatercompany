"""
PR #273: Typed, frozen configs for active-learning campaigns

An active-learning (AL) campaign repeatedly trains a surrogate model on the MOFs
labelled so far, picks the next batch with an acquisition function (UCB, EI or
greedy), and labels that batch with GCMC simulations, which compute CO2 uptake.
Settings used to be nested dicts. This PR replaces them with frozen dataclasses
that load from JSON, accept `AL_*` environment-variable overrides from our Slurm
scripts (`AL_BUDGET=200`, `AL_SIMULATION__TEMPERATURE=273`), are saved next to
the results, and get a short `run_id` so we can find earlier runs with the same
settings. With `use_cached_labels` on, uptakes already in the results store are
reused instead of re-running GCMC.
"""
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

DEFAULT_FORCEFIELD = {"name": "UFF", "cutoff": 12.8, "charges": "EQeq"}


@dataclass(frozen=True)
class SimulationConfig:
    temperature: float = 298.0  # K
    pressure: float = 1.0  # bar
    forcefield: dict = field(default_factory=lambda: DEFAULT_FORCEFIELD)

    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError(f"temperature is in K and must be > 0, got {self.temperature}")
        if not 0 < self.pressure <= 1000:
            raise ValueError(f"pressure is in bar and must be in (0, 1000], got {self.pressure}")


@dataclass(frozen=True)
class GCMCConfig(SimulationConfig):
    n_cycles: int = 5000
    n_init_cycles: int = 1000

    def __post_init__(self):
        if self.n_init_cycles >= self.n_cycles:
            raise ValueError("n_init_cycles must be smaller than n_cycles")


@dataclass(frozen=True)
class SurrogateConfig:
    hidden: int = 128
    lr: float = 1e-3
    ensemble_size: int = 5

    def __post_init__(self):
        if self.hidden < 1 or self.ensemble_size < 1:
            raise ValueError("hidden and ensemble_size must be at least 1")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")


@dataclass(frozen=True)
class CampaignConfig:
    name: str
    simulation: GCMCConfig = GCMCConfig()
    surrogate: SurrogateConfig = SurrogateConfig()
    acquisition: str = "ucb"
    beta: float = 2.0
    batch_size: int = 32
    budget: int = 500  # total number of GCMC simulations
    seed: int = 0
    use_cached_labels: bool = True
    output_dir: str = "runs"
    notes: str = field(default="", compare=False)

    def __post_init__(self):
        if self.acquisition not in ("ucb", "ei", "greedy"):
            raise ValueError(f"unknown acquisition function {self.acquisition!r}")
        if not 0 < self.batch_size <= self.budget:
            raise ValueError("need 0 < batch_size <= budget")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CampaignConfig":
        return cls(**d)

    @property
    def run_id(self) -> str:
        """Short, stable hash of the settings, used to find earlier runs."""
        blob = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @property
    def run_dir(self) -> str:
        return os.path.join(self.output_dir, f"{self.name}-{self.run_id}")

    def summary(self) -> str:
        sim = self.simulation
        return (f"{self.name} [{self.run_id}]: {self.acquisition}, budget {self.budget}, "
                f"GCMC at {sim.temperature} K / {sim.pressure} bar")

    def save(self) -> str:
        os.makedirs(self.run_dir, exist_ok=True)
        path = os.path.join(self.run_dir, "config.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


def _set_path(obj, path: list[str], value) -> None:
    *parents, last = path
    for part in parents:
        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
    if isinstance(obj, dict):
        obj[last] = value
    else:
        object.__setattr__(obj, last, value)  # configs are frozen


def load_config(path: str, environ=os.environ) -> CampaignConfig:
    """Load a campaign config from JSON, then apply AL_* environment overrides."""
    with open(path) as f:
        cfg = CampaignConfig.from_dict(json.load(f))
    for key, value in environ.items():
        if key.startswith("AL_"):
            _set_path(cfg, key[3:].lower().split("__"), value)
    return cfg


def find_previous_run(cfg: CampaignConfig) -> str | None:
    """The directory of an earlier run with the same settings, if there is one."""
    path = os.path.join(cfg.run_dir, "config.json")
    return cfg.run_dir if os.path.exists(path) else None
