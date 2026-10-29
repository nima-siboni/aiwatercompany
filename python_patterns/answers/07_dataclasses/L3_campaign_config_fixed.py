"""
Reference fix for PR #273 (python_patterns/exercises/07_dataclasses/L3_campaign_config.py).
See the answer key for the reasoning.
"""
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from functools import cached_property

DEFAULT_FORCEFIELD = {"name": "UFF", "cutoff": 12.8, "charges": "EQeq"}
_BOOLS = {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}


@dataclass(frozen=True)
class SimulationConfig:
    temperature: float = 298.0  # K
    pressure: float = 1.0  # bar
    forcefield: dict = field(default_factory=lambda: dict(DEFAULT_FORCEFIELD))

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
        super().__post_init__()
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
    # Run options rather than settings: left out of equality and of run_id.
    use_cached_labels: bool = field(default=True, compare=False)
    output_dir: str = field(default="runs", compare=False)
    notes: str = field(default="", compare=False)

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9][\w.-]*", self.name):
            raise ValueError(f"name must be a plain directory name, got {self.name!r}")
        if self.acquisition not in ("ucb", "ei", "greedy"):
            raise ValueError(f"unknown acquisition function {self.acquisition!r}")
        if not 0 < self.batch_size <= self.budget:
            raise ValueError("need 0 < batch_size <= budget")
        if self.acquisition != "greedy" and self.surrogate.ensemble_size < 2:
            raise ValueError(f"{self.acquisition} needs ensemble_size >= 2 to estimate uncertainty")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CampaignConfig":
        return _build(cls, d)

    @cached_property
    def run_id(self) -> str:
        """Short, stable hash of the settings, used to find earlier runs."""
        d = self.to_dict()
        settings = {f.name: d[f.name] for f in fields(self) if f.compare}
        blob = json.dumps(settings, sort_keys=True)
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


def _parse_bool(value: str) -> bool:
    try:
        return _BOOLS[value.strip().lower()]
    except KeyError:
        raise ValueError(f"expected a boolean, got {value!r}") from None


def _build(cls, data: dict):
    """Build a config dataclass from JSON-like data: nested sections, declared types, no unknown keys."""
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"unknown {cls.__name__} settings: {unknown}")
    kwargs = {}
    for name, value in data.items():
        f = known[name]
        if is_dataclass(f.type):
            kwargs[name] = _build(f.type, value)
        elif f.type is dict:
            kwargs[name] = {**f.default_factory(), **value}  # a partial section overrides key by key
        elif f.type is bool and isinstance(value, str):
            kwargs[name] = _parse_bool(value)
        else:
            kwargs[name] = f.type(value)
    return cls(**kwargs)


def _parse_env(value: str):
    """'273' -> 273, 'false' -> False, '12.8' -> 12.8; anything else stays a string."""
    try:
        return json.loads(value)
    except ValueError:
        return value


def _set_path(d: dict, path: list[str], value) -> None:
    *parents, last = path
    for part in parents:
        d = d.setdefault(part, {})
    d[last] = value


def load_config(path: str, environ=os.environ) -> CampaignConfig:
    """Load a campaign config from JSON, then apply AL_* environment overrides."""
    with open(path) as f:
        raw = json.load(f)
    for key, value in environ.items():
        if key.startswith("AL_"):
            _set_path(raw, key[3:].lower().split("__"), _parse_env(value))
    # Overrides go into the raw dict and the config is built once, so every value is
    # converted to its declared type and every __post_init__ check runs.
    return CampaignConfig.from_dict(raw)


def find_previous_run(cfg: CampaignConfig) -> str | None:
    """The directory of an earlier run with the same settings, if there is one."""
    path = os.path.join(cfg.run_dir, "config.json")
    return cfg.run_dir if os.path.exists(path) else None
