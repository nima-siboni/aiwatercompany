"""
Reference fix for PR #253 (python_patterns/exercises/05_factory_registry/L3_engine_plugins.py).
See the answer key for the reasoning.
"""
import functools
import importlib
import inspect
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FORCEFIELD_DIR = Path("/shared/forcefields")
REQUIRED_FIELDS = ("engine", "structure_id")

# Set from the deployment config, never from a request: importing a module runs its code.
ALLOWED_PLUGINS = frozenset({"aiwc_plugins.raspa", "aiwc_plugins.lammps"})

ENGINES: dict[str, type["Engine"]] = {}

_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class Engine(ABC):
    """A simulation code. Concrete engines register explicitly: class X(Engine, name="x")."""

    def __init_subclass__(cls, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if name is None:
            return  # shared base classes such as RaspaEngine are not requestable
        if name in ENGINES:
            other = ENGINES[name]
            where = f"{other.__module__}.{other.__qualname__}"
            raise ValueError(f"engine {name!r} is already registered by {where}")
        ENGINES[name] = cls

    @abstractmethod
    def prepare(self, structure_id: str, **conditions: Any) -> dict[str, str]:
        """Return the input files (file name -> contents) for one job."""


@functools.lru_cache(maxsize=None)
def read_mixing_rules(path: Path) -> str:
    """Parse a force-field file once per process: epsilon (K) and sigma (Å) per atom type."""
    lines = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        atom_type, epsilon, sigma = line.split()
        lines.append(f"{atom_type} lennard-jones {float(epsilon)} {float(sigma)}")
    return "\n".join(lines)


class RaspaEngine(Engine):
    """Shared set-up for RASPA jobs."""

    DEFAULT_CONDITIONS = {"temperature": 298.0, "pressure": 1e5}  # K, Pa

    def __init__(self, cycles: int = 20000, adsorbate: str = "CO2", forcefield: str = "UFF"):
        if not _SAFE_NAME.fullmatch(forcefield):
            raise ValueError(f"invalid force field name {forcefield!r}")
        self.cycles = cycles
        self.adsorbate = adsorbate
        self.forcefield = forcefield
        self.conditions = dict(self.DEFAULT_CONDITIONS)

    @property
    def mixing_rules(self) -> str:
        return read_mixing_rules(FORCEFIELD_DIR / f"{self.forcefield}.def")

    def input_files(
        self, structure_id: str, conditions: dict[str, Any], moves: tuple[str, ...]
    ) -> dict[str, str]:
        unknown = conditions.keys() - self.conditions.keys()
        if unknown:
            raise ValueError(f"unknown conditions: {sorted(unknown)}")
        cond = {**self.conditions, **conditions}  # per job; the engine's defaults stay untouched
        simulation = (
            "SimulationType MonteCarlo\n"
            f"NumberOfCycles {self.cycles}\n"
            f"FrameworkName {structure_id}\n"
            f"ExternalTemperature {cond['temperature']}\n"
            f"ExternalPressure {cond['pressure']}\n"
            f"Component 0 MoleculeName {self.adsorbate}\n"
        )
        simulation += "".join(f"            {move}Probability 1.0\n" for move in moves)
        return {"simulation.input": simulation, "force_field_mixing_rules.def": self.mixing_rules}


class GCMCEngine(RaspaEngine, name="gcmc"):
    """Uptake at a fixed temperature and pressure (grand canonical Monte Carlo)."""

    def prepare(self, structure_id: str, **conditions: Any) -> dict[str, str]:
        return self.input_files(structure_id, conditions, moves=("Translation", "Reinsertion", "Swap"))


class WidomEngine(RaspaEngine, name="widom"):
    """Henry coefficient: how strongly a single gas molecule binds to the framework."""

    def prepare(self, structure_id: str, **conditions: Any) -> dict[str, str]:
        return self.input_files(structure_id, conditions, moves=("Widom",))


def load_plugin(module_path: str) -> None:
    """Import an allow-listed plugin module; its engines register themselves."""
    if module_path not in ALLOWED_PLUGINS:
        raise ValueError(f"plugin {module_path!r} is not enabled on this deployment")
    importlib.import_module(module_path)


def available_engines() -> list[str]:
    """Engine names for the API's GET /engines."""
    return sorted(ENGINES)


def describe_engines() -> dict[str, dict[str, Any]]:
    """Parameters and their defaults for each engine, for the job form in the web UI."""
    return {
        name: {
            p.name: p.default
            for p in inspect.signature(cls).parameters.values()
            if p.default is not inspect.Parameter.empty
        }
        for name, cls in sorted(ENGINES.items())
    }


def get_engine(name: str, params: dict[str, Any]) -> Engine:
    """A new engine per request: engines are cheap, and nothing mutable is shared between jobs."""
    if name not in ENGINES:
        raise ValueError(f"unknown engine {name!r}; available: {available_engines()}")
    return ENGINES[name](**params)


def validate_request(request: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in request]
    if missing:
        raise ValueError(f"job request is missing {missing}")
    # structure_id ends up inside input files, so no newlines, spaces or path separators.
    if not isinstance(request["structure_id"], str) or not _SAFE_NAME.fullmatch(request["structure_id"]):
        raise ValueError(f"invalid structure id {request['structure_id']!r}")


def handle_request(request: dict[str, Any]) -> dict[str, str]:
    """POST /jobs: returns the input files that the scheduler submits."""
    validate_request(request)
    if "plugin" in request:
        load_plugin(request["plugin"])
    engine = get_engine(request["engine"], request.get("params", {}))
    files = engine.prepare(request["structure_id"], **request.get("conditions", {}))
    log.info("prepared %s job for %s", request["engine"], request["structure_id"])
    return files
