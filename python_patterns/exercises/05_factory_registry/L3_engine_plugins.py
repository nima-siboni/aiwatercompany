"""
PR #253: Plugin engines for the simulation service

The simulation service turns job requests from the internal API into input
files for the cluster scheduler. Engines are now plugins: every subclass of
Engine registers itself, and a request names the module that provides its
engine, so teams can ship new engines without touching the service:
    {"plugin": "aiwc_plugins.raspa", "engine": "gcmc", "params": {"cycles": 50000},
     "structure_id": "mof-00017", "conditions": {"temperature": 77}}
This PR ports our two RASPA job types: GCMC (gas uptake at a given temperature
and pressure) and Widom insertion (a cheap estimate of how strongly the gas
binds). Engines are created on first use and then reused, so we don't re-read
the force-field files (Lennard-Jones parameters per atom type) for every job.
"""
import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FORCEFIELD_DIR = Path("/shared/forcefields")
REQUIRED_FIELDS = ("engine", "structure_id")

ENGINES: dict[str, type["Engine"]] = {}
_instances: dict[str, "Engine"] = {}


class Engine(ABC):
    """A simulation code. Subclasses register under their class name minus "Engine"."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        ENGINES[cls.__name__.lower().removesuffix("engine")] = cls

    @abstractmethod
    def prepare(self, structure_id: str, **conditions: Any) -> dict[str, str]:
        """Return the input files (file name -> contents) for one job."""


class RaspaEngine(Engine):
    """Shared set-up for RASPA jobs."""

    DEFAULT_CONDITIONS = {"temperature": 298.0, "pressure": 1e5}  # K, Pa

    def __init__(self, cycles: int = 20000, adsorbate: str = "CO2", forcefield: str = "UFF"):
        self.cycles = cycles
        self.adsorbate = adsorbate
        self.forcefield = forcefield
        self.conditions = dict(self.DEFAULT_CONDITIONS)

    @property
    def mixing_rules(self) -> str:
        """The force field in RASPA's format: epsilon (K) and sigma (Å) per atom type."""
        lines = []
        for line in (FORCEFIELD_DIR / f"{self.forcefield}.def").read_text().splitlines():
            atom_type, epsilon, sigma = line.split()
            lines.append(f"{atom_type} lennard-jones {float(epsilon)} {float(sigma)}")
        return "\n".join(lines)

    def input_files(
        self, structure_id: str, conditions: dict[str, Any], moves: tuple[str, ...]
    ) -> dict[str, str]:
        unknown = conditions.keys() - self.conditions.keys()
        if unknown:
            raise ValueError(f"unknown conditions: {sorted(unknown)}")
        cond = self.conditions
        cond.update(conditions)
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


class GCMCEngine(RaspaEngine):
    """Uptake at a fixed temperature and pressure (grand canonical Monte Carlo)."""

    def prepare(self, structure_id: str, **conditions: Any) -> dict[str, str]:
        return self.input_files(structure_id, conditions, moves=("Translation", "Reinsertion", "Swap"))


class WidomEngine(RaspaEngine):
    """Henry coefficient: how strongly a single gas molecule binds to the framework."""

    def prepare(self, structure_id: str, **conditions: Any) -> dict[str, str]:
        return self.input_files(structure_id, conditions, moves=("Widom",))


def load_plugin(module_path: str) -> None:
    """Import a plugin module; its Engine subclasses register themselves."""
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
    engine = _instances.get(name)
    if engine is None:
        engine = ENGINES[name](**params)
        _instances[name] = engine
    return engine


def validate_request(request: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in request]
    if missing:
        raise ValueError(f"job request is missing {missing}")


def handle_request(request: dict[str, Any]) -> dict[str, str]:
    """POST /jobs: returns the input files that the scheduler submits."""
    validate_request(request)
    if "plugin" in request:
        load_plugin(request["plugin"])
    engine = get_engine(request["engine"], request.get("params", {}))
    files = engine.prepare(request["structure_id"], **request.get("conditions", {}))
    log.info("prepared %s job for %s", request["engine"], request["structure_id"])
    return files
