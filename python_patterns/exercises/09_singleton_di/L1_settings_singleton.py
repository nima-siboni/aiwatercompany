"""
PR #291: One shared Settings object for the screening services

Every service was reading its own environment variables and the defaults had
drifted apart, so this adds a single `Settings` object: call `Settings()`
anywhere and you get the same instance. Tests and notebooks can change a value
with `Settings().override(results_db="scratch.sqlite")`. It also caches the
element-property table that the composition featurisers use, because parsing it
takes a few seconds.
"""
import json
import os
from functools import cache

RESULTS_DB = os.environ.get("AIWC_RESULTS_DB", "/shared/aiwc/results.sqlite")
DEFAULT_TEMPERATURE_K = 298.0


class Settings:
    """Process-wide configuration for the screening services."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.results_db = RESULTS_DB
        self.gcmc_cycles = int(os.environ.get("AIWC_GCMC_CYCLES", "5000"))
        self.temperature_k = DEFAULT_TEMPERATURE_K
        self.dry_run = os.environ.get("AIWC_DRY_RUN", False)
        self.api_token = os.environ.get("AIWC_API_TOKEN", "")

    def override(self, **values) -> "Settings":
        for name, value in values.items():
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        return f"Settings({self.__dict__})"


@cache
def element_table(path: str) -> dict[str, dict[str, float]]:
    """Element properties keyed by symbol, e.g. table["Zn"]["electronegativity"]."""
    with open(path) as f:
        return json.load(f)
