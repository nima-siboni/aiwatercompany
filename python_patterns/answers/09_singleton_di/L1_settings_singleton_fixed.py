"""
Reference fix for PR #291 (python_patterns/exercises/09_singleton_di/L1_settings_singleton.py).
See the answer key for the reasoning.
"""
import json
import os
from functools import cache

DEFAULT_RESULTS_DB = "/shared/aiwc/results.sqlite"
DEFAULT_TEMPERATURE_K = 298.0


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """Process-wide configuration for the screening services."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            instance = super().__new__(cls)
            # Read the environment once, when the instance is created. There is no
            # __init__, so later Settings() calls can't reset overrides.
            instance._load_from_env()
            cls._instance = instance
        return cls._instance

    def _load_from_env(self) -> None:
        self.results_db = os.environ.get("AIWC_RESULTS_DB", DEFAULT_RESULTS_DB)
        self.gcmc_cycles = int(os.environ.get("AIWC_GCMC_CYCLES", "5000"))
        self.temperature_k = DEFAULT_TEMPERATURE_K
        self.dry_run = _env_flag("AIWC_DRY_RUN")
        self.api_token = os.environ.get("AIWC_API_TOKEN", "")

    def override(self, **values) -> "Settings":
        unknown = values.keys() - vars(self).keys()
        if unknown:
            raise AttributeError(f"unknown settings: {sorted(unknown)}")
        for name, value in values.items():
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        shown = {k: v for k, v in vars(self).items() if k != "api_token"}
        return f"Settings({shown}, api_token='***')"


@cache
def element_table(path: str) -> dict[str, dict[str, float]]:
    """Element properties keyed by symbol, e.g. table["Zn"]["electronegativity"]."""
    with open(path) as f:
        return json.load(f)
