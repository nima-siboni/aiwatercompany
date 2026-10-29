"""
Reference fix for PR #252 (python_patterns/exercises/05_factory_registry/L2_filter_registry.py).
See the answer key for the reasoning.

Registration is still an import side effect, so the package that owns this module
must import genfilters.py (e.g. in its __init__.py) before any campaign runs.
"""
from typing import Iterable

FILTERS: dict[str, type["Filter"]] = {}


def register_filter(name: str):
    """Class decorator: makes a filter available to campaign configs under `name`."""
    def decorator(cls):
        if name in FILTERS:
            raise ValueError(f"filter name {name!r} is already taken by {FILTERS[name].__qualname__}")
        FILTERS[name] = cls
        return cls
    return decorator


class Filter:
    """Base class for screening filters. The only extra config key allowed is "comment"."""

    def __init__(self, comment: str = ""):
        self.comment = comment

    def passes(self, record: dict) -> bool:
        raise NotImplementedError


@register_filter("pore_size")
class PoreSizeFilter(Filter):
    """The gas has to fit: CO2 needs a pore of at least ~3.3 Å."""

    def __init__(self, min_diameter: float = 3.3, max_diameter: float = 30.0, **kwargs):
        super().__init__(**kwargs)
        self.min_diameter = min_diameter
        self.max_diameter = max_diameter

    def passes(self, record: dict) -> bool:
        return self.min_diameter <= record["pore_diameter"] <= self.max_diameter


@register_filter("density")
class DensityFilter(Filter):
    """Very dense frameworks leave little room for gas."""

    def __init__(self, max_density: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.max_density = max_density

    def passes(self, record: dict) -> bool:
        return record["density"] <= self.max_density


@register_filter("surface_area")
class SurfaceAreaFilter(Filter):
    """Uptake roughly scales with surface area, so drop the low ones."""

    def __init__(self, min_area: float = 500.0, **kwargs):
        super().__init__(**kwargs)
        self.min_area = min_area

    def passes(self, record: dict) -> bool:
        return record["surface_area"] >= self.min_area


@register_filter("metals")
class MetalFilter(Filter):
    """Keeps frameworks whose metals are all allowed (cost, toxicity, water stability)."""

    def __init__(self, allowed: Iterable[str] = ("Zn", "Cu", "Zr", "Al", "Mg"), **kwargs):
        super().__init__(**kwargs)
        if isinstance(allowed, str):
            raise TypeError(f"allowed must be a list of element symbols, not the string {allowed!r}")
        self.allowed = set(allowed)

    def passes(self, record: dict) -> bool:
        return set(record["metals"]) <= self.allowed


# A live view, not a snapshot: it also sees filters registered after this module ran.
FILTER_NAMES = FILTERS.keys()


def make_filter(entry: dict) -> Filter:
    if "name" not in entry:
        raise ValueError(f"filter config {entry!r} has no 'name'")
    name = entry["name"]
    if name not in FILTER_NAMES:
        raise ValueError(f"unknown filter {name!r}; registered filters: {sorted(FILTER_NAMES)}")
    params = {key: value for key, value in entry.items() if key != "name"}
    return FILTERS[name](**params)


def screen(candidates: Iterable[dict], config: list[dict]) -> list[dict]:
    """Return the candidates that pass every filter in the campaign config."""
    filters = [make_filter(entry) for entry in config]  # build and validate once, even for no candidates
    return [c for c in candidates if all(f.passes(c) for f in filters)]
