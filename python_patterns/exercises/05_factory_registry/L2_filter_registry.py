"""
PR #252: Config-driven screening filters

Before a generated MOF gets GCMC time (our expensive adsorption simulation),
we run cheap filters on descriptors we have already computed for it: largest
pore diameter (Å), density (g/cm³), surface area (m²/g) and the metals it
contains. Filters now register themselves with @register_filter, and each
campaign config lists the ones it wants, e.g.
    [{"name": "pore_size", "min_diameter": 4.0}, {"name": "metals", "allowed": ["Zn", "Cu"]}]
The campaign API validates configs against FILTER_NAMES before queuing them.
The sanity filters for generative-model output live in genfilters.py and
register themselves the same way.
"""
from typing import Iterable

FILTERS: dict[str, type] = {}


def register_filter(name: str):
    """Class decorator: makes a filter available to campaign configs under `name`."""
    def decorator(cls):
        FILTERS[name] = cls
    return decorator


class Filter:
    """Base class for screening filters. Extra config keys such as "comment" are ignored."""

    def __init__(self, **kwargs):
        pass

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


@register_filter("density")
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
        self.allowed = set(allowed)

    def passes(self, record: dict) -> bool:
        return set(record["metals"]) <= self.allowed


FILTER_NAMES = frozenset(FILTERS)


def make_filter(entry: dict) -> Filter:
    name = entry["name"]
    if name not in FILTER_NAMES:
        raise ValueError(f"unknown filter {name!r}")
    params = {key: value for key, value in entry.items() if key != "name"}
    return FILTERS[name](**params)


def screen(candidates: Iterable[dict], config: list[dict]) -> list[dict]:
    """Return the candidates that pass every filter in the campaign config."""
    return [c for c in candidates if all(make_filter(entry).passes(c) for entry in config)]
