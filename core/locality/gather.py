"""Deterministic locality gather — one radius, honest counts.

This replaces the agent's old "decide a radius per category and widen on sparse
results" behaviour, which produced category counts taken at *different* radii than
the radius the prose claimed. Here every everyday category is queried once at a
single, caller-chosen radius, so the numbers the UI shows and the numbers the
blurb is written from are the same, measured at the same distance.

Airports are the one exception: they are essentially never within an everyday
walking/cycling radius, so they get their own fixed wide search
(:data:`AIRPORT_RADIUS_M`) rather than obeying the slider.

The agent (``core/locality/agent.py``) now consumes a :class:`GatheredLocality`
and only writes the prose; it no longer drives the data gathering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.locality.geocode import GeocodeResult
from core.locality.overpass import CATEGORIES, CategoryResult, Poi

# Radius bounds for the everyday-POI slider (metres). Default 3 km per the
# product decision; the slider exposes 1–10 km.
MIN_RADIUS_M = 1000
DEFAULT_RADIUS_M = 3000
MAX_RADIUS_M = 10000

# Airports get a fixed wide search — they're never inside the everyday radius.
AIRPORT_RADIUS_M = 50000

# Everyday categories obey the slider; "airport" is handled separately.
EVERYDAY_CATEGORIES: tuple[str, ...] = tuple(c for c in CATEGORIES if c != "airport")


def clamp_radius(radius_m: int) -> int:
    """Clamp a requested everyday radius into the supported slider range."""
    return max(MIN_RADIUS_M, min(MAX_RADIUS_M, radius_m))


@dataclass
class GatheredLocality:
    """The deterministic gather output: a centre, the radius used, and per-category
    results (everyday categories at ``radius_m``, airports at :data:`AIRPORT_RADIUS_M`)."""

    location_query: str
    center: GeocodeResult
    radius_m: int
    results: dict[str, CategoryResult]

    @property
    def category_counts(self) -> dict[str, int]:
        """True per-category totals, dropping categories with nothing nearby."""
        return {cat: res.total for cat, res in self.results.items() if res.total > 0}

    @property
    def pois(self) -> list[Poi]:
        """The nearest-sample POIs flattened across categories (for the drill-downs)."""
        return [poi for res in self.results.values() for poi in res.pois]


def _location_label(postal_code: str, street: str | None, country_code: str) -> str:
    """A human-readable query label kept for persistence/display."""
    head = f"{postal_code} {street}".strip() if street else postal_code
    return f"{head}, {country_code.upper()}"


def gather_locality(
    *,
    geocode_client: Any,
    overpass_client: Any,
    postal_code: str,
    street: str | None = None,
    country_code: str = "DE",
    radius_m: int = DEFAULT_RADIUS_M,
    sample_limit: int = 5,
) -> GatheredLocality | None:
    """Geocode the PIN, then gather every category once at a single radius.

    Returns ``None`` when the location cannot be geocoded. Everyday categories use
    the clamped ``radius_m``; airports use :data:`AIRPORT_RADIUS_M`.
    """
    radius_m = clamp_radius(radius_m)
    center = geocode_client.geocode(postal_code, street=street, country_code=country_code)
    if center is None:
        return None

    results: dict[str, CategoryResult] = {}
    for category in EVERYDAY_CATEGORIES:
        results[category] = overpass_client.gather(
            lat=center.latitude,
            lon=center.longitude,
            radius_m=radius_m,
            category=category,
            sample_limit=sample_limit,
        )
    results["airport"] = overpass_client.gather(
        lat=center.latitude,
        lon=center.longitude,
        radius_m=AIRPORT_RADIUS_M,
        category="airport",
        sample_limit=sample_limit,
    )

    return GatheredLocality(
        location_query=_location_label(postal_code, street, country_code),
        center=center,
        radius_m=radius_m,
        results=results,
    )
