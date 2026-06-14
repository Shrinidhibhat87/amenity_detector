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

import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from core.locality.geocode import GeocodeResult
from core.locality.overpass import CATEGORIES, CategoryResult, Poi

logger = logging.getLogger(__name__)

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


def _safe_gather(overpass_client: Any, *, category: str, **kwargs: Any) -> CategoryResult:
    """Gather one category, isolating failures.

    A single slow or failing Overpass category (timeout, 429, the heavy airport
    query) must not abort the whole enrichment — it degrades to an empty result
    for that category while the rest still render. The error is logged, not raised.
    """
    try:
        result: CategoryResult = overpass_client.gather(category=category, **kwargs)
        return result
    except Exception as exc:
        logger.warning("overpass gather failed for category %r: %s", category, exc)
        return CategoryResult(category=category, total=0, pois=[])


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

    @property
    def transit_breakdown(self) -> dict[str, int] | None:
        """Per-mode counts for the transit category (bus/tram/rail/…), or None."""
        transit = self.results.get("transit")
        return transit.subtype_counts if transit else None


def location_label(postal_code: str, street: str | None, country_code: str) -> str:
    """A human-readable query label kept for persistence/display."""
    head = f"{postal_code} {street}".strip() if street else postal_code
    return f"{head}, {country_code.upper()}"


@dataclass
class GatherProgress:
    """One category finished — emitted by :func:`iter_gather` for progress UIs."""

    category: str
    done: int
    total: int
    result: CategoryResult


def iter_gather(
    *,
    geocode_client: Any,
    overpass_client: Any,
    postal_code: str,
    street: str | None = None,
    country_code: str = "DE",
    radius_m: int = DEFAULT_RADIUS_M,
    sample_limit: int = 5,
) -> Generator[GatherProgress, None, GatheredLocality | None]:
    """Stream the gather one category at a time.

    Yields a :class:`GatherProgress` after each category completes (so a progress
    bar can fill determinately) and *returns* the assembled :class:`GatheredLocality`
    via ``StopIteration.value`` — or ``None`` if the PIN could not be geocoded.
    Geocode errors propagate to the caller; per-category Overpass failures are
    isolated by :func:`_safe_gather`.
    """
    radius_m = clamp_radius(radius_m)
    center = geocode_client.geocode(postal_code, street=street, country_code=country_code)
    if center is None:
        return None

    steps: list[tuple[str, int]] = [(c, radius_m) for c in EVERYDAY_CATEGORIES]
    steps.append(("airport", AIRPORT_RADIUS_M))
    total = len(steps)

    results: dict[str, CategoryResult] = {}
    for index, (category, radius) in enumerate(steps, start=1):
        result = _safe_gather(
            overpass_client,
            lat=center.latitude,
            lon=center.longitude,
            radius_m=radius,
            category=category,
            sample_limit=sample_limit,
        )
        results[category] = result
        yield GatherProgress(category=category, done=index, total=total, result=result)

    return GatheredLocality(
        location_query=location_label(postal_code, street, country_code),
        center=center,
        radius_m=radius_m,
        results=results,
    )


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
    the clamped ``radius_m``; airports use :data:`AIRPORT_RADIUS_M`. This drains
    :func:`iter_gather` for callers that don't need the per-category progress.
    """
    gen = iter_gather(
        geocode_client=geocode_client,
        overpass_client=overpass_client,
        postal_code=postal_code,
        street=street,
        country_code=country_code,
        radius_m=radius_m,
        sample_limit=sample_limit,
    )
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        result: GatheredLocality | None = stop.value
        return result
