"""Locality enrichment agent — neighbourhood prose from gathered facts.

The numbers are now gathered deterministically (``core/locality/gather.py``):
every category is counted once at a single radius, so counts can't drift across
radii or be capped. This module's remaining job is the one genuinely *language*
task: turn that structured summary into a short, factual "Lage" paragraph.

The agent makes a single constrained completion over OpenRouter. It is told the
radius, the per-category counts, and the closest few places, and must only
describe what it was given — no invented places, counts, or distances. If the LLM
is unavailable or returns nothing, a deterministic fallback blurb is synthesised
from the same counts, so the pipeline never depends on the model being up.

The OpenAI-SDK client is injected (pointed at OpenRouter in production via
:meth:`LocalityAgent.from_env`); unit tests inject a scripted fake.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Self

from core.locality.gather import (
    AIRPORT_RADIUS_M,
    DEFAULT_RADIUS_M,
    GatheredLocality,
    clamp_radius,
    gather_locality,
    iter_gather,
    location_label,
)
from core.locality.overpass import Poi

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "openai/gpt-4o-mini"
_ATTRIBUTION = "© OpenStreetMap contributors"

# Human labels for the OSM category keys (mirrors core/locality/overpass.py).
_CATEGORY_LABELS: dict[str, str] = {
    "school": "schools",
    "gym": "gyms",
    "supermarket": "supermarkets",
    "park": "parks",
    "transit": "transit stops",
    "pharmacy": "pharmacies",
    "airport": "airports",
}

_SYSTEM_PROMPT = """You write the "Lage" (neighbourhood) section of a property listing.

You are given a structured summary of what is near a location: a search radius
and, per category, how many places are nearby plus the closest few by name and
distance. Write a short, factual paragraph (2-4 sentences) describing the
neighbourhood for a prospective tenant.

Rules:
  - Only mention categories that actually have nearby results.
  - Use the counts and distances given. Never invent places, counts or distances.
  - State distances in kilometres. Keep it concrete and natural — no marketing fluff.
"""


@dataclass
class LocalityResult:
    """The agent's output — everything needed to persist + render the Lage section."""

    location_query: str
    display_name: str
    latitude: float | None
    longitude: float | None
    radius_m: int
    pois: list[Poi]
    category_counts: dict[str, int]
    blurb: str
    transit_breakdown: dict[str, int] | None = None
    attribution: str = _ATTRIBUTION


class LocalityAgent:
    """Gathers neighbourhood facts deterministically, then writes the Lage blurb."""

    def __init__(
        self,
        *,
        openai_client: Any,
        geocode_client: Any,
        overpass_client: Any,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._client = openai_client
        self._geocode = geocode_client
        self._overpass = overpass_client
        self._model = model

    @classmethod
    def from_env(cls, *, geocode_client: Any, overpass_client: Any) -> Self:
        from openai import OpenAI  # imported lazily so unit tests can mock it

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required to build a LocalityAgent.")
        timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "60"))
        client = OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=api_key, timeout=timeout)
        model = os.getenv("LOCALITY_AGENT_MODEL", _DEFAULT_MODEL)
        return cls(
            openai_client=client,
            geocode_client=geocode_client,
            overpass_client=overpass_client,
            model=model,
        )

    def run(
        self,
        *,
        postal_code: str,
        street: str | None = None,
        country_code: str = "DE",
        radius_m: int = DEFAULT_RADIUS_M,
    ) -> LocalityResult:
        """Gather what is nearby at ``radius_m`` and return the enriched result."""
        gathered = gather_locality(
            geocode_client=self._geocode,
            overpass_client=self._overpass,
            postal_code=postal_code,
            street=street,
            country_code=country_code,
            radius_m=radius_m,
        )
        if gathered is None:
            return LocalityResult(
                location_query=location_label(postal_code, street, country_code),
                display_name="",
                latitude=None,
                longitude=None,
                radius_m=clamp_radius(radius_m),
                pois=[],
                category_counts={},
                blurb="This location could not be found.",
            )

        return self._to_result(gathered, self._write_blurb(gathered))

    def run_streaming(
        self,
        *,
        postal_code: str,
        street: str | None = None,
        country_code: str = "DE",
        radius_m: int = DEFAULT_RADIUS_M,
    ) -> Iterator[dict[str, Any]]:
        """Same enrichment as :meth:`run`, but yielding progress events.

        Emits a ``category`` event as each category completes, a ``summary`` event
        while the blurb is written, and a final ``result`` event carrying the
        :class:`LocalityResult`. A geocode miss yields a single ``result`` with an
        empty payload. Geocode service errors propagate to the caller (the router
        turns them into an ``error`` event).
        """
        gen = iter_gather(
            geocode_client=self._geocode,
            overpass_client=self._overpass,
            postal_code=postal_code,
            street=street,
            country_code=country_code,
            radius_m=radius_m,
        )
        gathered: GatheredLocality | None = None
        try:
            while True:
                progress = next(gen)
                yield {
                    "type": "category",
                    "category": progress.category,
                    "done": progress.done,
                    "total": progress.total,
                    "count": progress.result.total,
                }
        except StopIteration as stop:
            gathered = stop.value

        if gathered is None:
            yield {
                "type": "result",
                "result": LocalityResult(
                    location_query=location_label(postal_code, street, country_code),
                    display_name="",
                    latitude=None,
                    longitude=None,
                    radius_m=clamp_radius(radius_m),
                    pois=[],
                    category_counts={},
                    blurb="This location could not be found.",
                ),
            }
            return

        yield {"type": "summary"}
        yield {"type": "result", "result": self._to_result(gathered, self._write_blurb(gathered))}

    def _write_blurb(self, gathered: GatheredLocality) -> str:
        """Ask the model for prose from the gathered summary; fall back deterministically."""
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _summarize(gathered)},
                ],
                temperature=0,
            )
            text = (completion.choices[0].message.content or "").strip()
        except Exception as exc:  # model/network failure → deterministic fallback
            logger.warning("locality blurb synthesis failed (%s); using fallback", exc)
            text = ""
        return text or _fallback_blurb(gathered.category_counts, gathered.radius_m)

    @staticmethod
    def _to_result(gathered: GatheredLocality, blurb: str) -> LocalityResult:
        return LocalityResult(
            location_query=gathered.location_query,
            display_name=gathered.center.display_name,
            latitude=gathered.center.latitude,
            longitude=gathered.center.longitude,
            radius_m=gathered.radius_m,
            pois=gathered.pois,
            category_counts=gathered.category_counts,
            blurb=blurb,
            transit_breakdown=gathered.transit_breakdown,
        )


def _summarize(gathered: GatheredLocality) -> str:
    """Render the gathered facts as a compact prompt the model writes prose from."""
    lines = [f"Search radius: {gathered.radius_m / 1000:g} km."]
    for category, res in gathered.results.items():
        if res.total == 0:
            continue
        label = _CATEGORY_LABELS.get(category, category)
        nearest = ", ".join(
            f"{p.name or '(unnamed)'} {p.distance_m / 1000:.1f} km" for p in res.pois[:3]
        )
        scope = " (within 50 km)" if category == "airport" else ""
        lines.append(f"{label}{scope}: {res.total} nearby. Closest: {nearest}.")
        if category == "transit" and res.subtype_counts:
            modes = ", ".join(f"{n} {mode}" for mode, n in res.subtype_counts.items())
            lines.append(f"  transit modes: {modes}.")
    return "\n".join(lines)


def _fallback_blurb(counts: dict[str, int], radius_m: int) -> str:
    """Deterministic blurb used when the model produced nothing."""
    everyday = {c: n for c, n in counts.items() if c != "airport"}
    if not everyday:
        return "No notable amenities were found nearby."
    parts = [f"{n} {_CATEGORY_LABELS.get(cat, cat)}" for cat, n in everyday.items()]
    blurb = f"Within {radius_m / 1000:g} km: " + ", ".join(parts) + "."
    if counts.get("airport"):
        blurb += f" Nearest airport within {AIRPORT_RADIUS_M // 1000} km."
    return blurb
