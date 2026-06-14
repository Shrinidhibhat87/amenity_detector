"""Overpass POI client.

Given a coordinate + radius, returns the nearby points of interest in one of a
fixed set of neighbourhood-relevant categories (schools, gyms, groceries, parks,
transit stops, pharmacies), using the free OpenStreetMap Overpass API.

Each category maps to one or more OSM tag selectors (``key=value``). We build a
single OverpassQL query per category that unions nodes + ways + relations
matching any selector within ``around:<radius>`` of the centre, ask for way/
relation centroids via ``out center``, then parse, compute great-circle distance
from the centre, sort nearest-first and cap the count.

Results are cached on ``(lat, lon, radius, category)`` — POIs don't move, so a
hit means zero network cost and no pressure on Overpass fair-use limits. The
in-memory cache ships here; a Postgres adapter is wired in with the locality
tables.

Environment variables (used by :meth:`OverpassClient.from_env`):
  OVERPASS_BASE_URL    — defaults to the public interpreter endpoint.
  NOMINATIM_USER_AGENT — reused as the descriptive User-Agent.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, Self, runtime_checkable

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://overpass-api.de/api"
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_DEFAULT_LIMIT = 25

# Category → list of (tag-key, tag-value) selectors. A POI matches a category if
# it carries ANY of the category's selectors. Kept deliberately small and
# mainstream so the blurb talks about things people actually search for.
CATEGORY_FILTERS: dict[str, tuple[tuple[str, str], ...]] = {
    "school": (("amenity", "school"),),
    "gym": (("leisure", "fitness_centre"), ("leisure", "sports_centre")),
    "supermarket": (
        ("shop", "supermarket"),
        ("shop", "convenience"),
        ("shop", "grocery"),
    ),
    "park": (("leisure", "park"),),
    "transit": (
        ("highway", "bus_stop"),
        ("railway", "tram_stop"),
        ("railway", "station"),
        ("station", "subway"),
    ),
    "pharmacy": (("amenity", "pharmacy"),),
    "airport": (("aeroway", "aerodrome"),),
}

# Public tuple of advertised categories.
CATEGORIES: tuple[str, ...] = tuple(CATEGORY_FILTERS)


@dataclass(frozen=True)
class Poi:
    """A single point of interest returned by Overpass."""

    category: str
    name: str  # may be empty — many parks / bus stops are unnamed in OSM
    latitude: float
    longitude: float
    osm_type: str  # "node" | "way" | "relation"
    osm_id: int
    distance_m: float
    # For transit POIs: "bus" | "tram" | "subway" | "light_rail" | "rail".
    # None for every other category (and for transit stops we can't classify).
    transit_type: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CategoryResult:
    """A category's nearby POIs at a fixed radius: the true total + a nearest sample.

    ``total`` is the real count at ``radius`` (NOT capped) so the UI never shows a
    cap-as-count. ``pois`` is the nearest handful for the drill-down list.
    """

    category: str
    total: int
    pois: list[Poi]


def _transit_type(tags: Mapping[str, str]) -> str | None:
    """Classify a transit stop from its OSM tags, or None if unclear."""
    if tags.get("highway") == "bus_stop":
        return "bus"
    if tags.get("railway") == "tram_stop" or tags.get("tram") == "yes":
        return "tram"
    if tags.get("station") == "subway" or tags.get("subway") == "yes":
        return "subway"
    if tags.get("station") == "light_rail" or tags.get("light_rail") == "yes":
        return "light_rail"
    if tags.get("railway") == "station":
        return "rail"
    return None


@runtime_checkable
class PoiCache(Protocol):
    """Cache contract keyed on a (lat, lon, radius, category) string."""

    def get(self, key: str) -> list[Poi] | None: ...

    def set(self, key: str, value: list[Poi]) -> None: ...


class InMemoryPoiCache:
    """Process-local POI cache for tests and single-process dev runs."""

    def __init__(self) -> None:
        self._store: dict[str, list[Poi]] = {}

    def get(self, key: str) -> list[Poi] | None:
        return self._store.get(key)

    def set(self, key: str, value: list[Poi]) -> None:
        self._store[key] = value


def _cache_key(lat: float, lon: float, radius_m: int, category: str) -> str:
    # Round the coordinate to ~11 m so trivially-different centres share a key.
    return f"{round(lat, 4)},{round(lon, 4)}|{radius_m}|{category}"


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _build_query(lat: float, lon: float, radius_m: int, category: str) -> str:
    selectors = CATEGORY_FILTERS[category]
    lines: list[str] = []
    for key, value in selectors:
        for kind in ("node", "way", "relation"):
            lines.append(f'  {kind}["{key}"="{value}"](around:{radius_m},{lat},{lon});')
    body = "\n".join(lines)
    # `out center` gives ways/relations a centroid; tags come along by default.
    return f"[out:json][timeout:25];\n(\n{body}\n);\nout center;"


class OverpassClient:
    """Coordinate + radius + category → distance-sorted list of :class:`Poi`."""

    def __init__(
        self,
        *,
        session: Any,
        cache: PoiCache,
        user_agent: str,
        base_url: str = _DEFAULT_BASE_URL,
        min_interval_seconds: float = 1.0,
        max_retries: int = 2,
        backoff_base_seconds: float = 1.0,
        timeout_seconds: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._session = session
        self._cache = cache
        self._user_agent = user_agent
        self._base_url = base_url.rstrip("/")
        self._min_interval = min_interval_seconds
        self._max_retries = max_retries
        self._backoff_base = backoff_base_seconds
        self._timeout = timeout_seconds
        self._sleep = sleep
        self._clock = clock
        self._last_call_at: float | None = None

    @classmethod
    def from_env(cls, *, cache: PoiCache | None = None) -> Self:
        import requests  # imported lazily so unit tests never need the dependency

        user_agent = os.getenv("NOMINATIM_USER_AGENT")
        if not user_agent:
            raise RuntimeError(
                "NOMINATIM_USER_AGENT is required (used as the Overpass User-Agent too)."
            )
        return cls(
            session=requests.Session(),
            cache=cache or InMemoryPoiCache(),
            user_agent=user_agent,
            base_url=os.getenv("OVERPASS_BASE_URL", _DEFAULT_BASE_URL),
        )

    def query(
        self,
        *,
        lat: float,
        lon: float,
        radius_m: int,
        category: str,
        limit: int = _DEFAULT_LIMIT,
    ) -> list[Poi]:
        """Return nearby POIs in ``category``, nearest-first, capped at ``limit``."""
        return self._all_pois(lat=lat, lon=lon, radius_m=radius_m, category=category)[:limit]

    def gather(
        self,
        *,
        lat: float,
        lon: float,
        radius_m: int,
        category: str,
        sample_limit: int = 5,
    ) -> CategoryResult:
        """Return the true count at ``radius_m`` plus the ``sample_limit`` nearest POIs.

        The count is honest — it is the full number of matches within the radius,
        not the length of the (capped) sample list — so the panel can show "12
        supermarkets" while only listing the closest 5.
        """
        pois = self._all_pois(lat=lat, lon=lon, radius_m=radius_m, category=category)
        return CategoryResult(category=category, total=len(pois), pois=pois[:sample_limit])

    def _all_pois(self, *, lat: float, lon: float, radius_m: int, category: str) -> list[Poi]:
        """Full distance-sorted POI list for a category (cached, uncapped)."""
        if category not in CATEGORY_FILTERS:
            raise ValueError(
                f"unknown category {category!r}; expected one of {', '.join(CATEGORIES)}"
            )

        key = _cache_key(lat, lon, radius_m, category)
        cached = self._cache.get(key)
        if cached is not None:
            logger.debug("overpass cache hit for %s", key)
            return cached

        elements = self._request(_build_query(lat, lon, radius_m, category))
        pois = self._parse(elements, lat, lon, category)
        pois.sort(key=lambda p: p.distance_m)
        self._cache.set(key, pois)
        return pois

    def _parse(
        self, elements: Sequence[Mapping[str, Any]], lat: float, lon: float, category: str
    ) -> list[Poi]:
        out: list[Poi] = []
        for el in elements:
            coord = _element_coord(el)
            if coord is None:
                continue
            plat, plon = coord
            tags = el.get("tags", {}) or {}
            out.append(
                Poi(
                    category=category,
                    name=str(tags.get("name", "")),
                    latitude=plat,
                    longitude=plon,
                    osm_type=str(el.get("type", "")),
                    osm_id=int(el.get("id", 0)),
                    distance_m=_haversine_m(lat, lon, plat, plon),
                    transit_type=_transit_type(tags) if category == "transit" else None,
                    tags=dict(tags),
                )
            )
        return out

    def _request(self, ql: str) -> list[dict[str, Any]]:
        headers = {"User-Agent": self._user_agent}
        last_status: int | None = None
        for attempt in range(self._max_retries + 1):
            self._respect_rate_limit()
            try:
                response = self._session.post(
                    f"{self._base_url}/interpreter",
                    data={"data": ql},
                    headers=headers,
                    timeout=self._timeout,
                )
            except Exception as exc:
                last_status = None
                logger.warning("overpass request error (attempt %d): %s", attempt, exc)
            else:
                if response.status_code == 200:
                    payload = response.json()
                    elements = payload.get("elements", []) if isinstance(payload, dict) else []
                    return elements if isinstance(elements, list) else []
                last_status = response.status_code
                if response.status_code not in _RETRYABLE_STATUS:
                    raise OverpassError(
                        f"Overpass returned non-retryable status {response.status_code}"
                    )
                logger.warning(
                    "overpass transient status %d (attempt %d)", response.status_code, attempt
                )

            if attempt < self._max_retries:
                self._sleep(self._backoff_base * (2**attempt))

        raise OverpassError(
            f"Overpass unreachable after {self._max_retries + 1} attempts "
            f"(last status: {last_status})"
        )

    def _respect_rate_limit(self) -> None:
        now = self._clock()
        if self._last_call_at is not None:
            remaining = self._min_interval - (now - self._last_call_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_call_at = self._clock()


class OverpassError(RuntimeError):
    """Raised when Overpass cannot be reached after exhausting retries."""


def _element_coord(el: Mapping[str, Any]) -> tuple[float, float] | None:
    """Pull a (lat, lon) from a node (direct) or a way/relation (``center``)."""
    if "lat" in el and "lon" in el:
        try:
            return float(el["lat"]), float(el["lon"])
        except (TypeError, ValueError):
            return None
    center = el.get("center")
    if isinstance(center, Mapping) and "lat" in center and "lon" in center:
        try:
            return float(center["lat"]), float(center["lon"])
        except (TypeError, ValueError):
            return None
    return None
