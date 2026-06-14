"""Nominatim geocode client.

Turns a PIN code (plus an optional street, within a given country) into a
coordinate plus a bounding box, using the free OpenStreetMap Nominatim service.
The structured query (``postalcode`` / ``street`` / ``countrycodes``) keeps a
PIN from resolving to an identical code in the wrong country.

Compliance is not optional here — the Nominatim usage policy requires:
  - a descriptive ``User-Agent`` identifying this application,
  - at most 1 request per second,
  - caching results so we don't re-query unchanged data.

So the client enforces a 1 req/s rate limit, sends a mandatory User-Agent, and
sits behind a pluggable cache. The cache is a small Protocol: this module ships
an in-memory implementation; a Postgres-backed adapter is wired in alongside the
``locality_insight`` tables so geocodes survive restarts (POIs don't move).

The HTTP session and the clock/sleep functions are injected so the unit tests
run fully offline and never actually wait a second between calls.

Environment variables (used by :meth:`GeocodeClient.from_env`):
  NOMINATIM_BASE_URL   — defaults to the public endpoint.
  NOMINATIM_USER_AGENT — required; descriptive UA with contact info.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, Self, runtime_checkable

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://nominatim.openstreetmap.org"
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class GeocodeError(RuntimeError):
    """Raised when Nominatim cannot be reached after exhausting retries."""


@dataclass(frozen=True)
class GeocodeResult:
    """A resolved location.

    ``bbox`` is ``(south, north, west, east)`` in decimal degrees — the same
    corner order Nominatim returns, kept so callers can size an Overpass search
    radius from the locality's actual extent.
    """

    query: str  # normalised cache key the result was resolved from
    latitude: float
    longitude: float
    bbox: tuple[float, float, float, float]
    display_name: str


@runtime_checkable
class GeocodeCache(Protocol):
    """Minimal cache contract — get/set keyed on the normalised query string."""

    def get(self, key: str) -> GeocodeResult | None: ...

    def set(self, key: str, value: GeocodeResult) -> None: ...


class InMemoryGeocodeCache:
    """Process-local cache. Fine for tests and single-process dev runs."""

    def __init__(self) -> None:
        self._store: dict[str, GeocodeResult] = {}

    def get(self, key: str) -> GeocodeResult | None:
        return self._store.get(key)

    def set(self, key: str, value: GeocodeResult) -> None:
        self._store[key] = value


def _norm(text: str) -> str:
    """Collapse whitespace + lowercase so trivially-different inputs share a key."""
    return " ".join(text.split()).lower()


def _cache_key(postal_code: str, street: str | None, country_code: str) -> str:
    """Stable key across country + street so identical PINs in different countries
    (e.g. DE ``10115`` vs US ``10115``) never collide on the same cached centre."""
    parts = [_norm(country_code), _norm(postal_code)]
    if street:
        parts.append(_norm(street))
    return "|".join(parts)


class GeocodeClient:
    """Free-text → :class:`GeocodeResult`, rate-limited and cached."""

    def __init__(
        self,
        *,
        session: Any,
        cache: GeocodeCache,
        user_agent: str,
        base_url: str = _DEFAULT_BASE_URL,
        min_interval_seconds: float = 1.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        timeout_seconds: float = 10.0,
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
    def from_env(cls, *, cache: GeocodeCache | None = None) -> Self:
        import requests  # imported lazily so unit tests never need the dependency

        user_agent = os.getenv("NOMINATIM_USER_AGENT")
        if not user_agent:
            raise RuntimeError(
                "NOMINATIM_USER_AGENT is required (Nominatim policy mandates a "
                "descriptive User-Agent with contact info). See .env.example."
            )
        return cls(
            session=requests.Session(),
            cache=cache or InMemoryGeocodeCache(),
            user_agent=user_agent,
            base_url=os.getenv("NOMINATIM_BASE_URL", _DEFAULT_BASE_URL),
        )

    def geocode(
        self,
        postal_code: str,
        *,
        street: str | None = None,
        country_code: str = "DE",
    ) -> GeocodeResult | None:
        """Resolve a PIN (+ optional street, within ``country_code``) to a coordinate.

        Uses Nominatim's *structured* query (``postalcode`` / ``street`` /
        ``countrycodes``) rather than a free-text ``q`` blob, so a PIN resolves to
        the right country instead of the first global match. Returns ``None`` when
        Nominatim finds nothing; raises :class:`GeocodeError` only when the service
        is unreachable after all retries.
        """
        pc = postal_code.strip()
        if not pc:
            return None
        cc = (country_code or "DE").strip() or "DE"
        st = street.strip() if street and street.strip() else None

        key = _cache_key(pc, st, cc)
        cached = self._cache.get(key)
        if cached is not None:
            logger.debug("geocode cache hit for %r", key)
            return cached

        payload = self._request(pc, st, cc)
        if not payload:
            return None

        top = payload[0]
        if len(payload) > 1:
            logger.info("geocode %r was ambiguous (%d hits); taking the top", key, len(payload))

        result = _parse_hit(key, top)
        if result is not None:
            self._cache.set(key, result)
        return result

    def _request(
        self, postal_code: str, street: str | None, country_code: str
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "postalcode": postal_code,
            "countrycodes": country_code.lower(),
            "format": "jsonv2",
            "limit": 5,  # ask for a few so we can log ambiguity; we use the top hit
            "addressdetails": 0,
        }
        if street:
            params["street"] = street
        headers = {"User-Agent": self._user_agent}

        last_status: int | None = None
        for attempt in range(self._max_retries + 1):
            self._respect_rate_limit()
            try:
                response = self._session.get(
                    f"{self._base_url}/search",
                    params=params,
                    headers=headers,
                    timeout=self._timeout,
                )
            except Exception as exc:  # network-level failure → retry
                last_status = None
                logger.warning("geocode request error (attempt %d): %s", attempt, exc)
            else:
                if response.status_code == 200:
                    data = response.json()
                    return data if isinstance(data, list) else []
                last_status = response.status_code
                if response.status_code not in _RETRYABLE_STATUS:
                    raise GeocodeError(
                        f"Nominatim returned non-retryable status {response.status_code}"
                    )
                logger.warning(
                    "geocode transient status %d (attempt %d)", response.status_code, attempt
                )

            if attempt < self._max_retries:
                self._sleep(self._backoff_base * (2**attempt))

        raise GeocodeError(
            f"Nominatim unreachable after {self._max_retries + 1} attempts "
            f"(last status: {last_status})"
        )

    def _respect_rate_limit(self) -> None:
        """Sleep just enough to keep consecutive network calls >= min_interval apart."""
        now = self._clock()
        if self._last_call_at is not None:
            elapsed = now - self._last_call_at
            remaining = self._min_interval - elapsed
            if remaining > 0:
                self._sleep(remaining)
        self._last_call_at = self._clock()


def _parse_hit(query: str, hit: dict[str, Any]) -> GeocodeResult | None:
    """Convert one Nominatim hit into a :class:`GeocodeResult`, or ``None`` if malformed."""
    try:
        lat = float(hit["lat"])
        lon = float(hit["lon"])
        south, north, west, east = (float(v) for v in hit["boundingbox"])
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("geocode hit for %r was malformed: %s", query, exc)
        return None

    return GeocodeResult(
        query=query,
        latitude=lat,
        longitude=lon,
        bbox=(south, north, west, east),
        display_name=str(hit.get("display_name", "")),
    )
