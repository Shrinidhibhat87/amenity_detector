"""Unit tests for the Nominatim geocode client.

The client is exercised entirely against a fake HTTP session so the tests run
offline and never touch the real Nominatim service. We assert on four things
the plan calls out: cache hit/miss, ambiguous-result handling, the mandatory
User-Agent header + 1 req/s rate limit, and retry/backoff on transient errors.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.locality.geocode import (
    GeocodeClient,
    GeocodeError,
    GeocodeResult,
    InMemoryGeocodeCache,
)


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    """Records every GET and replays a scripted queue of responses."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _FakeResponse:
        self.calls.append({"url": url, "params": params, "headers": headers})
        if not self._responses:
            raise AssertionError("FakeSession ran out of scripted responses")
        return self._responses.pop(0)


def _hit(
    *,
    lat: str = "50.1109",
    lon: str = "8.6821",
    bbox: list[str] | None = None,
    name: str = "60311 Frankfurt am Main, Germany",
) -> dict[str, Any]:
    return {
        "lat": lat,
        "lon": lon,
        # Nominatim order: south, north, west, east (all strings).
        "boundingbox": bbox or ["50.0969", "50.1249", "8.6621", "8.7021"],
        "display_name": name,
    }


def _make_client(
    session: _FakeSession,
    *,
    cache: InMemoryGeocodeCache | None = None,
    sleeps: list[float] | None = None,
) -> GeocodeClient:
    clock = {"t": 0.0}

    def fake_clock() -> float:
        return clock["t"]

    def fake_sleep(seconds: float) -> None:
        # Sleeping advances our fake monotonic clock so the rate limiter and the
        # backoff loop both see time pass without the test actually waiting.
        if sleeps is not None:
            sleeps.append(seconds)
        clock["t"] += seconds

    return GeocodeClient(
        session=session,
        cache=cache or InMemoryGeocodeCache(),
        user_agent="amenity-detector-test/1.0 (test@example.com)",
        min_interval_seconds=1.0,
        max_retries=3,
        backoff_base_seconds=1.0,
        sleep=fake_sleep,
        clock=fake_clock,
    )


def test_geocode_parses_a_single_hit() -> None:
    session = _FakeSession([_FakeResponse(status_code=200, payload=[_hit()])])
    client = _make_client(session)

    result = client.geocode("60311")

    assert isinstance(result, GeocodeResult)
    assert result.latitude == pytest.approx(50.1109)
    assert result.longitude == pytest.approx(8.6821)
    # bbox stored as (south, north, west, east) floats.
    assert result.bbox == pytest.approx((50.0969, 50.1249, 8.6621, 8.7021))
    assert "Frankfurt" in result.display_name


def test_user_agent_header_is_sent() -> None:
    session = _FakeSession([_FakeResponse(status_code=200, payload=[_hit()])])
    client = _make_client(session)

    client.geocode("60311")

    assert session.calls[0]["headers"]["User-Agent"].startswith("amenity-detector-test")


def test_cache_hit_skips_the_network() -> None:
    cache = InMemoryGeocodeCache()
    session = _FakeSession([_FakeResponse(status_code=200, payload=[_hit()])])
    client = _make_client(session, cache=cache)

    first = client.geocode("60311")
    second = client.geocode("  60311 ")  # different whitespace/case → same key

    assert first == second
    assert len(session.calls) == 1  # second call served from cache


def test_no_results_returns_none() -> None:
    session = _FakeSession([_FakeResponse(status_code=200, payload=[])])
    client = _make_client(session)

    assert client.geocode("nowhere-at-all-xyz") is None


def test_ambiguous_result_takes_the_top_hit() -> None:
    top = _hit(name="Berlin, Germany", lat="52.52", lon="13.405")
    other = _hit(name="Berlin, Maryland, USA", lat="38.32", lon="-75.21")
    session = _FakeSession([_FakeResponse(status_code=200, payload=[top, other])])
    client = _make_client(session)

    result = client.geocode("Berlin")

    assert result is not None
    assert result.latitude == pytest.approx(52.52)


def test_rate_limit_spaces_consecutive_calls() -> None:
    sleeps: list[float] = []
    session = _FakeSession(
        [
            _FakeResponse(status_code=200, payload=[_hit(name="A")]),
            _FakeResponse(status_code=200, payload=[_hit(name="B")]),
        ]
    )
    client = _make_client(session, sleeps=sleeps)

    client.geocode("query-a")
    client.geocode("query-b")  # distinct key → real second network call

    # The second call must wait out the remaining min_interval (~1s).
    assert any(s >= 1.0 for s in sleeps)


def test_retries_on_transient_error_then_succeeds() -> None:
    session = _FakeSession(
        [
            _FakeResponse(status_code=429, payload=None),
            _FakeResponse(status_code=503, payload=None),
            _FakeResponse(status_code=200, payload=[_hit()]),
        ]
    )
    client = _make_client(session)

    result = client.geocode("60311")

    assert result is not None
    assert len(session.calls) == 3


def test_retry_exhaustion_raises() -> None:
    session = _FakeSession(
        [
            _FakeResponse(status_code=503, payload=None),
            _FakeResponse(status_code=503, payload=None),
            _FakeResponse(status_code=503, payload=None),
            _FakeResponse(status_code=503, payload=None),
        ]
    )
    client = _make_client(session)

    with pytest.raises(GeocodeError):
        client.geocode("60311")


def test_blank_query_returns_none_without_network() -> None:
    session = _FakeSession([])
    client = _make_client(session)

    assert client.geocode("   ") is None
    assert session.calls == []


def test_structured_query_sends_postalcode_and_country() -> None:
    session = _FakeSession([_FakeResponse(status_code=200, payload=[_hit()])])
    client = _make_client(session)

    client.geocode("52062", country_code="DE")

    params = session.calls[0]["params"]
    assert params["postalcode"] == "52062"
    assert params["countrycodes"] == "de"  # lower-cased for Nominatim
    assert "q" not in params  # structured query, not free-text
    assert "street" not in params  # omitted when not supplied


def test_optional_street_is_included_when_given() -> None:
    session = _FakeSession([_FakeResponse(status_code=200, payload=[_hit()])])
    client = _make_client(session)

    client.geocode("52062", street="Bendelstrasse", country_code="DE")

    assert session.calls[0]["params"]["street"] == "Bendelstrasse"


def test_same_pin_different_country_does_not_collide_in_cache() -> None:
    cache = InMemoryGeocodeCache()
    session = _FakeSession(
        [
            _FakeResponse(status_code=200, payload=[_hit(name="10115 Berlin, Germany")]),
            _FakeResponse(status_code=200, payload=[_hit(name="10115 New York, USA")]),
        ]
    )
    client = _make_client(session, cache=cache)

    de = client.geocode("10115", country_code="DE")
    us = client.geocode("10115", country_code="US")

    assert de is not None and us is not None
    assert "Germany" in de.display_name
    assert "USA" in us.display_name
    assert len(session.calls) == 2  # distinct country → distinct key → second fetch
