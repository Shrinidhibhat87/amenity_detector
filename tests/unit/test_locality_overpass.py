"""Unit tests for the Overpass POI client.

Like the geocode tests, these run fully offline against a fake HTTP session.
We cover what the plan calls out: parsing mocked Overpass JSON (nodes + way
centers), category → tag-selector mapping, empty results, caching, and the
unknown-category guard.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.locality.overpass import (
    CATEGORIES,
    InMemoryPoiCache,
    OverpassClient,
    Poi,
)


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        *,
        data: dict[str, str],
        headers: dict[str, str],
        timeout: float,
    ) -> _FakeResponse:
        self.calls.append({"url": url, "data": data, "headers": headers})
        if not self._responses:
            raise AssertionError("FakeSession ran out of scripted responses")
        return self._responses.pop(0)


def _make_client(
    session: _FakeSession,
    *,
    cache: InMemoryPoiCache | None = None,
) -> OverpassClient:
    clock = {"t": 0.0}

    def fake_sleep(seconds: float) -> None:
        clock["t"] += seconds

    return OverpassClient(
        session=session,
        cache=cache or InMemoryPoiCache(),
        user_agent="amenity-detector-test/1.0 (test@example.com)",
        min_interval_seconds=1.0,
        max_retries=2,
        backoff_base_seconds=1.0,
        sleep=fake_sleep,
        clock=lambda: clock["t"],
    )


def _elements() -> list[dict[str, Any]]:
    return [
        # A node with explicit lat/lon, ~near the search centre.
        {
            "type": "node",
            "id": 1,
            "lat": 50.1110,
            "lon": 8.6822,
            "tags": {"amenity": "school", "name": "Goethe-Schule"},
        },
        # A way returns its centroid via `out center`.
        {
            "type": "way",
            "id": 2,
            "center": {"lat": 50.1200, "lon": 8.6900},
            "tags": {"amenity": "school", "name": "Far Gymnasium"},
        },
        # Malformed: no coordinate anywhere → dropped.
        {"type": "way", "id": 3, "tags": {"amenity": "school"}},
    ]


def test_parses_nodes_and_way_centers() -> None:
    session = _FakeSession([_FakeResponse(status_code=200, payload={"elements": _elements()})])
    client = _make_client(session)

    pois = client.query(lat=50.1109, lon=8.6821, radius_m=1500, category="school")

    assert all(isinstance(p, Poi) for p in pois)
    assert [p.name for p in pois] == ["Goethe-Schule", "Far Gymnasium"]  # malformed dropped
    assert pois[0].osm_type == "node"
    assert pois[1].osm_type == "way"


def test_results_are_distance_sorted_nearest_first() -> None:
    session = _FakeSession([_FakeResponse(status_code=200, payload={"elements": _elements()})])
    client = _make_client(session)

    pois = client.query(lat=50.1109, lon=8.6821, radius_m=1500, category="school")

    assert pois[0].distance_m < pois[1].distance_m
    assert pois[0].name == "Goethe-Schule"  # the closer one


def test_category_maps_to_tag_selector_in_query() -> None:
    session = _FakeSession([_FakeResponse(status_code=200, payload={"elements": []})])
    client = _make_client(session)

    client.query(lat=50.0, lon=8.0, radius_m=1000, category="pharmacy")

    sent_query = session.calls[0]["data"]["data"]
    assert '"amenity"="pharmacy"' in sent_query
    assert "around:1000" in sent_query


def test_empty_elements_returns_empty_list() -> None:
    session = _FakeSession([_FakeResponse(status_code=200, payload={"elements": []})])
    client = _make_client(session)

    assert client.query(lat=50.0, lon=8.0, radius_m=1000, category="park") == []


def test_cache_hit_skips_network() -> None:
    cache = InMemoryPoiCache()
    session = _FakeSession([_FakeResponse(status_code=200, payload={"elements": _elements()})])
    client = _make_client(session, cache=cache)

    first = client.query(lat=50.1109, lon=8.6821, radius_m=1500, category="school")
    second = client.query(lat=50.1109, lon=8.6821, radius_m=1500, category="school")

    assert first == second
    assert len(session.calls) == 1


def test_unknown_category_raises() -> None:
    session = _FakeSession([])
    client = _make_client(session)

    with pytest.raises(ValueError):
        client.query(lat=50.0, lon=8.0, radius_m=1000, category="nightclub")


def test_all_known_categories_build_a_query() -> None:
    # Every advertised category must map to at least one selector and produce QL.
    for category in CATEGORIES:
        session = _FakeSession([_FakeResponse(status_code=200, payload={"elements": []})])
        client = _make_client(session)
        client.query(lat=50.0, lon=8.0, radius_m=800, category=category)
        assert "around:800" in session.calls[0]["data"]["data"]


def test_result_count_is_capped() -> None:
    many = [
        {"type": "node", "id": i, "lat": 50.0 + i * 0.001, "lon": 8.0, "tags": {"leisure": "park"}}
        for i in range(100)
    ]
    session = _FakeSession([_FakeResponse(status_code=200, payload={"elements": many})])
    client = _make_client(session)

    pois = client.query(lat=50.0, lon=8.0, radius_m=5000, category="park", limit=10)

    assert len(pois) == 10
