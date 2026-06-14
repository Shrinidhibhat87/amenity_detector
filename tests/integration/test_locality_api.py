"""Integration tests for the locality endpoints.

The locality agent is replaced with a fake that returns a canned
``LocalityResult`` — these tests exercise the transport + persistence layers
(request validation, response shape, the 404 path, and upsert-on-rerun), not
the OSM network or the LLM loop (those are covered by the unit tests).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api.dependencies import get_locality_agent
from core.locality.agent import LocalityResult
from core.locality.overpass import Poi
from db.models import LocalityInsight, Property


class _FakeAgent:
    """Records calls and returns a fixed result built from the requested location."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def run(self, location_query: str, *, radius_hint: int | None = None) -> LocalityResult:
        self.calls.append((location_query, radius_hint))
        return LocalityResult(
            location_query=location_query,
            display_name="60311 Frankfurt am Main, Germany",
            latitude=50.1109,
            longitude=8.6821,
            radius_m=radius_hint or 1000,
            pois=[
                Poi(
                    category="school",
                    name="Goethe-Schule",
                    latitude=50.111,
                    longitude=8.682,
                    osm_type="node",
                    osm_id=1,
                    distance_m=120.0,
                    tags={},
                )
            ],
            category_counts={"school": 1},
            blurb="Central spot with a school nearby.",
        )


@pytest.fixture(scope="function")
def fake_agent(test_app) -> _FakeAgent:
    agent = _FakeAgent()
    test_app.dependency_overrides[get_locality_agent] = lambda: agent
    return agent


def test_preview_returns_enriched_result(client: TestClient, fake_agent: _FakeAgent) -> None:
    resp = client.post("/api/v1/locality", json={"location": "60311"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["display_name"].startswith("60311")
    assert body["category_counts"] == {"school": 1}
    assert body["pois"][0]["name"] == "Goethe-Schule"
    assert "OpenStreetMap" in body["attribution"]
    assert fake_agent.calls == [("60311", None)]


def test_preview_forwards_radius_hint(client: TestClient, fake_agent: _FakeAgent) -> None:
    resp = client.post("/api/v1/locality", json={"location": "Bockenheim", "radius_m": 2000})

    assert resp.status_code == 200
    assert resp.json()["radius_m"] == 2000
    assert fake_agent.calls == [("Bockenheim", 2000)]


def test_preview_rejects_blank_location(client: TestClient, fake_agent: _FakeAgent) -> None:
    resp = client.post("/api/v1/locality", json={"location": ""})
    assert resp.status_code == 422


def test_preview_does_not_persist(
    client: TestClient, fake_agent: _FakeAgent, db_session: Session
) -> None:
    client.post("/api/v1/locality", json={"location": "60311"})
    assert db_session.query(LocalityInsight).count() == 0


def test_persist_writes_insight_to_property(
    client: TestClient, fake_agent: _FakeAgent, db_session: Session
) -> None:
    prop = Property(name="Frankfurt flat")
    db_session.add(prop)
    db_session.commit()

    resp = client.post(f"/api/v1/properties/{prop.id}/locality", json={"location": "60311"})

    assert resp.status_code == 200
    insight = db_session.query(LocalityInsight).filter_by(property_id=prop.id).one()
    assert insight.blurb == "Central spot with a school nearby."
    assert insight.category_counts == {"school": 1}
    assert insight.pois[0]["name"] == "Goethe-Schule"


def test_persist_rerun_upserts_single_row(
    client: TestClient, fake_agent: _FakeAgent, db_session: Session
) -> None:
    prop = Property(name="Frankfurt flat")
    db_session.add(prop)
    db_session.commit()

    client.post(f"/api/v1/properties/{prop.id}/locality", json={"location": "60311"})
    client.post(f"/api/v1/properties/{prop.id}/locality", json={"location": "60322"})

    rows = db_session.query(LocalityInsight).filter_by(property_id=prop.id).all()
    assert len(rows) == 1
    assert rows[0].location_query == "60322"  # latest run wins


def test_persist_unknown_property_returns_404(client: TestClient, fake_agent: _FakeAgent) -> None:
    resp = client.post("/api/v1/properties/does-not-exist/locality", json={"location": "60311"})
    assert resp.status_code == 404
