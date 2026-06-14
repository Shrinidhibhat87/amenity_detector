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
    """Records calls and returns a fixed result built from the requested PIN."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(
        self,
        *,
        postal_code: str,
        street: str | None = None,
        country_code: str = "DE",
        radius_m: int = 3000,
    ) -> LocalityResult:
        self.calls.append(
            {
                "postal_code": postal_code,
                "street": street,
                "country_code": country_code,
                "radius_m": radius_m,
            }
        )
        return LocalityResult(
            location_query=f"{postal_code}, {country_code}",
            display_name="60311 Frankfurt am Main, Germany",
            latitude=50.1109,
            longitude=8.6821,
            radius_m=radius_m,
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
            category_counts={"school": 1, "transit": 14},
            transit_breakdown={"bus": 12, "rail": 2},
            blurb="Central spot with a school nearby.",
        )


@pytest.fixture(scope="function")
def fake_agent(test_app) -> _FakeAgent:
    agent = _FakeAgent()
    test_app.dependency_overrides[get_locality_agent] = lambda: agent
    return agent


def test_preview_returns_enriched_result(client: TestClient, fake_agent: _FakeAgent) -> None:
    resp = client.post("/api/v1/locality", json={"postal_code": "60311"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["display_name"].startswith("60311")
    assert body["category_counts"] == {"school": 1, "transit": 14}
    assert body["transit_breakdown"] == {"bus": 12, "rail": 2}
    assert body["pois"][0]["name"] == "Goethe-Schule"
    assert "OpenStreetMap" in body["attribution"]
    # Defaults applied: country DE, radius 3 km, no street.
    assert fake_agent.calls == [
        {"postal_code": "60311", "street": None, "country_code": "DE", "radius_m": 3000}
    ]


def test_preview_forwards_street_country_and_radius(
    client: TestClient, fake_agent: _FakeAgent
) -> None:
    resp = client.post(
        "/api/v1/locality",
        json={
            "postal_code": "52062",
            "street": "Bendelstrasse",
            "country_code": "de",
            "radius_m": 5000,
        },
    )

    assert resp.status_code == 200
    assert resp.json()["radius_m"] == 5000
    assert fake_agent.calls == [
        {
            "postal_code": "52062",
            "street": "Bendelstrasse",
            "country_code": "DE",  # upper-cased by the schema
            "radius_m": 5000,
        }
    ]


def test_preview_rejects_blank_postal_code(client: TestClient, fake_agent: _FakeAgent) -> None:
    resp = client.post("/api/v1/locality", json={"postal_code": ""})
    assert resp.status_code == 422


def test_preview_rejects_out_of_range_radius(client: TestClient, fake_agent: _FakeAgent) -> None:
    resp = client.post("/api/v1/locality", json={"postal_code": "60311", "radius_m": 50000})
    assert resp.status_code == 422


def test_upstream_failure_returns_503_not_opaque_error(client: TestClient, test_app) -> None:
    from core.locality.geocode import GeocodeError

    class _FailingAgent:
        def run(self, **_: object) -> LocalityResult:
            raise GeocodeError("Nominatim returned non-retryable status 403")

    test_app.dependency_overrides[get_locality_agent] = lambda: _FailingAgent()
    resp = client.post("/api/v1/locality", json={"postal_code": "52062"})

    assert resp.status_code == 503
    assert "temporarily unavailable" in resp.json()["detail"]


def test_persist_stores_transit_breakdown(
    client: TestClient, fake_agent: _FakeAgent, db_session: Session
) -> None:
    prop = Property(name="Aachen flat")
    db_session.add(prop)
    db_session.commit()

    client.post(f"/api/v1/properties/{prop.id}/locality", json={"postal_code": "52062"})

    insight = db_session.query(LocalityInsight).filter_by(property_id=prop.id).one()
    assert insight.transit_breakdown == {"bus": 12, "rail": 2}


def test_preview_does_not_persist(
    client: TestClient, fake_agent: _FakeAgent, db_session: Session
) -> None:
    client.post("/api/v1/locality", json={"postal_code": "60311"})
    assert db_session.query(LocalityInsight).count() == 0


def test_persist_writes_insight_to_property(
    client: TestClient, fake_agent: _FakeAgent, db_session: Session
) -> None:
    prop = Property(name="Frankfurt flat")
    db_session.add(prop)
    db_session.commit()

    resp = client.post(f"/api/v1/properties/{prop.id}/locality", json={"postal_code": "60311"})

    assert resp.status_code == 200
    insight = db_session.query(LocalityInsight).filter_by(property_id=prop.id).one()
    assert insight.blurb == "Central spot with a school nearby."
    assert insight.category_counts == {"school": 1, "transit": 14}
    assert insight.pois[0]["name"] == "Goethe-Schule"


def test_persist_rerun_upserts_single_row(
    client: TestClient, fake_agent: _FakeAgent, db_session: Session
) -> None:
    prop = Property(name="Frankfurt flat")
    db_session.add(prop)
    db_session.commit()

    client.post(f"/api/v1/properties/{prop.id}/locality", json={"postal_code": "60311"})
    client.post(f"/api/v1/properties/{prop.id}/locality", json={"postal_code": "60322"})

    rows = db_session.query(LocalityInsight).filter_by(property_id=prop.id).all()
    assert len(rows) == 1
    assert rows[0].location_query == "60322, DE"  # latest run wins


def test_persist_unknown_property_returns_404(client: TestClient, fake_agent: _FakeAgent) -> None:
    resp = client.post("/api/v1/properties/does-not-exist/locality", json={"postal_code": "60311"})
    assert resp.status_code == 404


def test_detail_exposes_persisted_insight(
    client: TestClient, fake_agent: _FakeAgent, db_session: Session
) -> None:
    prop = Property(name="Frankfurt flat")
    db_session.add(prop)
    db_session.commit()
    client.post(f"/api/v1/properties/{prop.id}/locality", json={"postal_code": "60311"})

    detail = client.get(f"/api/v1/properties/{prop.id}").json()

    assert detail["locality_insight"] is not None
    assert detail["locality_insight"]["blurb"] == "Central spot with a school nearby."
    assert "OpenStreetMap" in detail["locality_insight"]["attribution"]


def test_detail_without_insight_is_null(client: TestClient, db_session: Session) -> None:
    prop = Property(name="No locality yet")
    db_session.add(prop)
    db_session.commit()

    detail = client.get(f"/api/v1/properties/{prop.id}").json()

    assert detail["locality_insight"] is None
