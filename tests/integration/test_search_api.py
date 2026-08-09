"""Integration tests for ``POST /api/v1/search``.

The conftest already overrides the SearchPipeline dependency with a
parser-less / embedder-less pipeline, so these tests exercise:

  - the FastAPI wiring (request body validation, response serialisation),
  - the regex fallback parser (parser=None route),
  - the SQL builder against the real ORM and a SQLite database,
  - the scorer (zero cosine + zero FTS leaves boost-only ordering).

Real LLM and pgvector behaviour is covered by the unit tests; here we only
verify the end-to-end seam.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from db.models import DetectedAmenity, Property, PropertyImage


@pytest.fixture()
def seeded_db(db_session: Session) -> Session:
    """Seed three contrasting properties so the filter assertions have signal."""
    props = [
        Property(
            id="p-rent-3bhk-eur",
            name="Frankfurt 3BHK",
            description="Spacious 3-bedroom apartment with fireplace.",
            slug="frankfurt-3bhk",
            listing_type="rent",
            price=1400,
            currency="EUR",
            num_bedrooms=3,
            locality="Frankfurt",
            country_code="DE",
            created_at=datetime.now(UTC),
            status="published",
        ),
        Property(
            id="p-rent-2bhk-eur",
            name="Berlin 2BHK",
            description="Cozy two-bedroom flat near a park.",
            slug="berlin-2bhk",
            listing_type="rent",
            price=900,
            currency="EUR",
            num_bedrooms=2,
            locality="Berlin",
            country_code="DE",
            created_at=datetime.now(UTC),
            status="published",
        ),
        Property(
            id="p-sale-villa",
            name="Bali Villa",
            description="Villa for sale with a pool.",
            slug="bali-villa",
            listing_type="sale",
            price=300000,
            currency="USD",
            num_bedrooms=4,
            locality="Bali",
            country_code="ID",
            created_at=datetime.now(UTC),
            status="published",
        ),
    ]
    for p in props:
        db_session.add(p)
    db_session.flush()

    # One image per property so the response shape includes first_image_id.
    for p in props:
        db_session.add(
            PropertyImage(
                id=f"img-{p.id}",
                property_id=p.id,
                file_path=f"{p.id}/hero.jpg",
                room_type="living_room",
                is_primary=True,
                display_order=0,
            )
        )
    db_session.flush()

    # Detected amenities: fireplace in living_room on the 3BHK, pool on the villa.
    db_session.add(
        DetectedAmenity(
            id="da-frankfurt-fireplace",
            property_id="p-rent-3bhk-eur",
            image_id="img-p-rent-3bhk-eur",
            amenity_name="fireplace",
            room_type="living_room",
            is_present=True,
            confidence=1.0,
        )
    )
    db_session.add(
        DetectedAmenity(
            id="da-bali-pool",
            property_id="p-sale-villa",
            image_id="img-p-sale-villa",
            amenity_name="pool",
            room_type=None,
            is_present=True,
            confidence=1.0,
        )
    )
    db_session.flush()
    return db_session


class TestSearchEndpoint:
    def test_rejects_empty_query(self, client: TestClient, seeded_db: Session) -> None:
        resp = client.post("/api/v1/search", json={"query": ""})
        assert resp.status_code == 422

    def test_listing_type_and_bedrooms_filter(self, client: TestClient, seeded_db: Session) -> None:
        resp = client.post("/api/v1/search", json={"query": "3 bhk rent"})
        assert resp.status_code == 200
        ids = [p["id"] for p in resp.json()]
        # Only the 3BHK rent listing matches.
        assert ids == ["p-rent-3bhk-eur"]

    def test_price_ceiling_filter(self, client: TestClient, seeded_db: Session) -> None:
        resp = client.post("/api/v1/search", json={"query": "rent under 1000 EUR"})
        assert resp.status_code == 200
        ids = {p["id"] for p in resp.json()}
        # 1400 EUR row excluded; 2BHK Berlin is the only EUR rent under 1000.
        assert ids == {"p-rent-2bhk-eur"}

    def test_required_room_amenity_tuple(self, client: TestClient, seeded_db: Session) -> None:
        resp = client.post(
            "/api/v1/search",
            json={"query": "apartment with fireplace in living room"},
        )
        assert resp.status_code == 200
        ids = [p["id"] for p in resp.json()]
        assert ids == ["p-rent-3bhk-eur"]

    def test_bare_amenity_anywhere(self, client: TestClient, seeded_db: Session) -> None:
        resp = client.post("/api/v1/search", json={"query": "house with pool"})
        assert resp.status_code == 200
        ids = [p["id"] for p in resp.json()]
        assert ids == ["p-sale-villa"]

    def test_no_match_returns_empty_list(self, client: TestClient, seeded_db: Session) -> None:
        resp = client.post("/api/v1/search", json={"query": "10 bhk rent"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_response_shape_matches_summary(self, client: TestClient, seeded_db: Session) -> None:
        resp = client.post("/api/v1/search", json={"query": "rent"})
        assert resp.status_code == 200
        for item in resp.json():
            # Shape of PropertySummaryResponse — must include image_count, slug.
            assert "id" in item and "name" in item and "slug" in item
            assert "image_count" in item
            # The raw embedding vector must never leak to the wire.
            assert "description_embedding" not in item

    def test_limit_is_honoured(self, client: TestClient, seeded_db: Session) -> None:
        resp = client.post("/api/v1/search", json={"query": "rent", "limit": 1})
        assert resp.status_code == 200
        assert len(resp.json()) == 1
