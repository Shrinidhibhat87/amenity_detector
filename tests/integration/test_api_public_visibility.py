"""Integration tests for what the public surfaces are allowed to show.

Browse, the sitemap, llms.txt and the JSONL feed all read the same list
endpoint, and free-text search reads the hybrid pipeline. Neither may return
a listing that nobody published. The wizard's own by-id lookup is the one
deliberate exception: it is how a draft is previewed before publication.
"""

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from db.models import DetectedAmenity, Property, PropertyImage
from db.status import PropertyStatus


def _make_property(db: Session, name: str, status: str, slug: str) -> Property:
    prop = Property(name=name, status=status, slug=slug, description=f"{name} description")
    db.add(prop)
    db.flush()
    img = PropertyImage(property_id=prop.id, file_path=f"{prop.id}/a.jpg")
    db.add(img)
    db.flush()
    db.add(
        DetectedAmenity(
            property_id=prop.id,
            image_id=img.id,
            amenity_name="refrigerator",
            room_type="kitchen",
            is_present=True,
        )
    )
    db.commit()
    return prop


class TestListEndpointVisibility:
    def test_only_published_properties_are_listed(
        self, client: TestClient, db_session: Session
    ) -> None:
        published = _make_property(db_session, "Public flat", PropertyStatus.PUBLISHED, "public")
        _make_property(db_session, "Abandoned draft", PropertyStatus.DRAFT, "draft")
        _make_property(db_session, "Finished draft", PropertyStatus.COMPLETED, "completed")

        res = client.get("/api/v1/properties/")
        assert res.status_code == 200
        assert [p["id"] for p in res.json()] == [published.id]

    def test_amenity_search_skips_unpublished_properties(
        self, client: TestClient, db_session: Session
    ) -> None:
        published = _make_property(db_session, "Public flat", PropertyStatus.PUBLISHED, "public-2")
        _make_property(db_session, "Draft flat", PropertyStatus.DRAFT, "draft-2")

        res = client.get("/api/v1/properties/search", params={"amenities": "refrigerator"})
        assert res.status_code == 200
        assert [p["id"] for p in res.json()] == [published.id]


class TestSlugVisibility:
    def test_published_slug_resolves(self, client: TestClient, db_session: Session) -> None:
        _make_property(db_session, "Public flat", PropertyStatus.PUBLISHED, "public-3")

        res = client.get("/api/v1/properties/by-slug/public-3")
        assert res.status_code == 200

    def test_unpublished_slug_is_not_found(self, client: TestClient, db_session: Session) -> None:
        _make_property(db_session, "Draft flat", PropertyStatus.COMPLETED, "draft-3")

        res = client.get("/api/v1/properties/by-slug/draft-3")
        assert res.status_code == 404


class TestDraftPreviewStillWorks:
    def test_a_draft_is_reachable_by_id(self, client: TestClient, db_session: Session) -> None:
        draft = _make_property(db_session, "Draft flat", PropertyStatus.DRAFT, "draft-4")

        res = client.get(f"/api/v1/properties/{draft.id}")
        assert res.status_code == 200
        assert res.json()["status"] == "draft"


class TestFreeTextSearchVisibility:
    def test_search_returns_only_published_properties(
        self, client: TestClient, db_session: Session
    ) -> None:
        published = _make_property(db_session, "Bright flat", PropertyStatus.PUBLISHED, "public-5")
        _make_property(db_session, "Bright draft", PropertyStatus.DRAFT, "draft-5")

        res = client.post("/api/v1/search", json={"query": "bright flat"})
        assert res.status_code == 200
        returned = {p["id"] for p in res.json()}
        assert returned <= {published.id}
