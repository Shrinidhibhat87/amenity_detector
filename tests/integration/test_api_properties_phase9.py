"""Integration tests for the Phase 9 listing-metadata API.

These tests exercise the new optional fields on POST /api/v1/properties/, the
PATCH endpoint, and the validation rules enforced by the Pydantic schemas.

Existing rows that predate Phase 9 must still be retrievable, so the suite
includes a "legacy NULL property" path that simulates a pre-migration row.
"""

from fastapi.testclient import TestClient


class TestCreatePropertyWithListingMetadata:
    def test_minimal_create_still_works(self, client: TestClient) -> None:
        # The Phase 8 contract — name + model_name only — must remain valid.
        response = client.post(
            "/api/v1/properties/",
            json={"name": "Minimal", "model_name": "openai/gpt-4o-mini"},
        )
        assert response.status_code == 201
        body = response.json()["property"]
        assert body["name"] == "Minimal"
        assert body["price"] is None
        assert body["num_bedrooms"] is None
        assert body["slug"] is not None  # auto-generated even without metadata
        assert body["slug"].startswith("minimal-")

    def test_full_listing_metadata_round_trip(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/properties/",
            json={
                "name": "Frankfurt 3BHK",
                "model_name": "openai/gpt-4o-mini",
                "extra_info": "Two-bed flat near park",
                "listing_type": "rent",
                "price": 1500.0,
                "currency": "EUR",
                "price_period": "monthly",
                "num_bedrooms": 3,
                "num_bathrooms": 2,
                "area_sqm": 82.5,
                "property_type": "apartment",
                "furnishing": "semi_furnished",
                "available_from": "2026-06-01",
                "locality": "Sachsenhausen",
                "postal_code": "60594",
                "country_code": "DE",
                "latitude": 50.105200,
                "longitude": 8.681300,
                "owner_email": "owner@example.com",
            },
        )
        assert response.status_code == 201
        body = response.json()["property"]
        assert body["listing_type"] == "rent"
        assert float(body["price"]) == 1500.0
        assert body["currency"] == "EUR"
        assert body["price_period"] == "monthly"
        assert body["num_bedrooms"] == 3
        assert body["num_bathrooms"] == 2
        assert float(body["area_sqm"]) == 82.5
        assert body["property_type"] == "apartment"
        assert body["furnishing"] == "semi_furnished"
        assert body["available_from"] == "2026-06-01"
        assert body["locality"] == "Sachsenhausen"
        assert body["postal_code"] == "60594"
        assert body["country_code"] == "DE"
        assert body["owner_email"] == "owner@example.com"
        assert body["slug"].startswith("frankfurt-3bhk-")

    def test_invalid_listing_type_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/properties/",
            json={
                "name": "Bad",
                "model_name": "openai/gpt-4o-mini",
                "listing_type": "lease-to-buy",
            },
        )
        assert response.status_code == 422

    def test_country_code_must_be_two_chars(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/properties/",
            json={
                "name": "Bad",
                "model_name": "openai/gpt-4o-mini",
                "country_code": "DEU",
            },
        )
        assert response.status_code == 422

    def test_negative_price_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/properties/",
            json={
                "name": "Bad",
                "model_name": "openai/gpt-4o-mini",
                "price": -10,
            },
        )
        assert response.status_code == 422

    def test_two_same_names_get_distinct_slugs(self, client: TestClient) -> None:
        first = client.post(
            "/api/v1/properties/",
            json={"name": "Same Name", "model_name": "openai/gpt-4o-mini"},
        ).json()["property"]
        second = client.post(
            "/api/v1/properties/",
            json={"name": "Same Name", "model_name": "openai/gpt-4o-mini"},
        ).json()["property"]
        assert first["slug"] != second["slug"]


class TestPatchProperty:
    def _create(self, client: TestClient, name: str = "Patch Target") -> str:
        response = client.post(
            "/api/v1/properties/",
            json={"name": name, "model_name": "openai/gpt-4o-mini"},
        )
        property_id: str = response.json()["property_id"]
        return property_id

    def test_patch_updates_only_provided_fields(self, client: TestClient) -> None:
        property_id = self._create(client)

        patch_response = client.patch(
            f"/api/v1/properties/{property_id}",
            json={"num_bedrooms": 3, "currency": "EUR", "price": 1200.0},
        )
        assert patch_response.status_code == 200
        body = patch_response.json()
        assert body["num_bedrooms"] == 3
        assert body["currency"] == "EUR"
        assert float(body["price"]) == 1200.0
        # Untouched fields remain at their defaults.
        assert body["num_bathrooms"] is None
        assert body["locality"] is None

    def test_patch_can_clear_field_with_null(self, client: TestClient) -> None:
        property_id = self._create(client)
        client.patch(f"/api/v1/properties/{property_id}", json={"num_bedrooms": 4})

        cleared = client.patch(
            f"/api/v1/properties/{property_id}",
            json={"num_bedrooms": None},
        )
        assert cleared.status_code == 200
        assert cleared.json()["num_bedrooms"] is None

    def test_patch_unknown_property_returns_404(self, client: TestClient) -> None:
        response = client.patch(
            "/api/v1/properties/does-not-exist",
            json={"num_bedrooms": 1},
        )
        assert response.status_code == 404

    def test_patch_cannot_change_slug(self, client: TestClient) -> None:
        property_id = self._create(client)
        original_slug: str = client.get(f"/api/v1/properties/{property_id}").json()["slug"]

        # ``slug`` is intentionally absent from PropertyUpdateRequest, so sending
        # it is rejected by Pydantic's "extra=forbid" config.
        response = client.patch(
            f"/api/v1/properties/{property_id}",
            json={"slug": "hacked-slug"},
        )
        assert response.status_code == 422

        unchanged = client.get(f"/api/v1/properties/{property_id}").json()
        assert unchanged["slug"] == original_slug


class TestLegacyNullRowsStillLoad:
    def test_get_property_with_null_phase9_fields_returns_nulls(
        self, client: TestClient, db_session
    ) -> None:
        """A row inserted directly without Phase 9 fields must load through GET."""
        from db.models import Property

        legacy = Property(name="Legacy")
        db_session.add(legacy)
        db_session.commit()

        response = client.get(f"/api/v1/properties/{legacy.id}")
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "Legacy"
        # Every Phase 9 field surfaces as None on a legacy row.
        for field in (
            "slug",
            "listing_type",
            "price",
            "currency",
            "price_period",
            "num_bedrooms",
            "num_bathrooms",
            "area_sqm",
            "property_type",
            "furnishing",
            "available_from",
            "locality",
            "postal_code",
            "country_code",
            "latitude",
            "longitude",
            "owner_email",
        ):
            assert body[field] is None, f"{field} should be None on legacy row"


class TestGetPropertyBySlug:
    """Slug-based lookup for the public property URLs."""

    def _create(self, client: TestClient, db_session, name: str) -> dict[str, object]:
        response = client.post(
            "/api/v1/properties/",
            json={"name": name, "model_name": "openai/gpt-4o-mini"},
        )
        assert response.status_code == 201
        body: dict[str, object] = response.json()["property"]
        # By-slug is a public lookup, so it only resolves published listings.
        # Set the status directly: this test is about slug resolution, not
        # about walking the whole wizard to satisfy the publish preconditions.
        from db.models import Property

        prop = db_session.get(Property, body["id"])
        prop.status = "published"
        db_session.commit()
        return body

    def test_returns_property_when_slug_matches(self, client: TestClient, db_session) -> None:
        created = self._create(client, db_session, "Sachsenhausen 3BHK")
        slug = created["slug"]
        assert isinstance(slug, str) and len(slug) > 0

        response = client.get(f"/api/v1/properties/by-slug/{slug}")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == created["id"]
        assert body["slug"] == slug
        assert body["name"] == "Sachsenhausen 3BHK"
        # Detail response shape — must include images list, not just summary.
        assert isinstance(body["images"], list)

    def test_returns_404_for_unknown_slug(self, client: TestClient) -> None:
        response = client.get("/api/v1/properties/by-slug/does-not-exist-xyz123")
        assert response.status_code == 404

    def test_distinguishes_two_listings_with_similar_names(
        self, client: TestClient, db_session
    ) -> None:
        # The auto-generated slug suffix from the UUID prefix is what keeps
        # two listings with the same name addressable; verify each by-slug
        # call still resolves to the right row.
        first = self._create(client, db_session, "Same Name")
        second = self._create(client, db_session, "Same Name")
        assert first["slug"] != second["slug"]

        first_response = client.get(f"/api/v1/properties/by-slug/{first['slug']}")
        second_response = client.get(f"/api/v1/properties/by-slug/{second['slug']}")
        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert first_response.json()["id"] == first["id"]
        assert second_response.json()["id"] == second["id"]
