"""
Integration tests for /api/v1/properties/* endpoints.

These tests exercise the complete request-to-database cycle:
  HTTP request → FastAPI router → service layer → ORM → SQLite → response

The VLM is replaced by a fake that returns canned JSON, so tests are fast
and deterministic. The database uses in-memory SQLite (set up in conftest.py).

Test coverage:
  - POST /api/v1/properties/                — property shell creation
  - POST /api/v1/properties/{id}/images     — per-image processing
  - GET  /api/v1/properties/                — listing and pagination
  - GET  /api/v1/properties/{id}            — detail view
  - GET  /api/v1/properties/search          — amenity search
  - DELETE /api/v1/properties/{id}          — deletion
  - GET  /health                            — liveness check
"""

from typing import cast

from fastapi.testclient import TestClient


def _create_property_with_images(
    client: TestClient,
    sample_image_bytes: bytes,
    *,
    name: str,
    image_filenames: list[str],
) -> str:
    """Create a property shell then upload one or more images through the live API."""
    shell_response = client.post(
        "/api/v1/properties/",
        json={"name": name, "model_name": "openai/gpt-4o-mini"},
    )
    assert shell_response.status_code == 201
    property_id = cast(str, shell_response.json()["property_id"])

    for filename in image_filenames:
        image_response = client.post(
            f"/api/v1/properties/{property_id}/images",
            data={"model_name": "openai/gpt-4o-mini"},
            files={"file": (filename, sample_image_bytes, "image/jpeg")},
        )
        assert image_response.status_code == 201

    return property_id


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient):
        """The /health endpoint should always return 200 when the app is running."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")
        assert "database" in data


class TestCreateAndUploadEndpoints:
    def test_create_property_shell_success(self, client: TestClient):
        """Creating a property shell should not require images."""
        response = client.post(
            "/api/v1/properties/",
            json={
                "name": "Shell House",
                "model_name": "openai/gpt-4o-mini",
                "extra_info": "Top floor",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["property_id"]
        assert data["property"]["name"] == "Shell House"
        assert data["property"]["images"] == []

    def test_upload_single_image_to_shell_success(
        self, client: TestClient, sample_image_bytes: bytes
    ):
        """A shell property can receive one image and return its detection rows."""
        property_id = _create_property_with_images(
            client,
            sample_image_bytes,
            name="Per Image House",
            image_filenames=["kitchen.jpg"],
        )

        response = client.get(f"/api/v1/properties/{property_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data["images"]) == 1
        assert data["images"][0]["file_path"].endswith("kitchen.jpg")
        assert len(data["images"][0]["amenities"]) > 0

    def test_removed_upload_endpoint_is_absent_from_openapi(self, client: TestClient):
        """The deprecated batch upload endpoint should no longer be published."""
        schema = client.get("/openapi.json").json()
        removed_path = "/api/v1/properties" + "/upload"
        assert removed_path not in schema["paths"]

    def test_upload_creates_amenity_records(self, client: TestClient, sample_image_bytes: bytes):
        """Detection results should be present on the stored image record."""
        property_id = _create_property_with_images(
            client,
            sample_image_bytes,
            name="Amenity Check",
            image_filenames=["room.jpg"],
        )

        response = client.get(f"/api/v1/properties/{property_id}")
        assert response.status_code == 200
        images = response.json()["images"]
        assert len(images) == 1

        amenities = images[0]["amenities"]
        present = {a["amenity_name"]: a["is_present"] for a in amenities}
        assert present.get("refrigerator") is True
        assert present.get("oven") is True
        assert present.get("dishwasher") is False

    def test_upload_unknown_model_returns_400(self, client: TestClient, sample_image_bytes: bytes):
        """Requesting an unregistered model should return 400."""
        shell_response = client.post(
            "/api/v1/properties/",
            json={"name": "Unknown Model", "model_name": "openai/gpt-4o-mini"},
        )
        property_id = shell_response.json()["property_id"]

        response = client.post(
            f"/api/v1/properties/{property_id}/images",
            data={"model_name": "nonexistent-model"},
            files={"file": ("img.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 400
        assert "not supported" in response.json()["detail"].lower()

    def test_upload_non_image_file_returns_400(self, client: TestClient):
        """Uploading a text file should be rejected at content-type validation."""
        shell_response = client.post(
            "/api/v1/properties/",
            json={"name": "Bad File", "model_name": "openai/gpt-4o-mini"},
        )
        property_id = shell_response.json()["property_id"]

        response = client.post(
            f"/api/v1/properties/{property_id}/images",
            data={"model_name": "fake-model"},
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 400
        assert "unsupported type" in response.json()["detail"].lower()

    def test_upload_multiple_images(self, client: TestClient, sample_image_bytes: bytes):
        """Multiple images can be uploaded one-by-one to the same property shell."""
        property_id = _create_property_with_images(
            client,
            sample_image_bytes,
            name="Multi Image",
            image_filenames=["kitchen.jpg", "bedroom.jpg"],
        )

        response = client.get(f"/api/v1/properties/{property_id}")
        assert response.status_code == 200
        assert len(response.json()["images"]) == 2


class TestListEndpoint:
    def test_list_returns_empty_initially(self, client: TestClient):
        """Fresh database should return an empty list."""
        response = client.get("/api/v1/properties/")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_returns_properties_after_upload(
        self, client: TestClient, sample_image_bytes: bytes
    ):
        """After uploading, the property should appear in the list."""
        _create_property_with_images(
            client,
            sample_image_bytes,
            name="Listed Property",
            image_filenames=["img.jpg"],
        )

        response = client.get("/api/v1/properties/")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        names = [p["name"] for p in data]
        assert "Listed Property" in names

    def test_list_pagination_limit(self, client: TestClient, sample_image_bytes: bytes):
        """The limit parameter should cap the number of results returned."""
        for i in range(3):
            _create_property_with_images(
                client,
                sample_image_bytes,
                name=f"Paginated {i}",
                image_filenames=["img.jpg"],
            )

        response = client.get("/api/v1/properties/?limit=2")
        assert response.status_code == 200
        assert len(response.json()) <= 2


class TestDetailEndpoint:
    def test_get_existing_property_returns_200(self, client: TestClient, sample_image_bytes: bytes):
        """Getting a property by its ID should return full details."""
        property_id = _create_property_with_images(
            client,
            sample_image_bytes,
            name="Detail Test",
            image_filenames=["img.jpg"],
        )

        response = client.get(f"/api/v1/properties/{property_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == property_id
        assert data["name"] == "Detail Test"
        assert len(data["images"]) == 1

    def test_get_nonexistent_property_returns_404(self, client: TestClient):
        """A request for a non-existent property ID should return 404."""
        response = client.get("/api/v1/properties/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404


class TestSearchEndpoint:
    def test_search_finds_matching_property(self, client: TestClient, sample_image_bytes: bytes):
        """Searching for an amenity that was detected should return the property."""
        _create_property_with_images(
            client,
            sample_image_bytes,
            name="Searchable Kitchen",
            image_filenames=["kitchen.jpg"],
        )

        response = client.get("/api/v1/properties/search?amenities=refrigerator")
        assert response.status_code == 200
        names = [p["name"] for p in response.json()]
        assert "Searchable Kitchen" in names

    def test_search_excludes_non_matching_property(
        self, client: TestClient, sample_image_bytes: bytes
    ):
        """Searching for an amenity the fake VLM never returns should return no match."""
        _create_property_with_images(
            client,
            sample_image_bytes,
            name="No Pool Here",
            image_filenames=["img.jpg"],
        )

        response = client.get("/api/v1/properties/search?amenities=pool")
        assert response.status_code == 200
        names = [p["name"] for p in response.json()]
        assert "No Pool Here" not in names

    def test_search_empty_amenities_returns_400(self, client: TestClient):
        """An empty amenities string should be rejected."""
        response = client.get("/api/v1/properties/search?amenities=")
        assert response.status_code == 400


class TestDeleteEndpoint:
    def test_delete_existing_property_returns_204(
        self, client: TestClient, sample_image_bytes: bytes
    ):
        """Successfully deleting a property should return 204 No Content."""
        property_id = _create_property_with_images(
            client,
            sample_image_bytes,
            name="To Be Deleted",
            image_filenames=["img.jpg"],
        )

        response = client.delete(f"/api/v1/properties/{property_id}")
        assert response.status_code == 204

    def test_deleted_property_no_longer_retrievable(
        self, client: TestClient, sample_image_bytes: bytes
    ):
        """After deletion, GET should return 404."""
        property_id = _create_property_with_images(
            client,
            sample_image_bytes,
            name="Delete Then Get",
            image_filenames=["img.jpg"],
        )

        client.delete(f"/api/v1/properties/{property_id}")

        response = client.get(f"/api/v1/properties/{property_id}")
        assert response.status_code == 404

    def test_delete_nonexistent_property_returns_404(self, client: TestClient):
        """Trying to delete a property that doesn't exist should return 404."""
        response = client.delete("/api/v1/properties/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404


class TestModelsEndpoint:
    def test_models_returns_list(self, client: TestClient):
        """GET /api/v1/models/ should return a list of model info objects."""
        response = client.get("/api/v1/models/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert [model["name"] for model in data] == [
            "openai/gpt-4o-mini",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3.2-11b-vision-instruct",
            "qwen/qwen2-vl-72b-instruct",
        ]

    def test_models_include_required_fields(self, client: TestClient):
        """Each model entry must have name, available, and description."""
        response = client.get("/api/v1/models/")
        for model in response.json():
            assert "name" in model
            assert "available" in model
            assert "description" in model

    def test_models_descriptions_match_openrouter_catalog(self, client: TestClient):
        """Model descriptions should advertise the OpenRouter-backed catalog."""
        response = client.get("/api/v1/models/")
        descriptions = {model["name"]: model["description"] for model in response.json()}

        assert descriptions == {
            "openai/gpt-4o-mini": (
                "GPT-4o Mini via OpenRouter — cheapest reliable JSON-mode option."
            ),
            "google/gemini-pro-1.5": (
                "Gemini Pro 1.5 via OpenRouter — strong vision reasoning, JSON-safe."
            ),
            "meta-llama/llama-3.2-11b-vision-instruct": (
                "Llama 3.2 11B Vision via OpenRouter — open-weights baseline."
            ),
            "qwen/qwen2-vl-72b-instruct": (
                "Qwen2-VL 72B via OpenRouter — highest-capacity open-weights option."
            ),
        }
