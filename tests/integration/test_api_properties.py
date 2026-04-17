"""
Integration tests for /api/v1/properties/* endpoints.

These tests exercise the complete request-to-database cycle:
  HTTP request → FastAPI router → service layer → ORM → SQLite → response

The VLM is replaced by a fake that returns canned JSON, so tests are fast
and deterministic. The database uses in-memory SQLite (set up in conftest.py).

Test coverage:
  - POST /api/v1/properties/upload   — happy path and error cases
  - GET  /api/v1/properties/         — listing and pagination
  - GET  /api/v1/properties/{id}     — detail view
  - GET  /api/v1/properties/search   — amenity search
  - DELETE /api/v1/properties/{id}   — deletion
  - GET  /health                     — liveness check
"""

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient):
        """The /health endpoint should always return 200 when the app is running."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")  # degraded if DB unreachable
        assert "database" in data


class TestUploadEndpoint:
    def test_upload_single_image_success(self, client: TestClient, sample_image_bytes: bytes):
        """Uploading one valid image should create a property and return 201."""
        response = client.post(
            "/api/v1/properties/upload",
            data={
                "name": "Test House",
                "model_name": "fake-model",
                "extra_info": "Near the park",
            },
            files={"files": ("kitchen.jpg", sample_image_bytes, "image/jpeg")},
        )

        assert response.status_code == 201
        data = response.json()
        assert "property_id" in data
        assert data["message"].startswith("Property 'Test House'")
        # The response should include full property details
        prop = data["property"]
        assert prop["name"] == "Test House"
        assert prop["extra_info"] == "Near the park"
        assert prop["model_used"] == "fake-model"
        assert len(prop["images"]) == 1

    def test_upload_creates_amenity_records(self, client: TestClient, sample_image_bytes: bytes):
        """Detection results should be present in the response's amenities list."""
        response = client.post(
            "/api/v1/properties/upload",
            data={"name": "Amenity Check", "model_name": "fake-model"},
            files={"files": ("room.jpg", sample_image_bytes, "image/jpeg")},
        )

        assert response.status_code == 201
        prop = response.json()["property"]
        images = prop["images"]
        assert len(images) == 1

        amenities = images[0]["amenities"]
        # The fake VLM returns refrigerator=True, oven=True, dishwasher=False
        present = {a["amenity_name"]: a["is_present"] for a in amenities}
        assert present.get("refrigerator") is True
        assert present.get("oven") is True
        assert present.get("dishwasher") is False

    def test_upload_no_files_returns_400(self, client: TestClient):
        """Submitting the form with no files should return 400."""
        response = client.post(
            "/api/v1/properties/upload",
            data={"name": "No Files", "model_name": "fake-model"},
            # No files= argument
        )
        assert response.status_code == 422  # FastAPI validation: files is required

    def test_upload_unknown_model_returns_400(self, client: TestClient, sample_image_bytes: bytes):
        """Requesting an unregistered model should return 400."""
        response = client.post(
            "/api/v1/properties/upload",
            data={"name": "Unknown Model", "model_name": "nonexistent-model"},
            files={"files": ("img.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 400
        assert "not available" in response.json()["detail"].lower()

    def test_upload_non_image_file_returns_400(self, client: TestClient):
        """Uploading a text file should be rejected at content-type validation."""
        response = client.post(
            "/api/v1/properties/upload",
            data={"name": "Bad File", "model_name": "fake-model"},
            files={"files": ("notes.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 400
        assert "unsupported type" in response.json()["detail"].lower()

    def test_upload_multiple_images(self, client: TestClient, sample_image_bytes: bytes):
        """Multiple images in one upload should all be stored and processed."""
        response = client.post(
            "/api/v1/properties/upload",
            data={"name": "Multi Image", "model_name": "fake-model"},
            files=[
                ("files", ("kitchen.jpg", sample_image_bytes, "image/jpeg")),
                ("files", ("bedroom.jpg", sample_image_bytes, "image/jpeg")),
            ],
        )
        assert response.status_code == 201
        prop = response.json()["property"]
        assert len(prop["images"]) == 2


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
        # Upload a property first
        client.post(
            "/api/v1/properties/upload",
            data={"name": "Listed Property", "model_name": "fake-model"},
            files={"files": ("img.jpg", sample_image_bytes, "image/jpeg")},
        )

        response = client.get("/api/v1/properties/")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        names = [p["name"] for p in data]
        assert "Listed Property" in names

    def test_list_pagination_limit(self, client: TestClient, sample_image_bytes: bytes):
        """The limit parameter should cap the number of results returned."""
        # Upload 3 properties
        for i in range(3):
            client.post(
                "/api/v1/properties/upload",
                data={"name": f"Paginated {i}", "model_name": "fake-model"},
                files={"files": ("img.jpg", sample_image_bytes, "image/jpeg")},
            )

        response = client.get("/api/v1/properties/?limit=2")
        assert response.status_code == 200
        assert len(response.json()) <= 2


class TestDetailEndpoint:
    def test_get_existing_property_returns_200(self, client: TestClient, sample_image_bytes: bytes):
        """Getting a property by its ID should return full details."""
        upload_resp = client.post(
            "/api/v1/properties/upload",
            data={"name": "Detail Test", "model_name": "fake-model"},
            files={"files": ("img.jpg", sample_image_bytes, "image/jpeg")},
        )
        property_id = upload_resp.json()["property_id"]

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
        # The fake VLM always returns refrigerator=True
        client.post(
            "/api/v1/properties/upload",
            data={"name": "Searchable Kitchen", "model_name": "fake-model"},
            files={"files": ("kitchen.jpg", sample_image_bytes, "image/jpeg")},
        )

        response = client.get("/api/v1/properties/search?amenities=refrigerator")
        assert response.status_code == 200
        names = [p["name"] for p in response.json()]
        assert "Searchable Kitchen" in names

    def test_search_excludes_non_matching_property(
        self, client: TestClient, sample_image_bytes: bytes
    ):
        """
        Searching for an amenity the fake VLM never returns (pool)
        should return an empty list.
        """
        client.post(
            "/api/v1/properties/upload",
            data={"name": "No Pool Here", "model_name": "fake-model"},
            files={"files": ("img.jpg", sample_image_bytes, "image/jpeg")},
        )

        # The fake VLM never returns pool=True
        response = client.get("/api/v1/properties/search?amenities=pool")
        assert response.status_code == 200
        # "No Pool Here" should NOT be in the results
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
        upload_resp = client.post(
            "/api/v1/properties/upload",
            data={"name": "To Be Deleted", "model_name": "fake-model"},
            files={"files": ("img.jpg", sample_image_bytes, "image/jpeg")},
        )
        property_id = upload_resp.json()["property_id"]

        response = client.delete(f"/api/v1/properties/{property_id}")
        assert response.status_code == 204

    def test_deleted_property_no_longer_retrievable(
        self, client: TestClient, sample_image_bytes: bytes
    ):
        """After deletion, GET should return 404."""
        upload_resp = client.post(
            "/api/v1/properties/upload",
            data={"name": "Delete Then Get", "model_name": "fake-model"},
            files={"files": ("img.jpg", sample_image_bytes, "image/jpeg")},
        )
        property_id = upload_resp.json()["property_id"]

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
        assert len(data) > 0

    def test_models_include_required_fields(self, client: TestClient):
        """Each model entry must have name, available, and description."""
        response = client.get("/api/v1/models/")
        for model in response.json():
            assert "name" in model
            assert "available" in model
            assert "description" in model
