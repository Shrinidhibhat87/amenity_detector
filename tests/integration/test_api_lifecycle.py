"""Integration tests for the publication lifecycle over HTTP.

These walk the same route the wizard walks — create, upload, describe, save,
publish — and assert the status the server derives at each step, plus the
refusal to publish an incomplete listing.
"""

from fastapi.testclient import TestClient

MODEL = "openai/gpt-4o-mini"


def _create(client: TestClient, name: str = "Lifecycle flat") -> str:
    res = client.post("/api/v1/properties/", json={"name": name, "model_name": MODEL})
    assert res.status_code == 201
    property_id: str = res.json()["property_id"]
    return property_id


def _upload(client: TestClient, property_id: str, image_bytes: bytes) -> None:
    res = client.post(
        f"/api/v1/properties/{property_id}/images",
        files={"file": ("kitchen.jpg", image_bytes, "image/jpeg")},
        data={"model_name": MODEL},
    )
    assert res.status_code == 201


def _status(client: TestClient, property_id: str) -> str:
    res = client.get(f"/api/v1/properties/{property_id}")
    assert res.status_code == 200
    status: str = res.json()["status"]
    return status


class TestStatusProgression:
    def test_new_property_is_a_draft(self, client: TestClient) -> None:
        property_id = _create(client)
        assert _status(client, property_id) == "draft"

    def test_uploading_an_image_moves_the_draft_to_processing(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        property_id = _create(client)
        _upload(client, property_id, sample_image_bytes)
        assert _status(client, property_id) == "processing"

    def test_generating_a_description_marks_it_ready_for_review(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        property_id = _create(client)
        _upload(client, property_id, sample_image_bytes)

        res = client.post(
            f"/api/v1/properties/{property_id}/describe",
            json={"amenities": [], "model_name": MODEL},
        )
        assert res.status_code == 200
        assert _status(client, property_id) == "ready_for_review"

    def test_saving_the_description_completes_the_listing(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        property_id = _create(client)
        _upload(client, property_id, sample_image_bytes)

        res = client.patch(
            f"/api/v1/properties/{property_id}", json={"description": "A bright flat."}
        )
        assert res.status_code == 200
        assert res.json()["status"] == "completed"

    def test_editing_a_published_listing_does_not_unpublish_it(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        property_id = _create(client)
        _upload(client, property_id, sample_image_bytes)
        client.patch(f"/api/v1/properties/{property_id}", json={"description": "A flat."})
        client.post(f"/api/v1/properties/{property_id}/publish")

        res = client.patch(
            f"/api/v1/properties/{property_id}", json={"description": "A brighter flat."}
        )
        assert res.status_code == 200
        assert res.json()["status"] == "published"


class TestPublishEndpoint:
    def test_publishing_a_complete_listing_succeeds(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        property_id = _create(client)
        _upload(client, property_id, sample_image_bytes)
        client.patch(f"/api/v1/properties/{property_id}", json={"description": "A flat."})

        res = client.post(f"/api/v1/properties/{property_id}/publish")
        assert res.status_code == 200
        assert res.json()["status"] == "published"

    def test_publishing_without_a_description_is_refused(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        property_id = _create(client)
        _upload(client, property_id, sample_image_bytes)

        res = client.post(f"/api/v1/properties/{property_id}/publish")
        assert res.status_code == 409
        assert "description" in res.json()["detail"].lower()
        assert _status(client, property_id) == "processing"

    def test_publishing_without_an_image_is_refused(self, client: TestClient) -> None:
        property_id = _create(client)
        client.patch(f"/api/v1/properties/{property_id}", json={"description": "A flat."})

        res = client.post(f"/api/v1/properties/{property_id}/publish")
        assert res.status_code == 409
        assert "image" in res.json()["detail"].lower()

    def test_publishing_an_unknown_property_is_404(self, client: TestClient) -> None:
        res = client.post("/api/v1/properties/does-not-exist/publish")
        assert res.status_code == 404
