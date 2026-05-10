"""Integration tests for PATCH /api/v1/images/{image_id} (Phase 9)."""

from typing import cast

from fastapi.testclient import TestClient


def _upload_one_image(client: TestClient, sample_image_bytes: bytes) -> tuple[str, str]:
    create = client.post(
        "/api/v1/properties/",
        json={"name": "Patch Img", "model_name": "openai/gpt-4o-mini"},
    )
    pid = cast(str, create.json()["property_id"])
    upload = client.post(
        f"/api/v1/properties/{pid}/images",
        data={"model_name": "openai/gpt-4o-mini"},
        files={"file": ("a.jpg", sample_image_bytes, "image/jpeg")},
    )
    return pid, cast(str, upload.json()["image"]["id"])


def test_patch_image_updates_alt_text_and_caption(
    client: TestClient, sample_image_bytes: bytes
) -> None:
    _pid, image_id = _upload_one_image(client, sample_image_bytes)
    response = client.patch(
        f"/api/v1/images/{image_id}",
        json={"alt_text": "Edited alt text", "caption": "Renovated 2025"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["alt_text"] == "Edited alt text"
    assert body["caption"] == "Renovated 2025"


def test_patch_image_set_is_primary_clears_siblings(
    client: TestClient, sample_image_bytes: bytes
) -> None:
    create = client.post(
        "/api/v1/properties/",
        json={"name": "Multi Img", "model_name": "openai/gpt-4o-mini"},
    )
    pid = create.json()["property_id"]
    image_ids: list[str] = []
    for filename in ("kitchen.jpg", "bedroom.jpg", "bath.jpg"):
        upload = client.post(
            f"/api/v1/properties/{pid}/images",
            data={"model_name": "openai/gpt-4o-mini"},
            files={"file": (filename, sample_image_bytes, "image/jpeg")},
        )
        image_ids.append(upload.json()["image"]["id"])

    # Mark the first image primary, then the second — the first must be cleared.
    client.patch(f"/api/v1/images/{image_ids[0]}", json={"is_primary": True})
    client.patch(f"/api/v1/images/{image_ids[1]}", json={"is_primary": True})

    detail = client.get(f"/api/v1/properties/{pid}").json()
    primary_flags = {img["id"]: img["is_primary"] for img in detail["images"]}
    assert primary_flags[image_ids[0]] is False
    assert primary_flags[image_ids[1]] is True
    assert primary_flags[image_ids[2]] is False


def test_patch_image_404_for_unknown(client: TestClient) -> None:
    response = client.patch("/api/v1/images/no-such-id", json={"alt_text": "x"})
    assert response.status_code == 404


def test_patch_image_rejects_unknown_keys(client: TestClient, sample_image_bytes: bytes) -> None:
    _pid, image_id = _upload_one_image(client, sample_image_bytes)
    response = client.patch(
        f"/api/v1/images/{image_id}",
        json={"file_path": "/etc/passwd"},
    )
    assert response.status_code == 422


def test_patch_image_empty_body_rejected(client: TestClient, sample_image_bytes: bytes) -> None:
    _pid, image_id = _upload_one_image(client, sample_image_bytes)
    response = client.patch(f"/api/v1/images/{image_id}", json={})
    assert response.status_code == 400


def test_patch_image_display_order_round_trip(
    client: TestClient, sample_image_bytes: bytes
) -> None:
    _pid, image_id = _upload_one_image(client, sample_image_bytes)
    response = client.patch(f"/api/v1/images/{image_id}", json={"display_order": 5})
    assert response.status_code == 200
    assert response.json()["display_order"] == 5
