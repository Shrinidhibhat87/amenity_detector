"""Integration tests for GET /api/v1/images/{image_id}."""

from pathlib import Path

from api.dependencies import get_image_storage_dir
from db.models import PropertyImage
from db.session import get_db


def _upload_image(client, sample_image_bytes: bytes) -> tuple[str, str]:
    create = client.post(
        "/api/v1/properties/",
        json={"name": "Img Flat", "model_name": "openai/gpt-4o-mini"},
    )
    assert create.status_code == 201
    pid = create.json()["property_id"]

    upload = client.post(
        f"/api/v1/properties/{pid}/images",
        data={"model_name": "openai/gpt-4o-mini"},
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert upload.status_code == 201
    return pid, upload.json()["image"]["id"]


def test_serve_image_returns_bytes(client, sample_image_bytes: bytes) -> None:
    _pid, image_id = _upload_image(client, sample_image_bytes)

    resp = client.get(f"/api/v1/images/{image_id}")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")
    assert len(resp.content) > 0


def test_serve_image_404_for_unknown_id(client) -> None:
    resp = client.get("/api/v1/images/not-a-real-id")

    assert resp.status_code == 404


def test_serve_image_404_when_file_missing(client, sample_image_bytes: bytes) -> None:
    _pid, image_id = _upload_image(client, sample_image_bytes)
    session = client.app.dependency_overrides[get_db]()
    storage_dir = client.app.dependency_overrides[get_image_storage_dir]()
    img = session.get(PropertyImage, image_id)
    assert img is not None
    Path(storage_dir, img.file_path).unlink()

    resp = client.get(f"/api/v1/images/{image_id}")

    assert resp.status_code == 404


def test_serve_image_blocks_path_traversal(client, sample_image_bytes: bytes, tmp_path) -> None:
    _pid, image_id = _upload_image(client, sample_image_bytes)
    rogue = tmp_path / "rogue.txt"
    rogue.write_text("secret", encoding="utf-8")

    session = client.app.dependency_overrides[get_db]()
    img = session.get(PropertyImage, image_id)
    assert img is not None
    img.file_path = str(rogue)
    session.commit()

    resp = client.get(f"/api/v1/images/{image_id}")

    assert resp.status_code == 403
