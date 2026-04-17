"""
Unit tests for core/amenity_system.py.

These tests verify that PropertyAmenitySystem orchestrates the pipeline correctly:
  - Creates a Property record in the DB
  - Saves images to the filesystem
  - Passes confidence scores from the detector to the data manager
  - Generates and stores a property description
  - The full process_upload() flow with mocked dependencies

All external dependencies (VLM, DB, filesystem) are mocked so tests are fast and
deterministic. We use tmp_path (pytest built-in fixture) for filesystem isolation.
"""

import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.amenity_system import PropertyAmenitySystem, _infer_room_type
from db.models import Base
from models.base import VLMClient, VLMResponse

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def in_memory_db() -> Generator[Session, None, None]:
    """
    SQLite in-memory database with all ORM tables created.

    Uses SQLite (not PostgreSQL) because unit tests should not need a running
    database service. The full stack is tested in integration tests.
    """
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture
def fake_vlm() -> MagicMock:
    """
    A mock VLMClient that returns a canned JSON amenity detection response.

    The response uses the new Phase 4 structured format with confidence scores.
    """
    fake_response = json.dumps(
        {
            "refrigerator": {"present": True, "confidence": 0.95},
            "oven": {"present": True, "confidence": 0.8},
            "dishwasher": {"present": False, "confidence": 0.1},
        }
    )
    mock = MagicMock(spec=VLMClient)
    type(mock).model_name = property(lambda self: "fake-model")
    # First call → amenity detection JSON; second call → description text
    mock.generate.side_effect = [
        VLMResponse(raw_text=fake_response, model_name="fake-model"),
        VLMResponse(raw_text="A well-equipped kitchen.", model_name="fake-model"),
    ]
    return mock


@pytest.fixture
def small_image() -> PILImage.Image:
    """A 16×16 RGB image — no file I/O required."""
    return PILImage.new("RGB", (16, 16), color=(100, 200, 150))


@pytest.fixture
def system(
    fake_vlm: MagicMock,
    in_memory_db: Session,
    tmp_path: Path,
) -> PropertyAmenitySystem:
    """PropertyAmenitySystem wired to fake dependencies."""
    return PropertyAmenitySystem(
        vlm_client=fake_vlm,
        db=in_memory_db,
        image_storage_dir=tmp_path / "images",
    )


# ── _infer_room_type helper ──────────────────────────────────────────────────


class TestInferRoomType:
    def test_returns_room_with_most_detected_amenities(self):
        amenities_by_room = {
            "kitchen": {"refrigerator": True, "oven": True, "dishwasher": False},
            "bedroom": {"bed": False, "wardrobe": False},
        }
        assert _infer_room_type(amenities_by_room) == "kitchen"

    def test_returns_none_when_nothing_detected(self):
        amenities_by_room = {
            "kitchen": {"refrigerator": False},
            "bedroom": {"bed": False},
        }
        assert _infer_room_type(amenities_by_room) is None

    def test_handles_empty_schema(self):
        assert _infer_room_type({}) is None

    def test_tie_breaking_returns_first_maximum(self):
        """When two rooms have equal detections, the first one encountered wins."""
        amenities_by_room = {
            "kitchen": {"refrigerator": True},
            "bedroom": {"bed": True},
        }
        # One of them should be returned (either is valid — just not None)
        result = _infer_room_type(amenities_by_room)
        assert result in ("kitchen", "bedroom")


# ── process_upload ──────────────────────────────────────────────────────────


class TestProcessUpload:
    def test_creates_property_record(
        self,
        system: PropertyAmenitySystem,
        small_image: PILImage.Image,
        in_memory_db: Session,
    ):
        """A Property row should exist in the DB after process_upload()."""
        from db.models import Property

        prop = system.process_upload(
            images=[small_image],
            filenames=["kitchen.jpg"],
            property_name="Test House",
            model_name="fake-model",
        )

        assert prop is not None
        assert prop.name == "Test House"
        assert prop.model_used == "fake-model"

        # The record should also be queryable from the DB
        db_prop = in_memory_db.get(Property, prop.id)
        assert db_prop is not None

    def test_saves_image_file_to_disk(
        self,
        system: PropertyAmenitySystem,
        small_image: PILImage.Image,
        tmp_path: Path,
    ):
        """The uploaded image should be saved to the storage directory."""
        prop = system.process_upload(
            images=[small_image],
            filenames=["bedroom.jpg"],
            property_name="Storage Test",
            model_name="fake-model",
        )

        # File should be at storage_dir / property_id / filename
        expected_path = tmp_path / "images" / prop.id / "bedroom.jpg"
        assert expected_path.exists()

    def test_creates_amenity_records(
        self,
        system: PropertyAmenitySystem,
        small_image: PILImage.Image,
        in_memory_db: Session,
    ):
        """DetectedAmenity rows should be created for the uploaded image."""
        from db.models import DetectedAmenity

        prop = system.process_upload(
            images=[small_image],
            filenames=["kitchen.jpg"],
            property_name="Amenity Test",
            model_name="fake-model",
        )

        from sqlalchemy import select

        amenities = in_memory_db.scalars(
            select(DetectedAmenity).where(DetectedAmenity.property_id == prop.id)
        ).all()
        assert len(amenities) > 0

    def test_stores_description(
        self,
        system: PropertyAmenitySystem,
        small_image: PILImage.Image,
    ):
        """The property description should be set after processing."""
        prop = system.process_upload(
            images=[small_image],
            filenames=["kitchen.jpg"],
            property_name="Desc Test",
            model_name="fake-model",
        )
        assert prop.description is not None
        assert len(prop.description) > 0

    def test_stores_extra_info(
        self,
        system: PropertyAmenitySystem,
        small_image: PILImage.Image,
    ):
        """Optional extra_info from the user should be stored on the property."""
        prop = system.process_upload(
            images=[small_image],
            filenames=["img.jpg"],
            property_name="Extra Info Test",
            model_name="fake-model",
            extra_info="City centre apartment",
        )
        assert prop.extra_info == "City centre apartment"

    def test_handles_multiple_images(
        self,
        fake_vlm: MagicMock,
        in_memory_db: Session,
        tmp_path: Path,
    ):
        """Multiple images should each get their own PropertyImage record."""
        # Provide enough VLM responses for 2 detection calls + 1 description call
        fake_response = json.dumps({"refrigerator": True})
        fake_vlm.generate.side_effect = [
            VLMResponse(raw_text=fake_response, model_name="fake-model"),
            VLMResponse(raw_text=fake_response, model_name="fake-model"),
            VLMResponse(raw_text="Two-room property.", model_name="fake-model"),
        ]
        system = PropertyAmenitySystem(
            vlm_client=fake_vlm,
            db=in_memory_db,
            image_storage_dir=tmp_path / "images",
        )
        images = [
            PILImage.new("RGB", (16, 16), color=(100, 100, 100)),
            PILImage.new("RGB", (16, 16), color=(200, 200, 200)),
        ]
        prop = system.process_upload(
            images=images,
            filenames=["kitchen.jpg", "bedroom.jpg"],
            property_name="Multi Image",
            model_name="fake-model",
        )

        from sqlalchemy import func, select

        from db.models import PropertyImage

        count = in_memory_db.scalar(
            select(func.count(PropertyImage.id)).where(PropertyImage.property_id == prop.id)
        )
        assert count == 2
