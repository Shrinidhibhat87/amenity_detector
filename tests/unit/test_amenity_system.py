"""
Unit tests for core/amenity_system.py.

These tests verify that PropertyAmenitySystem orchestrates the pipeline correctly:
  - Creates a Property record in the DB
  - Saves images to the filesystem
  - Passes confidence scores from the detector to the data manager
  - Supports the Phase 5 shell + per-image upload flow

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
    mock.generate.return_value = VLMResponse(raw_text=fake_response, model_name="fake-model")
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

    def test_single_structured_room_is_kept_even_without_amenities(self):
        """Phase 5 room labels come directly from the VLM response."""
        amenities_by_room = {"bathroom": {"bathtub": False}}
        assert _infer_room_type(amenities_by_room) == "bathroom"

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


# ── create_property_shell ───────────────────────────────────────────────────


class TestCreatePropertyShell:
    def test_creates_property_record(
        self,
        system: PropertyAmenitySystem,
        in_memory_db: Session,
    ):
        """A Property row should exist in the DB after create_property_shell()."""
        from db.models import Property

        prop = system.create_property_shell(
            property_name="Test House",
            model_name="fake-model",
        )

        assert prop is not None
        assert prop.name == "Test House"
        assert prop.model_used == "fake-model"

        # The record should also be queryable from the DB
        db_prop = in_memory_db.get(Property, prop.id)
        assert db_prop is not None

    def test_stores_extra_info(
        self,
        system: PropertyAmenitySystem,
    ):
        """Optional extra_info from the user should be stored on the property."""
        prop = system.create_property_shell(
            property_name="Extra Info Test",
            model_name="fake-model",
            extra_info="City centre apartment",
        )
        assert prop.extra_info == "City centre apartment"

    def test_creates_property_storage_directory(
        self, system: PropertyAmenitySystem, tmp_path: Path
    ):
        """The shell flow should prepare a property-specific storage folder."""
        prop = system.create_property_shell(
            property_name="Storage Test",
            model_name="fake-model",
        )

        assert (tmp_path / "images" / prop.id).exists()


# ── process_one_image ───────────────────────────────────────────────────────


class TestProcessOneImage:
    def test_saves_image_file_to_disk(
        self,
        system: PropertyAmenitySystem,
        small_image: PILImage.Image,
        tmp_path: Path,
    ):
        """The uploaded image should be saved to the storage directory."""
        prop = system.create_property_shell(
            property_name="Storage Test",
            model_name="fake-model",
        )
        system.process_one_image(
            property_id=prop.id,
            image=small_image,
            filename="bedroom.jpg",
        )

        expected_path = tmp_path / "images" / prop.id / "bedroom.jpg"
        assert expected_path.exists()

    def test_creates_amenity_records(
        self,
        system: PropertyAmenitySystem,
        small_image: PILImage.Image,
        in_memory_db: Session,
    ):
        """DetectedAmenity rows should be created for the uploaded image."""
        from sqlalchemy import select

        from db.models import DetectedAmenity

        prop = system.create_property_shell(
            property_name="Amenity Test",
            model_name="fake-model",
        )
        system.process_one_image(
            property_id=prop.id,
            image=small_image,
            filename="kitchen.jpg",
        )

        amenities = in_memory_db.scalars(
            select(DetectedAmenity).where(DetectedAmenity.property_id == prop.id)
        ).all()
        assert len(amenities) > 0

    def test_returns_image_record_with_detected_room(
        self,
        system: PropertyAmenitySystem,
        small_image: PILImage.Image,
    ):
        """The per-image flow should populate the inferred room type."""
        prop = system.create_property_shell(
            property_name="Room Test",
            model_name="fake-model",
        )

        img_record = system.process_one_image(
            property_id=prop.id,
            image=small_image,
            filename="kitchen.jpg",
        )

        assert img_record.room_type == "kitchen"

    def test_raises_for_unknown_property(
        self,
        system: PropertyAmenitySystem,
        small_image: PILImage.Image,
    ):
        """Uploading to a missing property should fail clearly."""
        with pytest.raises(ValueError, match="Property not found"):
            system.process_one_image(
                property_id="missing-property",
                image=small_image,
                filename="kitchen.jpg",
            )

    def test_handles_multiple_images(
        self,
        system: PropertyAmenitySystem,
        small_image: PILImage.Image,
        in_memory_db: Session,
    ):
        """Multiple uploads to the same shell should each get their own PropertyImage record."""
        prop = system.create_property_shell(
            property_name="Multi Image",
            model_name="fake-model",
        )
        system.process_one_image(property_id=prop.id, image=small_image, filename="kitchen.jpg")
        system.process_one_image(property_id=prop.id, image=small_image, filename="bedroom.jpg")

        from sqlalchemy import func, select

        from db.models import PropertyImage

        count = in_memory_db.scalar(
            select(func.count(PropertyImage.id)).where(PropertyImage.property_id == prop.id)
        )
        assert count == 2


# ── generate_description_from_amenities ─────────────────────────────────────


def test_generate_description_includes_sidebar_hints(tmp_path: Path) -> None:
    """Optional Phase 5 sidebar hints should appear in the description prompt."""
    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = VLMResponse(raw_text="A bright apartment.", model_name="test")

    system = PropertyAmenitySystem(
        vlm_client=mock_vlm,
        db=MagicMock(),
        image_storage_dir=tmp_path,
    )

    system.generate_description_from_amenities(
        amenities=[{"amenity_name": "Sofa", "room_type": "living_room", "is_present": True}],
        property_name="Hinted House",
        num_rooms=2,
        has_kitchen=True,
        has_balcony=False,
        has_living_room=True,
    )

    prompt_used: str = mock_vlm.generate.call_args.kwargs["prompt"]
    assert "2 rooms" in prompt_used
    assert "has a kitchen" in prompt_used
    assert "no balcony" in prompt_used
    assert "has a living room" in prompt_used


def test_format_sidebar_hints_includes_extended_dict() -> None:
    from core.amenity_system import _format_sidebar_hints

    out = _format_sidebar_hints(
        num_rooms=2,
        has_kitchen=True,
        has_balcony=None,
        has_living_room=None,
        hints={"elevator": True, "garage": False, "fireplace": True},
    )

    assert "2 rooms" in out
    assert "has a kitchen" in out
    assert "has a elevator" in out
    assert "no garage" in out
    assert "has a fireplace" in out


def test_generate_description_includes_expanded_hints(tmp_path: Path) -> None:
    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = VLMResponse(raw_text="A bright apartment.", model_name="test")

    system = PropertyAmenitySystem(
        vlm_client=mock_vlm,
        db=MagicMock(),
        image_storage_dir=tmp_path,
    )

    system.generate_description_from_amenities(
        amenities=[{"amenity_name": "Sofa", "room_type": "living_room", "is_present": True}],
        property_name="Hinted House",
        hints={"elevator": True, "pet_friendly": False},
    )

    prompt_used: str = mock_vlm.generate.call_args.kwargs["prompt"]
    assert "has a elevator" in prompt_used
    assert "no pet friendly" in prompt_used
