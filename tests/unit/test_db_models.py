"""
Unit tests for db/models.py — ORM model definitions.

These tests use an in-memory SQLite database (no PostgreSQL, no Docker needed).
They verify:
  1. Tables can be created from the ORM models
  2. Records can be inserted, read back, and deleted
  3. Relationships (Property → PropertyImage → DetectedAmenity) work correctly
  4. CASCADE delete removes child records when a parent is deleted
  5. Default values (UUID, timestamp) are auto-populated
"""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base, DetectedAmenity, Property, PropertyImage


@pytest.fixture(scope="function")
def db_session() -> Session:
    """
    Create a fresh in-memory SQLite database for each test function.

    scope="function" means every test gets a clean database — no test pollution.
    SQLite in-memory is faster than file-based and cleans up automatically.
    """
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


class TestPropertyModel:
    def test_create_minimal_property(self, db_session: Session):
        """A property with only a name should be insertable."""
        prop = Property(name="Test Property")
        db_session.add(prop)
        db_session.commit()

        retrieved = db_session.get(Property, prop.id)
        assert retrieved is not None
        assert retrieved.name == "Test Property"

    def test_id_is_auto_generated_uuid(self, db_session: Session):
        """The id field should be auto-populated as a UUID string."""
        prop = Property(name="Auto ID Test")
        db_session.add(prop)
        db_session.commit()

        assert prop.id is not None
        assert len(prop.id) == 36  # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        assert prop.id.count("-") == 4

    def test_two_properties_get_different_ids(self, db_session: Session):
        """Each property must have a unique ID."""
        p1 = Property(name="House A")
        p2 = Property(name="House B")
        db_session.add_all([p1, p2])
        db_session.commit()

        assert p1.id != p2.id

    def test_created_at_is_set_automatically(self, db_session: Session):
        """created_at should be populated without us setting it explicitly."""
        prop = Property(name="Timestamp Test")
        db_session.add(prop)
        db_session.commit()

        assert prop.created_at is not None
        # created_at should be roughly "now" (within a few seconds of the test)
        # We just check it's a datetime — the exact value depends on clock speed
        assert isinstance(prop.created_at, datetime)

    def test_optional_fields_default_to_none(self, db_session: Session):
        """description, model_used, extra_info should all default to None."""
        prop = Property(name="Minimal")
        db_session.add(prop)
        db_session.commit()

        assert prop.description is None
        assert prop.model_used is None
        assert prop.extra_info is None

    def test_all_optional_fields_can_be_set(self, db_session: Session):
        """All optional fields should be storable and retrievable."""
        prop = Property(
            name="Full Property",
            description="A lovely flat",
            model_used="gemini-2.0-flash",
            extra_info="Near the park, 2 bedrooms",
        )
        db_session.add(prop)
        db_session.commit()

        retrieved = db_session.get(Property, prop.id)
        assert retrieved.description == "A lovely flat"
        assert retrieved.model_used == "gemini-2.0-flash"
        assert retrieved.extra_info == "Near the park, 2 bedrooms"


class TestPropertyImageModel:
    def test_create_image_for_property(self, db_session: Session):
        """An image with a valid property_id should be insertable."""
        prop = Property(name="Parent Property")
        db_session.add(prop)
        db_session.flush()

        img = PropertyImage(property_id=prop.id, file_path="abc123/kitchen.jpg")
        db_session.add(img)
        db_session.commit()

        retrieved = db_session.get(PropertyImage, img.id)
        assert retrieved is not None
        assert retrieved.file_path == "abc123/kitchen.jpg"
        assert retrieved.property_id == prop.id

    def test_image_accessible_via_property_relationship(self, db_session: Session):
        """The Property.images relationship should include the child image."""
        prop = Property(name="Relship Test")
        db_session.add(prop)
        db_session.flush()

        img = PropertyImage(property_id=prop.id, file_path="p/i.jpg")
        db_session.add(img)
        db_session.commit()

        db_session.refresh(prop)
        assert len(prop.images) == 1
        assert prop.images[0].id == img.id


class TestDetectedAmenityModel:
    def test_create_amenity_for_image(self, db_session: Session):
        """A DetectedAmenity should link to both a property and an image."""
        prop = Property(name="Amenity Prop")
        db_session.add(prop)
        db_session.flush()

        img = PropertyImage(property_id=prop.id, file_path="p/img.jpg")
        db_session.add(img)
        db_session.flush()

        amenity = DetectedAmenity(
            property_id=prop.id,
            image_id=img.id,
            amenity_name="refrigerator",
            room_type="kitchen",
            is_present=True,
            confidence=1.0,
        )
        db_session.add(amenity)
        db_session.commit()

        retrieved = db_session.get(DetectedAmenity, amenity.id)
        assert retrieved is not None
        assert retrieved.amenity_name == "refrigerator"
        assert retrieved.is_present is True
        assert retrieved.confidence == 1.0

    def test_false_amenity_stored_correctly(self, db_session: Session):
        """is_present=False should be stored and retrieved as False."""
        prop = Property(name="P")
        db_session.add(prop)
        db_session.flush()

        img = PropertyImage(property_id=prop.id, file_path="x.jpg")
        db_session.add(img)
        db_session.flush()

        amenity = DetectedAmenity(
            property_id=prop.id,
            image_id=img.id,
            amenity_name="pool",
            is_present=False,
            confidence=0.0,
        )
        db_session.add(amenity)
        db_session.commit()

        db_session.refresh(amenity)
        assert amenity.is_present is False


class TestCascadeDelete:
    def test_deleting_property_deletes_images_and_amenities(self, db_session: Session):
        """
        Deleting a Property should CASCADE to its PropertyImages and DetectedAmenities.
        This verifies the cascade="all, delete-orphan" setting on the relationships.
        """
        prop = Property(name="Cascade Test")
        db_session.add(prop)
        db_session.flush()

        img = PropertyImage(property_id=prop.id, file_path="img.jpg")
        db_session.add(img)
        db_session.flush()

        amenity = DetectedAmenity(
            property_id=prop.id,
            image_id=img.id,
            amenity_name="pool",
            is_present=True,
        )
        db_session.add(amenity)
        db_session.commit()

        # Store IDs before deletion
        prop_id = prop.id
        img_id = img.id
        amenity_id = amenity.id

        # Delete the property
        db_session.delete(prop)
        db_session.commit()

        # All related records should be gone
        assert db_session.get(Property, prop_id) is None
        assert db_session.get(PropertyImage, img_id) is None
        assert db_session.get(DetectedAmenity, amenity_id) is None
