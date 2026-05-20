"""
SQLAlchemy ORM models for the amenity detector.

These classes map directly to the database tables defined in SPEC.md.
SQLAlchemy 2.0 uses the "mapped_column + Mapped[]" style which gives us
full type-checker support — mypy can verify column types at analysis time.

Table overview:
  properties        — one row per real estate property submitted by the user
  images            — one row per uploaded image (many per property)
  detected_amenities — one row per amenity checked per image (many per image)

Why UUID as String(36)?
  PostgreSQL has a native UUID type, but SQLite (used in tests) does not.
  Storing UUIDs as CHAR(36) strings works transparently on both databases,
  so we can run tests without Docker while still using PostgreSQL in production.
"""

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Numeric,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from db.types import Embedding


class Base(DeclarativeBase):
    """
    Base class that all ORM models inherit from.

    DeclarativeBase (SQLAlchemy 2.0) replaces the older declarative_base() function.
    All models that inherit from Base are automatically picked up by Alembic.
    """

    pass


class Property(Base):
    """
    A real estate property submitted by the user for amenity detection.

    One property can have many images. The description field is populated
    by the VLM after processing all uploaded images.
    """

    __tablename__ = "properties"

    # Primary key — we generate UUID in Python so we know the ID before the DB INSERT.
    # This makes it easy to associate images and amenities before committing.
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Human-readable label provided by the user at upload time (e.g., "Frankfurt House 1")
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # VLM-generated description, written after all images are processed.
    # Nullable because it's set after image processing, not at creation time.
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Which VLM produced the amenity detections (stored for auditing / comparison)
    model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Free-text metadata provided by the user (e.g., "2-bed flat, central Frankfurt")
    extra_info: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Auto-set at INSERT time — tells us when this property was processed.
    # timezone=True stores as TIMESTAMPTZ in PostgreSQL (always UTC-aware).
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # --- Phase 9 listing metadata --------------------------------------------
    # All fields below are nullable so legacy rows continue to load.
    # Enum-like columns (listing_type, price_period, property_type, furnishing)
    # are stored as plain strings; Pydantic schemas at the API boundary enforce
    # the allowed value set so we keep portability across SQLite tests and
    # PostgreSQL production without DB-native ENUMs.
    slug: Mapped[str | None] = mapped_column(String(160), unique=True, nullable=True)
    listing_type: Mapped[str | None] = mapped_column(String(8), nullable=True)
    price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    price_period: Mapped[str | None] = mapped_column(String(8), nullable=True)
    num_bedrooms: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    num_bathrooms: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    area_sqm: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)
    property_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    furnishing: Mapped[str | None] = mapped_column(String(16), nullable=True)
    available_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    locality: Mapped[str | None] = mapped_column(String(120), nullable=True)
    postal_code: Mapped[str | None] = mapped_column(String(16), nullable=True)
    country_code: Mapped[str | None] = mapped_column(String(2), nullable=True)
    latitude: Mapped[Decimal | None] = mapped_column(Numeric(9, 6), nullable=True)
    longitude: Mapped[Decimal | None] = mapped_column(Numeric(9, 6), nullable=True)
    owner_email: Mapped[str | None] = mapped_column(String(255), nullable=True)

    description_embedding: Mapped[list[float] | None] = mapped_column(
        Embedding(),
        nullable=True,
    )

    # --- Relationships --------------------------------------------------------
    # back_populates = "property" means PropertyImage.property points back here.
    # cascade="all, delete-orphan" means deleting a Property also deletes its images
    # and amenities — no orphaned rows.
    images: Mapped[list["PropertyImage"]] = relationship(
        "PropertyImage", back_populates="property", cascade="all, delete-orphan"
    )
    amenities: Mapped[list["DetectedAmenity"]] = relationship(
        "DetectedAmenity", back_populates="property", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Property id={self.id!r} name={self.name!r}>"


class PropertyImage(Base):
    """
    A single image uploaded for a property.

    Each image gets its own detection pass — the VLM analyses one image at a time.
    The detected room_type (kitchen, bedroom, etc.) is filled in after detection.
    """

    __tablename__ = "images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Foreign key links this image to its parent property
    property_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("properties.id"), nullable=False
    )

    # Path to the image file on disk (relative to IMAGE_STORAGE_DIR)
    # e.g., "b3f1a2c4-..../kitchen.jpg"
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)

    # Detected room type — e.g. "kitchen", "bedroom". Nullable if detection failed.
    room_type: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # --- Phase 9 SEO + ordering ----------------------------------------------
    # alt_text is generated by the same VLM call that detects amenities and is
    # user-editable in the review UI. caption is user-only. is_primary marks
    # the hero image used by Phase 12 listing cards + JSON-LD; display_order
    # controls gallery ordering.
    alt_text: Mapped[str | None] = mapped_column(String(500), nullable=True)
    caption: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    display_order: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)

    # --- Relationships --------------------------------------------------------
    property: Mapped["Property"] = relationship("Property", back_populates="images")
    amenities: Mapped[list["DetectedAmenity"]] = relationship(
        "DetectedAmenity", back_populates="image", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<PropertyImage id={self.id!r} property_id={self.property_id!r}>"


class DetectedAmenity(Base):
    """
    A single amenity detection result for one image within a property.

    One row per (image, amenity_name) pair. The is_present flag is the primary
    signal; confidence is an optional score derived from the VLM output or
    heuristics (not all models provide this explicitly).
    """

    __tablename__ = "detected_amenities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Which property and image this detection belongs to
    property_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("properties.id"), nullable=False
    )
    image_id: Mapped[str] = mapped_column(String(36), ForeignKey("images.id"), nullable=False)

    # e.g. "refrigerator", "bathtub", "wifi" — taken directly from amenity_schema
    amenity_name: Mapped[str] = mapped_column(String(200), nullable=False)

    # The room type this amenity was detected in (e.g., "kitchen")
    room_type: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Confidence score 0.0–1.0. Currently derived from whether the VLM said True/False.
    # Set to 1.0 if present=True, 0.0 if present=False, until we have proper scoring.
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # The core detection result: True = amenity is visible in the image
    is_present: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # --- Relationships --------------------------------------------------------
    property: Mapped["Property"] = relationship("Property", back_populates="amenities")
    image: Mapped["PropertyImage"] = relationship("PropertyImage", back_populates="amenities")

    def __repr__(self) -> str:
        return (
            f"<DetectedAmenity amenity={self.amenity_name!r} "
            f"present={self.is_present} image_id={self.image_id!r}>"
        )
