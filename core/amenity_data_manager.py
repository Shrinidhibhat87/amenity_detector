"""
AmenityDataManager — service layer for all database read/write operations.

Why a separate service layer instead of putting DB code in the router?
  - Testability: unit tests can instantiate AmenityDataManager with a test session
    without needing to spin up the full FastAPI app.
  - Reuse: the same methods can be called from scripts, the API (routers),
    and future scripts without duplicating SQL logic.
  - Clarity: routers stay thin (HTTP concerns only); this class owns DB concerns.

Usage pattern in FastAPI:
    @router.post("/properties")
    def upload(db: Session = Depends(get_db)):
        manager = AmenityDataManager(db)
        prop = manager.create_property(name="My House", model_used="openai/gpt-4o-mini")
        ...
"""

import logging
from typing import Any

from sqlalchemy import and_, exists, select
from sqlalchemy.orm import Session

from db.models import DetectedAmenity, Property, PropertyImage


class AmenityDataManager:
    """
    Handles all database interactions for the amenity detection pipeline.

    Each instance is tied to a single SQLAlchemy Session (one per HTTP request).
    The session is passed in from the outside (injected) rather than created here,
    which makes the class easy to test — just pass a test session with a test DB.
    """

    def __init__(self, db: Session, logger: logging.Logger | None = None) -> None:
        """
        Initialise with an active database session.

        Args:
            db:     An open SQLAlchemy Session. The caller is responsible for
                    committing and closing it (FastAPI's get_db() handles this).
            logger: Optional logger. Defaults to the module logger.
        """
        self.db = db
        self.logger = logger or logging.getLogger(__name__)

    # ── CREATE ──────────────────────────────────────────────────────────────────

    def create_property(
        self,
        name: str,
        model_used: str | None = None,
        extra_info: str | None = None,
        listing_metadata: dict[str, Any] | None = None,
    ) -> Property:
        """
        Insert a new Property row and return it.

        The description is not set here — it's filled in after the VLM runs.

        Args:
            name:             Human-readable label for this property.
            model_used:       The VLM that will process this property
                              (e.g., "openai/gpt-4o-mini").
            extra_info:       Free-text notes from the user.
            listing_metadata: Optional dict of Phase 9 listing fields
                              (price, num_bedrooms, locality, ...). Only keys
                              present in the dict are applied; ``None`` values
                              are written through (so callers can clear fields).

        Returns:
            The newly created Property ORM object (not yet committed).
        """
        prop = Property(name=name, model_used=model_used, extra_info=extra_info)
        if listing_metadata:
            for key, value in listing_metadata.items():
                setattr(prop, key, value)
        self.db.add(prop)
        self.db.flush()  # Assign the UUID without a full commit — we still need to add images
        self.logger.info("Created property id=%s name=%r", prop.id, prop.name)
        return prop

    def update_property(
        self,
        property_id: str,
        fields: dict[str, Any],
    ) -> Property | None:
        """
        Patch an existing property with the supplied fields and return it.

        Only keys present in ``fields`` are written; this gives the API caller
        explicit control between "leave alone" (omit the key) and "clear"
        (pass ``None``). The caller is responsible for committing.

        Args:
            property_id: UUID of the property to update.
            fields:      Dict of column-name → new value.

        Returns:
            The updated Property, or ``None`` if no row matched.
        """
        prop = self.db.get(Property, property_id)
        if prop is None:
            return None
        for key, value in fields.items():
            setattr(prop, key, value)
        self.db.flush()
        self.logger.info("Patched property id=%s fields=%s", property_id, sorted(fields.keys()))
        return prop

    def save_image(
        self,
        property_id: str,
        file_path: str,
        room_type: str | None = None,
    ) -> PropertyImage:
        """
        Insert a new PropertyImage row and return it.

        Args:
            property_id: The UUID of the parent Property.
            file_path:   Path to the stored image file (relative to IMAGE_STORAGE_DIR).
            room_type:   Detected room type, e.g. "kitchen". Can be set later via
                         update_image_room_type() once the VLM has run.

        Returns:
            The newly created PropertyImage ORM object (not yet committed).
        """
        img = PropertyImage(
            property_id=property_id,
            file_path=file_path,
            room_type=room_type,
        )
        self.db.add(img)
        self.db.flush()  # Get the image UUID before saving its amenities
        self.logger.info("Saved image id=%s for property_id=%s", img.id, property_id)
        return img

    def save_amenities(
        self,
        property_id: str,
        image_id: str,
        amenities: dict[str, bool],
        room_type: str | None = None,
        confidences: dict[str, float] | None = None,
    ) -> list[DetectedAmenity]:
        """
        Bulk-insert DetectedAmenity rows for a single image.

        One row is created for every amenity in the dict, regardless of whether
        it was detected or not — storing False lets us query "not detected" too.

        Confidence values come from the VLM's structured JSON output (Phase 4 improvement).
        If no confidences dict is provided the method falls back to 1.0/0.0 heuristics.

        Args:
            property_id: UUID of the parent Property.
            image_id:    UUID of the parent PropertyImage.
            amenities:   Dict of amenity_name → bool from AmenityDetector.
            room_type:   The room type this image was classified as.
            confidences: Optional dict of amenity_name → float (0.0–1.0).
                         When provided, these values are stored directly.
                         When absent, 1.0 is used for present amenities and 0.0 for absent.

        Returns:
            List of DetectedAmenity ORM objects (not yet committed).
        """
        records: list[DetectedAmenity] = []
        for amenity_name, is_present in amenities.items():
            # Use model-supplied confidence if available; fall back to 1.0 / 0.0
            if confidences is not None and amenity_name in confidences:
                confidence = confidences[amenity_name]
            else:
                confidence = 1.0 if is_present else 0.0

            record = DetectedAmenity(
                property_id=property_id,
                image_id=image_id,
                amenity_name=amenity_name,
                room_type=room_type,
                is_present=is_present,
                confidence=confidence,
            )
            self.db.add(record)
            records.append(record)

        self.logger.info("Saved %d amenity records for image_id=%s", len(records), image_id)
        return records

    def update_image(
        self,
        image_id: str,
        fields: dict[str, Any],
    ) -> PropertyImage | None:
        """
        Patch a ``PropertyImage`` with the supplied fields and return it.

        When ``is_primary=True`` is in ``fields``, the flag is cleared on every
        other image belonging to the same property in the same transaction —
        the public listing page picks exactly one hero image.

        Args:
            image_id: UUID of the image to update.
            fields:   Dict of column-name → new value.

        Returns:
            The updated PropertyImage, or ``None`` when no row matches.
        """
        img = self.db.get(PropertyImage, image_id)
        if img is None:
            return None

        if fields.get("is_primary") is True:
            siblings = (
                self.db.query(PropertyImage)
                .filter(
                    PropertyImage.property_id == img.property_id,
                    PropertyImage.id != image_id,
                    PropertyImage.is_primary.is_(True),
                )
                .all()
            )
            for sibling in siblings:
                sibling.is_primary = False

        for key, value in fields.items():
            setattr(img, key, value)
        self.db.flush()
        self.logger.info("Patched image id=%s fields=%s", image_id, sorted(fields.keys()))
        return img

    def update_property_description(self, property_id: str, description: str) -> Property:
        """
        Set the VLM-generated description on an existing Property.

        Called after all images have been processed and the description has been
        synthesised from the detection results.

        Args:
            property_id:  UUID of the Property to update.
            description:  The natural-language description to store.

        Returns:
            The updated Property ORM object.

        Raises:
            ValueError: If no property with the given ID exists.
        """
        prop = self.db.get(Property, property_id)
        if prop is None:
            raise ValueError(f"Property not found: {property_id}")
        prop.description = description
        self.logger.info("Updated description for property_id=%s", property_id)
        return prop

    # ── READ ────────────────────────────────────────────────────────────────────

    def get_property(self, property_id: str) -> Property | None:
        """
        Fetch a single Property with all its images and amenities pre-loaded.

        SQLAlchemy lazily loads relationships by default, but we use selectinload
        (via the relationship cascade setting) so the images and amenities are
        available without extra queries once the property is returned.

        Args:
            property_id: UUID of the property to fetch.

        Returns:
            The Property ORM object, or None if not found.
        """
        return self.db.get(Property, property_id)

    def get_property_by_slug(self, slug: str) -> Property | None:
        """
        Fetch a single Property by its URL-safe slug.

        Slugs are unique (enforced by the unique index added in
        0002_phase9_listing_metadata) and immutable, so this lookup is the
        canonical entry point for public listing URLs.

        Args:
            slug: Slug value to match exactly.

        Returns:
            The Property ORM object, or None if no row has that slug.
        """
        stmt = select(Property).where(Property.slug == slug).limit(1)
        return self.db.scalars(stmt).one_or_none()

    def list_properties(self, offset: int = 0, limit: int = 20) -> list[Property]:
        """
        Return a paginated list of all properties, newest first.

        Args:
            offset: Number of rows to skip (for pagination). Default: 0.
            limit:  Maximum number of rows to return. Default: 20.

        Returns:
            List of Property ORM objects.
        """
        stmt = select(Property).order_by(Property.created_at.desc()).offset(offset).limit(limit)
        return list(self.db.scalars(stmt))

    def search_properties_by_amenities(self, amenity_names: list[str]) -> list[Property]:
        """
        Return properties where ALL requested amenities are present in at least one image.

        How the query works:
          For each amenity name in the list, we check that there EXISTS at least one
          DetectedAmenity row for that property where amenity_name matches and is_present=True.
          We AND these existence checks together so ALL amenities must be present.

        Example: search_properties_by_amenities(["wifi", "pool"])
          Returns properties that have BOTH wifi and pool detected.

        Args:
            amenity_names: List of amenity names to search for (case-sensitive).

        Returns:
            List of matching Property ORM objects, newest first.
        """
        if not amenity_names:
            return self.list_properties()

        # Build one EXISTS subquery per amenity, then AND them all
        conditions = [
            exists(
                select(DetectedAmenity.id).where(
                    and_(
                        DetectedAmenity.property_id == Property.id,
                        DetectedAmenity.amenity_name == name,
                        DetectedAmenity.is_present.is_(True),
                    )
                )
            )
            for name in amenity_names
        ]

        stmt = select(Property).where(and_(*conditions)).order_by(Property.created_at.desc())
        return list(self.db.scalars(stmt))

    # ── DELETE ──────────────────────────────────────────────────────────────────

    def delete_property(self, property_id: str) -> bool:
        """
        Delete a property and all its related images and amenities.

        The cascade="all, delete-orphan" on the ORM relationships ensures that
        SQLAlchemy automatically deletes the related PropertyImage and
        DetectedAmenity rows when the Property is deleted.

        Args:
            property_id: UUID of the property to delete.

        Returns:
            True if the property was found and deleted, False if not found.
        """
        prop = self.db.get(Property, property_id)
        if prop is None:
            self.logger.warning("Attempted to delete non-existent property: %s", property_id)
            return False

        self.db.delete(prop)
        self.logger.info("Deleted property id=%s", property_id)
        return True
