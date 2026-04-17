"""
PropertyAmenitySystem — high-level pipeline orchestrator.

This class ties together the three main components:
  1. AmenityDetector  — asks the VLM which amenities are in an image
  2. AmenityDataManager — saves/retrieves results from the database
  3. Image file storage — saves uploaded images to the local file store

Usage:
  This class is used directly by the FastAPI upload endpoint:

    system = PropertyAmenitySystem(vlm_client, db_session, amenity_schema, storage_dir)
    result = system.process_upload(
        images=[pil_image1, pil_image2],
        filenames=["kitchen.jpg", "bedroom.jpg"],
        property_name="Frankfurt House 1",
        extra_info="Near city centre",
    )

Design note:
  PropertyAmenitySystem is intentionally lightweight — it just calls the right
  methods in the right order. All the real logic lives in AmenityDetector (VLM prompts)
  and AmenityDataManager (SQL queries). This makes each component testable in isolation.
"""

import logging
from pathlib import Path

from PIL.Image import Image
from sqlalchemy.orm import Session

from core.amenity_data_manager import AmenityDataManager
from core.amenity_detector import AmenityDetector
from core.amenity_schema import load_amenity_schema
from db.models import Property
from models.base import VLMClient


class PropertyAmenitySystem:
    """
    Orchestrates the full amenity detection pipeline for a property upload.

    One instance per request is fine — AmenityDetector is stateless once initialised,
    and AmenityDataManager takes a per-request DB session.
    """

    def __init__(
        self,
        vlm_client: VLMClient,
        db: Session,
        image_storage_dir: str | Path,
        amenity_schema: dict[str, list[str]] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise the system with all required dependencies.

        Args:
            vlm_client:         VLMClient instance (Ollama or Gemini).
            db:                 Active SQLAlchemy Session (per-request).
            image_storage_dir:  Directory where uploaded images will be saved.
                                Created automatically if it doesn't exist.
            amenity_schema:     Dict mapping room type → list of amenity names.
                                If None, loads the built-in default schema.
            logger:             Optional logger. Defaults to module logger.
        """
        self.logger = logger or logging.getLogger(__name__)

        # Load the amenity schema (what rooms and amenities to look for)
        self.amenity_schema: dict[str, list[str]] = (
            amenity_schema if amenity_schema is not None else load_amenity_schema()
        )

        self.detector = AmenityDetector(
            vlm_client=vlm_client,
            amenity_schema=self.amenity_schema,
            logger=self.logger,
        )
        self.data_manager = AmenityDataManager(db=db, logger=self.logger)

        # Ensure the image storage directory exists
        self.storage_dir = Path(image_storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def process_upload(
        self,
        images: list[Image],
        filenames: list[str],
        property_name: str,
        model_name: str,
        extra_info: str | None = None,
    ) -> Property:
        """
        Run the full pipeline for a new property upload.

        Steps:
          1. Create a Property record in the database.
          2. For each image:
             a. Save the image file to the local file store.
             b. Create a PropertyImage record.
             c. Run amenity detection.
             d. Save DetectedAmenity records.
          3. Generate a property description from the last image's detection results.
          4. Update the Property record with the description.
          5. Commit everything to the database.

        Args:
            images:        List of PIL Image objects (one per uploaded file).
            filenames:     Original filenames (same order as images).
            property_name: Human-readable label for this property.
            model_name:    Name of the VLM used (stored for auditing).
            extra_info:    Optional free-text notes from the user.

        Returns:
            The fully populated Property ORM object (committed to the database).
        """
        self.logger.info(
            "Processing upload: property=%r, model=%s, images=%d",
            property_name,
            model_name,
            len(images),
        )

        # Step 1: Create the Property record (gets a UUID assigned via flush)
        prop = self.data_manager.create_property(
            name=property_name,
            model_used=model_name,
            extra_info=extra_info,
        )

        # Create a sub-folder per property so files don't collide
        property_dir = self.storage_dir / prop.id
        property_dir.mkdir(parents=True, exist_ok=True)

        last_flat_amenities: dict[str, bool] = {}
        last_image: Image | None = None

        # Step 2: Process each image
        for pil_image, filename in zip(images, filenames, strict=True):
            # Step 2a: Save image file to disk
            file_path = property_dir / filename
            pil_image.save(file_path)

            # Store path relative to storage root so it's portable (not an absolute path)
            relative_path = str(Path(prop.id) / filename)

            # Step 2b: Create PropertyImage record (room_type filled in after detection)
            img_record = self.data_manager.save_image(
                property_id=prop.id,
                file_path=relative_path,
                room_type=None,  # Will be detected below
            )

            # Step 2c: Run amenity detection
            amenities_by_room, flat_amenities = self.detector.detect_from_image(pil_image)
            self.logger.info(
                "Detected %d present amenities in %s",
                sum(1 for v in flat_amenities.values() if v),
                filename,
            )

            # Determine room type — take the room with the most detected amenities
            detected_room = _infer_room_type(amenities_by_room)
            img_record.room_type = detected_room

            # Step 2d: Save detection results to the database
            self.data_manager.save_amenities(
                property_id=prop.id,
                image_id=img_record.id,
                amenities=flat_amenities,
                room_type=detected_room,
            )

            # Track for description generation
            last_flat_amenities = flat_amenities
            last_image = pil_image

        # Step 3: Generate description from the last processed image
        if last_image is not None:
            description = self.detector.generate_description(last_image, last_flat_amenities)
        else:
            description = "No images were processed."

        # Step 4: Update Property description
        self.data_manager.update_property_description(prop.id, description)

        # Step 5: Commit the entire transaction
        self.data_manager.db.commit()
        self.data_manager.db.refresh(prop)

        self.logger.info("Upload complete: property_id=%s", prop.id)
        return prop


def _infer_room_type(amenities_by_room: dict[str, dict[str, bool]]) -> str | None:
    """
    Determine the most likely room type based on which room had the most detected amenities.

    This is a simple heuristic: the room type whose amenities had the most True values
    is declared the winner. Returns None if no amenities were detected at all.

    Args:
        amenities_by_room: {room_type: {amenity_name: bool}}

    Returns:
        The room type string with the most detected amenities, or None.
    """
    best_room: str | None = None
    best_count = 0

    for room_type, amenities in amenities_by_room.items():
        count = sum(1 for present in amenities.values() if present)
        if count > best_count:
            best_count = count
            best_room = room_type

    return best_room
