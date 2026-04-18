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

from PIL import Image as PILImage
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
            # detect_from_image returns three dicts:
            #   amenities_by_room — results organised by room type (for room inference)
            #   flat_amenities    — simple bool dict for all amenities
            #   flat_confidences  — model confidence per amenity (0.0–1.0, Phase 4)
            amenities_by_room, flat_amenities, flat_confidences = self.detector.detect_from_image(
                pil_image
            )
            self.logger.info(
                "Detected %d present amenities in %s",
                sum(1 for v in flat_amenities.values() if v),
                filename,
            )

            # Determine room type — take the room with the most detected amenities
            detected_room = _infer_room_type(amenities_by_room)
            img_record.room_type = detected_room

            # Step 2d: Save detection results to the database (with confidence scores)
            self.data_manager.save_amenities(
                property_id=prop.id,
                image_id=img_record.id,
                amenities=flat_amenities,
                room_type=detected_room,
                confidences=flat_confidences,
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

    def generate_description_from_amenities(
        self,
        amenities: list[dict[str, object]],
        property_name: str,
        extra_info: str | None = None,
    ) -> str:
        """
        Generate a property description from a user-edited amenity list.

        Called when the user has reviewed the detected amenities, made edits,
        and clicks 'Confirm & Generate Description'. Only amenities where
        is_present=True are included in the prompt.

        Args:
            amenities:     List of dicts with keys: amenity_name, room_type, is_present.
            property_name: Used to personalise the description.
            extra_info:    Optional context (e.g. "2-bed flat in Frankfurt").

        Returns:
            A natural-language description string from the VLM.
        """
        # Group confirmed amenities by room
        by_room: dict[str, list[str]] = {}
        for item in amenities:
            if item.get("is_present"):
                room = str(item.get("room_type", "unknown"))
                name = str(item.get("amenity_name", ""))
                if name:
                    by_room.setdefault(room, []).append(name)

        if not by_room:
            return "No amenities were confirmed as present."

        # Build a structured prompt so the VLM has clear input
        room_lines = "\n".join(
            f"  - {room.title()}: {', '.join(items)}" for room, items in by_room.items()
        )
        context = f" Additional context: {extra_info}." if extra_info else ""
        prompt = (
            f"You are writing a property listing description for '{property_name}'.{context}\n"
            f"The following amenities have been confirmed as present:\n{room_lines}\n\n"
            "Write a warm, professional 3-4 sentence description of the property that "
            "highlights these amenities. Do not invent amenities not listed above."
        )

        # Use a blank 1x1 white image as a placeholder — this endpoint uses text-only context.
        # Most VLMs accept an image; we pass a minimal one to keep the interface consistent.
        placeholder = PILImage.new("RGB", (1, 1), color=(255, 255, 255))

        try:
            response = self.detector.client.generate(image=placeholder, prompt=prompt)
            return str(response.raw_text).strip()
        except Exception as e:
            self.logger.error("Description generation failed: %s", e)
            raise RuntimeError(f"Description generation failed: {e}") from e


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
