"""
PropertyAmenitySystem — high-level pipeline orchestrator.

This class ties together the three main components:
  1. AmenityDetector  — asks the VLM which amenities are in an image
  2. AmenityDataManager — saves/retrieves results from the database
  3. Image file storage — saves uploaded images to the local file store

Usage:
  This class is used directly by the FastAPI property/image endpoints:

    system = PropertyAmenitySystem(vlm_client, db_session, amenity_schema, storage_dir)
    prop = system.create_property_shell(property_name="Frankfurt House 1", model_name="...")
    image = system.process_one_image(property_id=prop.id, image=pil_image1, filename="kitchen.jpg")

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
from db.models import Property, PropertyImage
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
            vlm_client:         VLMClient instance.
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

    def create_property_shell(
        self,
        property_name: str,
        model_name: str,
        extra_info: str | None = None,
    ) -> Property:
        """
        Create a Property row without running detection.

        Phase 5 uses this before uploading images one-by-one, which lets the UI
        show progress after each image instead of waiting for a whole batch.
        """
        prop = self.data_manager.create_property(
            name=property_name,
            model_used=model_name,
            extra_info=extra_info,
        )
        self.data_manager.db.commit()
        self.data_manager.db.refresh(prop)
        (self.storage_dir / prop.id).mkdir(parents=True, exist_ok=True)
        return prop

    def process_one_image(
        self,
        property_id: str,
        image: Image,
        filename: str,
    ) -> PropertyImage:
        """
        Save and process one image for an existing property.

        Args:
            property_id: Existing Property UUID.
            image:       PIL image to persist and analyse.
            filename:    Original filename from the upload.

        Returns:
            The created PropertyImage ORM record with amenities populated.

        Raises:
            ValueError: If the property does not exist.
        """
        prop = self.data_manager.get_property(property_id)
        if prop is None:
            raise ValueError(f"Property not found: {property_id}")

        property_dir = self.storage_dir / property_id
        property_dir.mkdir(parents=True, exist_ok=True)

        file_path = property_dir / filename
        image.save(file_path)
        relative_path = str(Path(property_id) / filename)

        img_record = self.data_manager.save_image(
            property_id=property_id,
            file_path=relative_path,
            room_type=None,
        )

        amenities_by_room, flat_amenities, flat_confidences = self.detector.detect_from_image(image)
        detected_room = _infer_room_type(amenities_by_room)
        img_record.room_type = detected_room

        self.data_manager.save_amenities(
            property_id=property_id,
            image_id=img_record.id,
            amenities=flat_amenities,
            room_type=detected_room,
            confidences=flat_confidences,
        )
        self.data_manager.db.commit()
        self.data_manager.db.refresh(img_record)
        return img_record

    def generate_description_from_amenities(
        self,
        amenities: list[dict[str, object]],
        property_name: str,
        extra_info: str | None = None,
        num_rooms: int | None = None,
        has_kitchen: bool | None = None,
        has_balcony: bool | None = None,
        has_living_room: bool | None = None,
        hints: dict[str, bool] | None = None,
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
            num_rooms:     Optional user hint for the number of rooms.
            has_kitchen:   Optional user hint for kitchen presence.
            has_balcony:   Optional user hint for balcony presence.
            has_living_room: Optional user hint for living room presence.
            hints:         Optional expanded amenity hint dict.

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
        hints_text = _format_sidebar_hints(
            num_rooms=num_rooms,
            has_kitchen=has_kitchen,
            has_balcony=has_balcony,
            has_living_room=has_living_room,
            hints=hints,
        )
        context_parts = []
        if extra_info:
            context_parts.append(f"Additional context: {extra_info}.")
        if hints_text:
            context_parts.append(f"User-provided property hints: {hints_text}.")
        context = " " + " ".join(context_parts) if context_parts else ""
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
    if len(amenities_by_room) == 1:
        return next(iter(amenities_by_room))

    best_room: str | None = None
    best_count = 0

    for room_type, amenities in amenities_by_room.items():
        count = sum(1 for present in amenities.values() if present)
        if count > best_count:
            best_count = count
            best_room = room_type

    return best_room


def _format_sidebar_hints(
    num_rooms: int | None = None,
    has_kitchen: bool | None = None,
    has_balcony: bool | None = None,
    has_living_room: bool | None = None,
    hints: dict[str, bool] | None = None,
) -> str:
    """
    Convert optional UI hints into concise prompt text.

    Unspecified values are omitted entirely so the model does not treat missing
    UI input as a negative signal. The expanded hints dict is additive and wins
    over the legacy three flags when both provide the same concept.
    """
    merged: dict[str, bool] = {}
    legacy = {
        "kitchen": has_kitchen,
        "balcony": has_balcony,
        "living room": has_living_room,
    }
    for label, value in legacy.items():
        if value is not None:
            merged[label] = value
    if hints:
        for key, value in hints.items():
            if value is not None:
                merged[key.replace("_", " ")] = bool(value)

    parts: list[str] = []
    if num_rooms is not None and num_rooms > 0:
        noun = "room" if num_rooms == 1 else "rooms"
        parts.append(f"{num_rooms} {noun}")
    for label, value in merged.items():
        parts.append(f"has a {label}" if value else f"no {label}")
    return ", ".join(parts)
