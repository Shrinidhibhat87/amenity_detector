"""
Properties router — handles all /api/v1/properties/* endpoints.

Endpoints:
  POST   /api/v1/properties/{id}/describe   Regenerate description from edited amenity list
  GET    /api/v1/properties/                List all properties (paginated)
  GET    /api/v1/properties/search          Filter by required amenities
  GET    /api/v1/properties/{id}            Full property details
  DELETE /api/v1/properties/{id}            Remove a property

Design notes:
  - Routes are thin: they validate input, call the service layer, return responses.
  - All DB work goes through AmenityDataManager (in core/amenity_data_manager.py).
  - All VLM + file work goes through PropertyAmenitySystem (in core/amenity_system.py).
  - Pydantic response schemas are in api/schemas.py.
"""

import io
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from PIL import Image
from sqlalchemy.orm import Session

from api.dependencies import get_image_storage_dir, get_model_registry
from api.schemas import (
    DescribeRequest,
    DescribeResponse,
    DetectedAmenityResponse,
    ImageDetectionResponse,
    PropertyCreateRequest,
    PropertyCreateResponse,
    PropertyDetailResponse,
    PropertyImageResponse,
    PropertySummaryResponse,
    PropertyUpdateRequest,
)
from core.amenity_data_manager import AmenityDataManager
from core.amenity_system import PropertyAmenitySystem
from db.session import get_db
from models.registry import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/properties", tags=["properties"])

# Allowed image MIME types — reject anything else at the boundary
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/", response_model=PropertyCreateResponse, status_code=201)
def create_property(
    body: PropertyCreateRequest,
    db: Session = Depends(get_db),
    registry: ModelRegistry = Depends(get_model_registry),
    storage_dir: Path = Depends(get_image_storage_dir),
) -> PropertyCreateResponse:
    """
    Create an empty property shell without running VLM inference.

    The Phase 5 UI calls this once, then uploads images individually to
    /api/v1/properties/{id}/images so progress can update per image.
    """
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Property name is required.")

    try:
        vlm_client = registry.get(body.model_name)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Pull the optional Phase 9 listing fields off the request body. We use
    # ``model_dump(exclude_unset=True)`` so a missing field stays NULL rather
    # than being overwritten with the schema default.
    metadata = body.model_dump(
        exclude={"name", "model_name", "extra_info"},
        exclude_unset=True,
    )

    try:
        system = PropertyAmenitySystem(
            vlm_client=vlm_client,
            db=db,
            image_storage_dir=storage_dir,
        )
        prop = system.create_property_shell(
            property_name=name,
            model_name=body.model_name,
            extra_info=body.extra_info.strip() if body.extra_info else None,
            listing_metadata=metadata or None,
        )
    except Exception as e:
        logger.exception("Property shell creation failed for property '%s'", name)
        raise HTTPException(status_code=500, detail=f"Property creation failed: {e}") from e

    return PropertyCreateResponse(
        property_id=prop.id,
        message=f"Property '{prop.name}' created successfully.",
        property=_build_detail_response(prop),
    )


@router.patch("/{property_id}", response_model=PropertyDetailResponse)
def patch_property(
    property_id: str,
    body: PropertyUpdateRequest,
    db: Session = Depends(get_db),
) -> PropertyDetailResponse:
    """
    Update one or more listing-metadata fields on an existing property.

    Only fields actually present in the request body are written. Sending
    ``null`` for a field clears it; omitting the field leaves it untouched.

    The ``slug`` field is intentionally not patchable — keeping public URLs
    stable is the whole point of having a slug. The Pydantic schema rejects
    unknown keys with HTTP 422.
    """
    fields = body.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields provided to update.")

    manager = AmenityDataManager(db)
    updated = manager.update_property(property_id, fields)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Property '{property_id}' not found.")
    db.commit()
    db.refresh(updated)
    return _build_detail_response(updated)


@router.post("/{property_id}/images", response_model=ImageDetectionResponse, status_code=201)
def upload_property_image(
    property_id: str,
    file: UploadFile = File(..., description="One property image"),
    model_name: str = Form(..., description="VLM to use for detection"),
    db: Session = Depends(get_db),
    registry: ModelRegistry = Depends(get_model_registry),
    storage_dir: Path = Depends(get_image_storage_dir),
) -> ImageDetectionResponse:
    """
    Upload and process one image for an existing property.

    This endpoint is intentionally single-image so Gradio can yield progress
    after every completed VLM call.
    """
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File '{file.filename}' has unsupported type '{file.content_type}'. "
                f"Allowed: {sorted(_ALLOWED_CONTENT_TYPES)}"
            ),
        )

    try:
        vlm_client = registry.get(model_name)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    raw = file.file.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image '{file.filename}': {e}",
        ) from e

    try:
        system = PropertyAmenitySystem(
            vlm_client=vlm_client,
            db=db,
            image_storage_dir=storage_dir,
        )
        img_record = system.process_one_image(
            property_id=property_id,
            image=pil_image,
            filename=file.filename or "image.jpg",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception("Single image processing failed for property '%s'", property_id)
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}") from e

    return ImageDetectionResponse(
        property_id=property_id,
        image=_build_image_response(img_record),
    )


@router.get("/", response_model=list[PropertySummaryResponse])
def list_properties(
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(20, ge=1, le=100, description="Max items to return"),
    db: Session = Depends(get_db),
) -> list[PropertySummaryResponse]:
    """
    List all properties (newest first), with pagination.

    Args:
        offset: Skip this many results (useful for page 2+).
        limit:  Return at most this many results per page.
        db:     Database session.

    Returns:
        List of PropertySummaryResponse objects.
    """
    manager = AmenityDataManager(db)
    properties = manager.list_properties(offset=offset, limit=limit)
    return [
        PropertySummaryResponse(
            **{k: v for k, v in prop.__dict__.items() if not k.startswith("_")},
            image_count=len(prop.images),
            first_image_id=prop.images[0].id if prop.images else None,
        )
        for prop in properties
    ]


@router.get("/search", response_model=list[PropertySummaryResponse])
def search_properties(
    amenities: str = Query(
        ...,
        description="Comma-separated list of required amenities (e.g. 'wifi,pool,gym')",
    ),
    db: Session = Depends(get_db),
) -> list[PropertySummaryResponse]:
    """
    Search for properties that have ALL the specified amenities detected.

    Args:
        amenities: Comma-separated amenity names (case-sensitive, must match schema).
        db:        Database session.

    Returns:
        List of matching properties as summary objects.

    Example:
        GET /api/v1/properties/search?amenities=refrigerator,oven
    """
    amenity_list = [a.strip() for a in amenities.split(",") if a.strip()]
    if not amenity_list:
        raise HTTPException(status_code=400, detail="No amenity names provided.")

    manager = AmenityDataManager(db)
    properties = manager.search_properties_by_amenities(amenity_list)
    return [
        PropertySummaryResponse(
            **{k: v for k, v in prop.__dict__.items() if not k.startswith("_")},
            image_count=len(prop.images),
            first_image_id=prop.images[0].id if prop.images else None,
        )
        for prop in properties
    ]


@router.post("/{property_id}/describe", response_model=DescribeResponse)
def regenerate_description(
    property_id: str,
    body: DescribeRequest,
    db: Session = Depends(get_db),
    registry: ModelRegistry = Depends(get_model_registry),
    storage_dir: Path = Depends(get_image_storage_dir),
) -> DescribeResponse:
    """
    Regenerate a property description from the user's edited amenity list.

    Called after the user reviews and edits the detected amenities in the UI.
    The VLM is asked to write a fresh description based only on the amenities
    the user confirmed as present.

    Args:
        property_id: UUID of the existing property (must exist in the DB).
        body:        Edited amenity list + the model to use for generation.

    Returns:
        DescribeResponse with the new description text.

    Raises:
        400: If the model is unknown.
        404: If the property does not exist.
        500: If VLM inference fails.
    """
    from db.models import Property as PropertyModel

    prop = db.get(PropertyModel, property_id)
    if prop is None:
        raise HTTPException(status_code=404, detail=f"Property '{property_id}' not found.")

    try:
        vlm_client = registry.get(body.model_name)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        system = PropertyAmenitySystem(
            vlm_client=vlm_client,
            db=db,
            # storage_dir is required by PropertyAmenitySystem.__init__ (it calls mkdir).
            # No images are written in this describe-only path.
            image_storage_dir=storage_dir,
        )
        amenities_as_dicts = [
            {"amenity_name": a.amenity_name, "room_type": a.room_type, "is_present": a.is_present}
            for a in body.amenities
        ]
        description = system.generate_description_from_amenities(
            amenities=amenities_as_dicts,
            property_name=prop.name,
            extra_info=prop.extra_info,
            num_rooms=body.num_rooms,
            has_kitchen=body.has_kitchen,
            has_balcony=body.has_balcony,
            has_living_room=body.has_living_room,
            hints=body.hints,
        )
    except Exception as e:
        logger.exception("Description regeneration failed for property '%s'", property_id)
        raise HTTPException(status_code=500, detail=f"Description generation failed: {e}") from e

    return DescribeResponse(description=description)


@router.get("/{property_id}", response_model=PropertyDetailResponse)
def get_property(
    property_id: str,
    db: Session = Depends(get_db),
) -> PropertyDetailResponse:
    """
    Get full details for a single property, including all images and amenities.

    Args:
        property_id: UUID of the property.
        db:          Database session.

    Returns:
        PropertyDetailResponse with images and nested amenity results.

    Raises:
        404: If no property with the given ID exists.
    """
    manager = AmenityDataManager(db)
    prop = manager.get_property(property_id)
    if prop is None:
        raise HTTPException(status_code=404, detail=f"Property '{property_id}' not found.")
    return _build_detail_response(prop)


@router.delete("/{property_id}", status_code=204)
def delete_property(
    property_id: str,
    db: Session = Depends(get_db),
) -> None:
    """
    Delete a property and all its images and detected amenities.

    Note: This does NOT delete the image files from disk — only the DB records.
    Image cleanup from the file store is a future enhancement (Phase 4).

    Args:
        property_id: UUID of the property to delete.
        db:          Database session.

    Raises:
        404: If no property with the given ID exists.
    """
    manager = AmenityDataManager(db)
    deleted = manager.delete_property(property_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Property '{property_id}' not found.")
    db.commit()


# ── Helper ───────────────────────────────────────────────────────────────────


_PROPERTY_METADATA_FIELDS = (
    "slug",
    "listing_type",
    "price",
    "currency",
    "price_period",
    "num_bedrooms",
    "num_bathrooms",
    "area_sqm",
    "property_type",
    "furnishing",
    "available_from",
    "locality",
    "postal_code",
    "country_code",
    "latitude",
    "longitude",
    "owner_email",
)


def _build_detail_response(prop: object) -> PropertyDetailResponse:
    """
    Convert a Property ORM object into a PropertyDetailResponse Pydantic model.

    We do this manually (rather than relying on from_attributes=True alone) because
    we need to map the ORM's `images` relationship into the nested response structure,
    including the amenities nested under each image.

    Args:
        prop: A Property ORM instance with `images` and `images[*].amenities` loaded.

    Returns:
        A fully populated PropertyDetailResponse.
    """
    from db.models import Property as PropertyModel  # avoid circular import at module level

    assert isinstance(prop, PropertyModel)

    image_responses = [_build_image_response(img) for img in prop.images]

    metadata = {field: getattr(prop, field) for field in _PROPERTY_METADATA_FIELDS}

    return PropertyDetailResponse(
        id=prop.id,
        name=prop.name,
        description=prop.description,
        model_used=prop.model_used,
        extra_info=prop.extra_info,
        created_at=prop.created_at,
        images=image_responses,
        **metadata,
    )


def _build_image_response(img: object) -> PropertyImageResponse:
    """Convert a PropertyImage ORM object into an API response schema."""
    from db.models import PropertyImage as PropertyImageModel

    assert isinstance(img, PropertyImageModel)

    return PropertyImageResponse(
        id=img.id,
        file_path=img.file_path,
        room_type=img.room_type,
        amenities=[
            DetectedAmenityResponse(
                id=a.id,
                amenity_name=a.amenity_name,
                room_type=a.room_type,
                is_present=a.is_present,
                confidence=a.confidence,
            )
            for a in img.amenities
        ],
        alt_text=img.alt_text,
        caption=img.caption,
        is_primary=img.is_primary,
        display_order=img.display_order,
    )
