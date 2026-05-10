"""
Images router: serves stored property image bytes by image id.

The Browse Properties UI uses this endpoint for lazy thumbnails. File paths
are resolved under the configured storage directory before serving so tampered
database rows cannot escape the image store.
"""

import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from api.dependencies import get_image_storage_dir
from api.schemas import DetectedAmenityResponse, ImageUpdateRequest, PropertyImageResponse
from core.amenity_data_manager import AmenityDataManager
from db.models import DetectedAmenity, PropertyImage
from db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/images", tags=["images"])


@router.get("/{image_id}", response_class=FileResponse)
def serve_image(
    image_id: str,
    db: Session = Depends(get_db),
    storage_dir: Path = Depends(get_image_storage_dir),
) -> FileResponse:
    """Return raw bytes for the stored ``PropertyImage`` with ``image_id``."""
    img = db.get(PropertyImage, image_id)
    if img is None:
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found.")

    base = storage_dir.resolve()
    raw_path = Path(img.file_path)
    requested = (raw_path if raw_path.is_absolute() else base / raw_path).resolve()
    if not requested.is_relative_to(base):
        logger.warning(
            "Refusing to serve image %s because %s escapes %s",
            image_id,
            requested,
            base,
        )
        raise HTTPException(status_code=403, detail="Forbidden image path.")

    if not requested.exists() or not requested.is_file():
        raise HTTPException(status_code=404, detail="Image file missing on disk.")

    mime, _encoding = mimetypes.guess_type(str(requested))
    return FileResponse(requested, media_type=mime or "application/octet-stream")


@router.patch("/{image_id}", response_model=PropertyImageResponse)
def patch_image(
    image_id: str,
    body: ImageUpdateRequest,
    db: Session = Depends(get_db),
) -> PropertyImageResponse:
    """
    Update one or more SEO / ordering fields on a stored property image.

    Sending ``{"is_primary": true}`` flips this image to the hero and clears
    the flag on the other images for the same property. Setting any other
    field follows the same "only-provided-keys-are-touched" semantics as the
    property PATCH endpoint.
    """
    fields = body.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields provided to update.")

    manager = AmenityDataManager(db)
    updated = manager.update_image(image_id, fields)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found.")
    db.commit()
    db.refresh(updated)

    amenities = db.query(DetectedAmenity).filter(DetectedAmenity.image_id == image_id).all()
    return PropertyImageResponse(
        id=updated.id,
        file_path=updated.file_path,
        room_type=updated.room_type,
        amenities=[
            DetectedAmenityResponse(
                id=a.id,
                amenity_name=a.amenity_name,
                room_type=a.room_type,
                is_present=a.is_present,
                confidence=a.confidence,
            )
            for a in amenities
        ],
        alt_text=updated.alt_text,
        caption=updated.caption,
        is_primary=updated.is_primary,
        display_order=updated.display_order,
    )
