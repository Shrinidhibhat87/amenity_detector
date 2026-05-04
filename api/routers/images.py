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
from db.models import PropertyImage
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
