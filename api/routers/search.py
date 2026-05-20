"""Natural-language search endpoint.

``POST /api/v1/search`` accepts a single free-text ``query`` field and
returns a ranked list of :class:`PropertySummaryResponse`. The heavy
lifting lives in :class:`core.search.pipeline.SearchPipeline`; this router
is the thin transport layer that turns a request into a pipeline call and
serialises the resulting ORM rows into the wire schema.

The same response shape as ``GET /api/v1/properties/`` is used so the web
client can render results with the same component as the browse list.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_search_pipeline
from api.schemas import PropertySummaryResponse, SearchRequest
from core.search.pipeline import SearchPipeline
from db.models import Property

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.post("", response_model=list[PropertySummaryResponse])
def search(
    body: SearchRequest,
    db: Session = Depends(get_db),
    pipeline: SearchPipeline = Depends(get_search_pipeline),
) -> list[PropertySummaryResponse]:
    """Run a hybrid natural-language search and return ranked results."""
    properties = pipeline.search(db, body.query, limit=body.limit)
    return [_to_summary(prop) for prop in properties]


def _to_summary(prop: Property) -> PropertySummaryResponse:
    """Shape an ORM ``Property`` into the wire summary, matching list_properties.

    Excludes private SQLAlchemy attributes and the heavy
    ``description_embedding`` vector — clients never need the raw vector.
    """
    payload = {k: v for k, v in prop.__dict__.items() if not k.startswith("_")}
    payload.pop("description_embedding", None)
    return PropertySummaryResponse(
        **payload,
        image_count=len(prop.images),
        first_image_id=prop.images[0].id if prop.images else None,
    )
