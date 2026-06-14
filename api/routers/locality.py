"""Locality enrichment endpoints.

Two shapes, matching the plan's "try-before-create + commit on save":

  POST /api/v1/locality
      Standalone preview. Runs the locality agent for a free-text location and
      returns the result without touching any property. Used by the upload
      wizard's side panel while images are still processing.

  POST /api/v1/properties/{id}/locality
      Runs the agent and persists the result on the property's locality insight.

Both share the agent dependency and the same response shape. The router is the
thin transport layer; the agent + persistence live in core/locality.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_locality_agent
from api.schemas import LocalityRequest, LocalityResponse, PoiResponse
from core.locality.agent import LocalityAgent, LocalityResult
from core.locality.geocode import GeocodeError
from core.locality.overpass import OverpassError
from core.locality.service import persist_insight
from db.models import Property

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["locality"])


def _run_agent(agent: LocalityAgent, body: LocalityRequest) -> LocalityResult:
    """Run the agent, turning upstream-service failures into a clean 503.

    Without this, a Nominatim 403/429 or an Overpass outage would bubble up as an
    unhandled 500 generated *outside* the CORS middleware — which the browser
    surfaces as an opaque "Failed to fetch". An HTTPException is handled inside
    CORS, so the client gets a real status and message instead.
    """
    try:
        return agent.run(
            postal_code=body.postal_code,
            street=body.street,
            country_code=body.country_code,
            radius_m=body.radius_m,
        )
    except (GeocodeError, OverpassError) as exc:
        logger.warning("locality enrichment upstream failure: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Location lookup is temporarily unavailable. Please try again.",
        ) from exc


@router.post("/locality", response_model=LocalityResponse)
def preview_locality(
    body: LocalityRequest,
    agent: LocalityAgent = Depends(get_locality_agent),
) -> LocalityResponse:
    """Run locality enrichment for a PIN (+ optional street) without persisting."""
    return _to_response(_run_agent(agent, body))


@router.post("/properties/{property_id}/locality", response_model=LocalityResponse)
def persist_locality(
    property_id: str,
    body: LocalityRequest,
    db: Session = Depends(get_db),
    agent: LocalityAgent = Depends(get_locality_agent),
) -> LocalityResponse:
    """Run locality enrichment and store the result on the property."""
    prop = db.get(Property, property_id)
    if prop is None:
        raise HTTPException(status_code=404, detail=f"Property {property_id!r} not found")

    result = _run_agent(agent, body)
    persist_insight(db, property_id, result)
    return _to_response(result)


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Tell nginx/proxies not to buffer the stream, so events arrive as they happen.
    "X-Accel-Buffering": "no",
}


def _sse(payload: dict[str, object]) -> str:
    """Format one Server-Sent Event frame."""
    return f"data: {json.dumps(payload)}\n\n"


def _event_stream(
    agent: LocalityAgent,
    body: LocalityRequest,
    *,
    db: Session | None = None,
    property_id: str | None = None,
) -> Iterator[str]:
    """Drive the agent's streaming run, emitting SSE frames for each step.

    Progress events pass straight through; the final ``result`` event is persisted
    (when a property is given) and re-serialised through the response model. A
    geocode/Overpass failure mid-stream becomes an ``error`` frame rather than a
    dropped connection, so the client can show a real message.
    """
    try:
        for event in agent.run_streaming(
            postal_code=body.postal_code,
            street=body.street,
            country_code=body.country_code,
            radius_m=body.radius_m,
        ):
            if event.get("type") == "result":
                result = event["result"]
                if db is not None and property_id is not None:
                    persist_insight(db, property_id, result)
                yield _sse({"type": "result", "data": _to_response(result).model_dump(mode="json")})
            else:
                yield _sse(event)
    except (GeocodeError, OverpassError) as exc:
        logger.warning("locality stream upstream failure: %s", exc)
        yield _sse(
            {
                "type": "error",
                "detail": "Location lookup is temporarily unavailable. Please try again.",
            }
        )


@router.post("/locality/stream")
def preview_locality_stream(
    body: LocalityRequest,
    agent: LocalityAgent = Depends(get_locality_agent),
) -> StreamingResponse:
    """Streaming (SSE) preview — emits per-category progress then the final result."""
    return StreamingResponse(
        _event_stream(agent, body),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/properties/{property_id}/locality/stream")
def persist_locality_stream(
    property_id: str,
    body: LocalityRequest,
    db: Session = Depends(get_db),
    agent: LocalityAgent = Depends(get_locality_agent),
) -> StreamingResponse:
    """Streaming (SSE) enrichment that persists the final result on the property."""
    prop = db.get(Property, property_id)
    if prop is None:
        raise HTTPException(status_code=404, detail=f"Property {property_id!r} not found")
    return StreamingResponse(
        _event_stream(agent, body, db=db, property_id=property_id),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


def _to_response(result: LocalityResult) -> LocalityResponse:
    return LocalityResponse(
        location_query=result.location_query,
        display_name=result.display_name,
        latitude=result.latitude,
        longitude=result.longitude,
        radius_m=result.radius_m,
        blurb=result.blurb,
        category_counts=result.category_counts,
        transit_breakdown=result.transit_breakdown,
        pois=[
            PoiResponse(
                category=p.category,
                name=p.name,
                latitude=p.latitude,
                longitude=p.longitude,
                distance_m=p.distance_m,
                osm_type=p.osm_type,
                osm_id=p.osm_id,
                transit_type=p.transit_type,
            )
            for p in result.pois
        ],
        attribution=result.attribution,
    )
