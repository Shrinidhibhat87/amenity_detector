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

import logging

from fastapi import APIRouter, Depends, HTTPException
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
