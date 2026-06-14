"""Locality service layer — wiring + persistence.

Two responsibilities the API routers lean on:

  - :func:`build_locality_agent` constructs a :class:`LocalityAgent` whose
    geocode + Overpass clients use the Postgres-backed caches bound to the
    request's session (so lookups persist across requests).
  - :func:`persist_insight` upserts a :class:`LocalityResult` onto a property's
    single :class:`LocalityInsight` row.
"""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy.orm import Session

from core.locality.agent import LocalityAgent, LocalityResult
from core.locality.cache import DbGeocodeCache, DbPoiCache, poi_to_dict
from core.locality.geocode import GeocodeClient
from core.locality.overpass import OverpassClient
from db.models import LocalityInsight


def build_locality_agent(session: Session) -> LocalityAgent:
    """Build an agent whose OSM clients cache into this session's database."""
    geocode = GeocodeClient.from_env(cache=DbGeocodeCache(session))
    overpass = OverpassClient.from_env(cache=DbPoiCache(session))
    return LocalityAgent.from_env(geocode_client=geocode, overpass_client=overpass)


def persist_insight(session: Session, property_id: str, result: LocalityResult) -> LocalityInsight:
    """Upsert ``result`` onto the property's locality insight row."""
    insight = session.query(LocalityInsight).filter_by(property_id=property_id).one_or_none()
    if insight is None:
        insight = LocalityInsight(property_id=property_id)
        session.add(insight)

    insight.location_query = result.location_query
    insight.display_name = result.display_name or None
    insight.latitude = _as_decimal(result.latitude)
    insight.longitude = _as_decimal(result.longitude)
    insight.radius_m = result.radius_m
    insight.pois = [poi_to_dict(p) for p in result.pois]
    insight.category_counts = result.category_counts
    insight.transit_breakdown = result.transit_breakdown
    insight.blurb = result.blurb
    insight.attribution = result.attribution

    session.commit()
    session.refresh(insight)
    return insight


def _as_decimal(value: float | None) -> Decimal | None:
    return Decimal(str(value)) if value is not None else None
