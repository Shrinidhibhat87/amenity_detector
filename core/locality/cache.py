"""Postgres-backed cache adapters for the geocode + Overpass clients.

These satisfy the ``GeocodeCache`` / ``PoiCache`` protocols the clients depend
on, persisting results in the ``geocode_cache`` / ``poi_cache`` tables so they
survive restarts. POIs and geocodes don't move, so a stored row means zero
future network cost and no pressure on OSM fair-use limits.

Each adapter wraps a single SQLAlchemy ``Session``. The clients call ``get`` on
a cache miss path and ``set`` after a successful network fetch; here ``set`` does
an upsert (merge) so re-resolving the same key is idempotent.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from core.locality.geocode import GeocodeResult
from core.locality.overpass import Poi
from db.models import GeocodeCache, PoiCache


class DbGeocodeCache:
    """Geocode cache backed by the ``geocode_cache`` table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, key: str) -> GeocodeResult | None:
        row = self._session.get(GeocodeCache, key)
        if row is None:
            return None
        south, north, west, east = row.bbox
        return GeocodeResult(
            query=key,
            latitude=row.latitude,
            longitude=row.longitude,
            bbox=(south, north, west, east),
            display_name=row.display_name,
        )

    def set(self, key: str, value: GeocodeResult) -> None:
        self._session.merge(
            GeocodeCache(
                query_key=key,
                latitude=value.latitude,
                longitude=value.longitude,
                bbox=list(value.bbox),
                display_name=value.display_name,
            )
        )
        self._session.commit()


class DbPoiCache:
    """POI cache backed by the ``poi_cache`` table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, key: str) -> list[Poi] | None:
        row = self._session.get(PoiCache, key)
        if row is None:
            return None
        return [poi_from_dict(d) for d in row.pois]

    def set(self, key: str, value: list[Poi]) -> None:
        self._session.merge(PoiCache(cache_key=key, pois=[poi_to_dict(p) for p in value]))
        self._session.commit()


def poi_to_dict(poi: Poi) -> dict[str, Any]:
    """Serialise a :class:`Poi` for JSON storage."""
    return {
        "category": poi.category,
        "name": poi.name,
        "latitude": poi.latitude,
        "longitude": poi.longitude,
        "osm_type": poi.osm_type,
        "osm_id": poi.osm_id,
        "distance_m": poi.distance_m,
        "transit_type": poi.transit_type,
        "tags": dict(poi.tags),
    }


def poi_from_dict(data: dict[str, Any]) -> Poi:
    """Rebuild a :class:`Poi` from a stored dict."""
    return Poi(
        category=data["category"],
        name=data.get("name", ""),
        latitude=data["latitude"],
        longitude=data["longitude"],
        osm_type=data.get("osm_type", ""),
        osm_id=data.get("osm_id", 0),
        distance_m=data.get("distance_m", 0.0),
        transit_type=data.get("transit_type"),
        tags=data.get("tags", {}),
    )
