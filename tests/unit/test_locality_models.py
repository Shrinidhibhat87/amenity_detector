"""Unit tests for the locality ORM models + the Postgres cache adapters.

Run against in-memory SQLite — they verify the SQLAlchemy definitions persist
and retrieve, the per-property uniqueness constraint, and that the cache
adapters round-trip a GeocodeResult / list[Poi] through the DB unchanged.
Migration coverage lives in tests/integration.
"""

from __future__ import annotations

from collections.abc import Generator
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from core.locality.cache import DbGeocodeCache, DbPoiCache
from core.locality.geocode import GeocodeResult
from core.locality.overpass import Poi
from db.models import Base, LocalityInsight, Property


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


def _property(session: Session) -> Property:
    prop = Property(name="Frankfurt flat")
    session.add(prop)
    session.commit()
    return prop


def test_locality_insight_round_trip(db_session: Session) -> None:
    prop = _property(db_session)
    insight = LocalityInsight(
        property_id=prop.id,
        location_query="60311",
        display_name="60311 Frankfurt am Main, Germany",
        latitude=Decimal("50.110900"),
        longitude=Decimal("8.682100"),
        radius_m=1500,
        pois=[{"category": "school", "name": "Goethe-Schule"}],
        category_counts={"school": 1, "park": 2},
        blurb="Lively central location with schools and parks nearby.",
    )
    db_session.add(insight)
    db_session.commit()
    db_session.expire_all()

    loaded = db_session.query(LocalityInsight).filter_by(property_id=prop.id).one()
    assert loaded.location_query == "60311"
    assert loaded.category_counts == {"school": 1, "park": 2}
    assert loaded.pois[0]["name"] == "Goethe-Schule"
    # Attribution defaults to the OSM credit even when not set explicitly.
    assert "OpenStreetMap" in loaded.attribution
    # Relationship wiring both directions.
    assert loaded.property.id == prop.id
    db_session.refresh(prop)
    assert prop.locality_insight is not None
    assert prop.locality_insight.id == loaded.id


def test_one_insight_per_property(db_session: Session) -> None:
    prop = _property(db_session)
    db_session.add(LocalityInsight(property_id=prop.id, location_query="a"))
    db_session.commit()

    db_session.add(LocalityInsight(property_id=prop.id, location_query="b"))
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_db_geocode_cache_round_trip(db_session: Session) -> None:
    cache = DbGeocodeCache(db_session)
    assert cache.get("60311") is None  # miss

    result = GeocodeResult(
        query="60311",
        latitude=50.1109,
        longitude=8.6821,
        bbox=(50.0969, 50.1249, 8.6621, 8.7021),
        display_name="60311 Frankfurt am Main, Germany",
    )
    cache.set("60311", result)

    loaded = cache.get("60311")
    assert loaded == result


def test_db_geocode_cache_set_is_idempotent(db_session: Session) -> None:
    cache = DbGeocodeCache(db_session)
    first = GeocodeResult(
        query="berlin",
        latitude=52.52,
        longitude=13.405,
        bbox=(52.3, 52.7, 13.0, 13.8),
        display_name="Berlin",
    )
    cache.set("berlin", first)
    # Re-resolving (e.g. data refresh) must upsert, not raise on the PK.
    cache.set("berlin", first)
    assert cache.get("berlin") == first


def test_db_poi_cache_round_trip(db_session: Session) -> None:
    cache = DbPoiCache(db_session)
    assert cache.get("k") is None

    pois = [
        Poi(
            category="school",
            name="Goethe-Schule",
            latitude=50.111,
            longitude=8.682,
            osm_type="node",
            osm_id=1,
            distance_m=42.0,
            tags={"amenity": "school", "name": "Goethe-Schule"},
        )
    ]
    cache.set("k", pois)

    loaded = cache.get("k")
    assert loaded == pois
