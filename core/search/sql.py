"""SQL builder for the hybrid search candidate set.

Given a ``SearchFilter`` (from the LLM parser), build a SQLAlchemy ``Select``
that returns the candidate property IDs to feed into the scorer.

The hard filters (price, bedrooms, listing_type, locality, …) live in
``WHERE``. Each ``required_amenities`` tuple becomes its own EXISTS
subquery — one ``AND`` per tuple, so all of them must match. The
``optional_amenities`` and ``near`` hints do NOT live in the SQL at all;
they are scored after the candidate set has been fetched.

``CANDIDATE_LIMIT`` caps the candidate set so the scorer never has to walk
the entire catalogue. 200 is a deliberate guess: it lets a reasonable
``ts_rank_cd`` blend in before rerank without melting the box.
"""

from __future__ import annotations

from sqlalchemy import Select, and_, exists, select

from core.search.filter import SearchFilter
from db.models import DetectedAmenity, Property
from db.status import PropertyStatus

CANDIDATE_LIMIT = 200


def build_candidate_query(filter_: SearchFilter) -> Select:
    """Build the candidate-set SELECT for a structured filter.

    Returns a SQLAlchemy ``Select`` over ``Property``. The caller decides
    whether to ``execute`` it directly or wrap it for ranking.
    """
    stmt = select(Property)
    # Search is a public surface: an unpublished draft must never rank.
    conditions: list = [Property.status == PropertyStatus.PUBLISHED]
    conditions.extend(_scalar_predicates(filter_))
    conditions.extend(_amenity_exists_clauses(filter_))
    if conditions:
        stmt = stmt.where(and_(*conditions))
    return stmt.limit(CANDIDATE_LIMIT)


def _scalar_predicates(f: SearchFilter):
    """Yield WHERE predicates for every non-``None`` scalar field."""
    if f.listing_type is not None:
        yield Property.listing_type == f.listing_type
    if f.min_bedrooms is not None:
        yield Property.num_bedrooms >= f.min_bedrooms
    if f.max_bedrooms is not None:
        yield Property.num_bedrooms <= f.max_bedrooms
    if f.min_bathrooms is not None:
        yield Property.num_bathrooms >= f.min_bathrooms
    if f.min_price is not None:
        yield Property.price >= f.min_price
    if f.max_price is not None:
        yield Property.price <= f.max_price
    if f.currency is not None:
        yield Property.currency == f.currency.upper()
    if f.property_type is not None:
        yield Property.property_type == f.property_type
    if f.furnishing is not None:
        yield Property.furnishing == f.furnishing
    if f.locality is not None:
        yield Property.locality.ilike(f"%{f.locality}%")
    if f.country_code is not None:
        yield Property.country_code == f.country_code.upper()


def _amenity_exists_clauses(f: SearchFilter):
    """Yield an EXISTS subquery per ``required_amenities`` tuple.

    Each tuple becomes:
        EXISTS (
            SELECT 1 FROM detected_amenities da
            WHERE da.property_id = properties.id
              AND da.is_present = true
              AND da.amenity_name = '<amenity>'
              AND da.room_type = '<room>'   -- only when room_type is not None
        )
    """
    for ra in f.required_amenities:
        conds = [
            DetectedAmenity.property_id == Property.id,
            DetectedAmenity.is_present.is_(True),
            DetectedAmenity.amenity_name == ra.amenity_name,
        ]
        if ra.room_type is not None:
            conds.append(DetectedAmenity.room_type == ra.room_type)
        yield exists(select(DetectedAmenity.id).where(and_(*conds)))
