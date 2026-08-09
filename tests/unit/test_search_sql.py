"""Unit tests for the SQL builder.

We compile each ``Select`` against the PostgreSQL dialect with literal binds
so the assertions read like the SQL that would actually run in production —
no parameter placeholders to chase.
"""

from __future__ import annotations

from sqlalchemy.dialects import postgresql

from core.search import RoomAmenity, SearchFilter
from core.search.sql import CANDIDATE_LIMIT, build_candidate_query


def _sql(filter_: SearchFilter) -> str:
    stmt = build_candidate_query(filter_)
    return str(
        stmt.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


class TestEmptyFilter:
    def test_no_constraints_selects_all_published_with_cap(self) -> None:
        sql = _sql(SearchFilter())
        assert "FROM properties" in sql
        # Search is public, so the published predicate is always there even
        # when the parsed filter is empty.
        assert "properties.status" in sql
        assert f"LIMIT {CANDIDATE_LIMIT}" in sql


class TestScalarFilters:
    def test_listing_type_adds_where(self) -> None:
        sql = _sql(SearchFilter(listing_type="rent"))
        assert "properties.listing_type = 'rent'" in sql

    def test_bedrooms_range(self) -> None:
        sql = _sql(SearchFilter(min_bedrooms=2, max_bedrooms=4))
        assert "properties.num_bedrooms >= 2" in sql
        assert "properties.num_bedrooms <= 4" in sql

    def test_min_bathrooms(self) -> None:
        sql = _sql(SearchFilter(min_bathrooms=2))
        assert "properties.num_bathrooms >= 2" in sql

    def test_price_ceiling_with_currency(self) -> None:
        sql = _sql(SearchFilter(max_price=1500.0, currency="EUR"))
        assert "properties.price <= 1500" in sql
        assert "properties.currency = 'EUR'" in sql

    def test_price_floor_only(self) -> None:
        sql = _sql(SearchFilter(min_price=500.0))
        assert "properties.price >= 500" in sql

    def test_property_type_and_furnishing(self) -> None:
        sql = _sql(SearchFilter(property_type="apartment", furnishing="furnished"))
        assert "properties.property_type = 'apartment'" in sql
        assert "properties.furnishing = 'furnished'" in sql

    def test_locality_uses_case_insensitive_like(self) -> None:
        sql = _sql(SearchFilter(locality="Frankfurt"))
        # SQLAlchemy compiles literal LIKE binds with doubled `%%` to escape
        # the psycopg paramstyle; production SQL still sees the single `%`.
        assert "ILIKE '%%Frankfurt%%'" in sql

    def test_country_code_uppercased(self) -> None:
        sql = _sql(SearchFilter(country_code="de"))
        # Country codes stored as ISO alpha-2 uppercase.
        assert "properties.country_code = 'DE'" in sql


class TestRequiredAmenities:
    def test_single_required_amenity_emits_exists(self) -> None:
        sql = _sql(
            SearchFilter(
                required_amenities=[RoomAmenity(room_type="living_room", amenity_name="fireplace")]
            )
        )
        assert "EXISTS" in sql
        assert "detected_amenities" in sql
        assert "amenity_name = 'fireplace'" in sql
        assert "room_type = 'living_room'" in sql
        assert "is_present" in sql.lower()

    def test_room_anywhere_omits_room_predicate(self) -> None:
        sql = _sql(
            SearchFilter(required_amenities=[RoomAmenity(room_type=None, amenity_name="wifi")])
        )
        assert "amenity_name = 'wifi'" in sql
        # No room constraint when room_type is None.
        assert "room_type =" not in sql

    def test_multiple_required_emit_multiple_exists(self) -> None:
        sql = _sql(
            SearchFilter(
                required_amenities=[
                    RoomAmenity(room_type="kitchen", amenity_name="oven"),
                    RoomAmenity(room_type="living_room", amenity_name="fireplace"),
                ]
            )
        )
        assert sql.count("EXISTS") == 2
        assert "amenity_name = 'oven'" in sql
        assert "amenity_name = 'fireplace'" in sql


class TestCombinedFilter:
    def test_full_filter_emits_all_predicates(self) -> None:
        sql = _sql(
            SearchFilter(
                listing_type="rent",
                min_bedrooms=3,
                max_bedrooms=3,
                max_price=1500.0,
                currency="EUR",
                required_amenities=[
                    RoomAmenity(room_type="living_room", amenity_name="fireplace"),
                ],
            )
        )
        assert "properties.listing_type = 'rent'" in sql
        assert "properties.num_bedrooms >= 3" in sql
        assert "properties.num_bedrooms <= 3" in sql
        assert "properties.price <= 1500" in sql
        assert "properties.currency = 'EUR'" in sql
        assert "EXISTS" in sql
