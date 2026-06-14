"""Locality enrichment tables.

Adds the three tables behind the locality agent:

  - ``locality_insights`` — one row per property: resolved coordinate, raw nearby
    POIs (JSON), per-category counts, the synthesized Lage blurb, and the OSM
    attribution. ``property_id`` is unique so re-running enrichment upserts.
  - ``geocode_cache`` — persistent Nominatim cache keyed on the normalised query.
  - ``poi_cache`` — persistent Overpass cache keyed on (lat, lon, radius, category).

JSON columns are JSONB on PostgreSQL (indexable) and plain JSON on SQLite (the
test path), matching the ``_JSON_DOC`` variant used in db/models.py.

Revision ID: 0004_locality_enrichment
Revises: 0003_hybrid_search
Create Date: 2026-06-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0004_locality_enrichment"
down_revision: str | None = "0003_hybrid_search"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Same dialect-aware JSON used by the ORM models.
_JSON_DOC = sa.JSON(none_as_null=True).with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "locality_insights",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "property_id",
            sa.String(length=36),
            sa.ForeignKey("properties.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("location_query", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=512), nullable=True),
        sa.Column("latitude", sa.Numeric(precision=9, scale=6), nullable=True),
        sa.Column("longitude", sa.Numeric(precision=9, scale=6), nullable=True),
        sa.Column("radius_m", sa.Integer(), nullable=True),
        sa.Column("pois", _JSON_DOC, nullable=False),
        sa.Column("category_counts", _JSON_DOC, nullable=False),
        sa.Column("blurb", sa.Text(), nullable=True),
        sa.Column("attribution", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "geocode_cache",
        sa.Column("query_key", sa.String(length=255), primary_key=True),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("bbox", _JSON_DOC, nullable=False),
        sa.Column("display_name", sa.String(length=512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "poi_cache",
        sa.Column("cache_key", sa.String(length=255), primary_key=True),
        sa.Column("pois", _JSON_DOC, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("poi_cache")
    op.drop_table("geocode_cache")
    op.drop_table("locality_insights")
