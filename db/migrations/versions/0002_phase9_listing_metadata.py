"""Phase 9 — listing metadata expansion.

Adds 17 nullable columns to ``properties`` and 4 columns to ``images`` so the
schema can carry real-listing data (price, rooms, location, SEO alt text,
hero-image flag) instead of just a name and a free-text blob.

Also adds a partial index on ``detected_amenities`` for the hot path used by
Phase 11 search: filter by present amenities for a given property/room.

Notes:
  - All new ``properties`` columns are nullable so existing rows survive.
  - ``images.is_primary`` and ``images.display_order`` are NOT NULL with
    server defaults; existing rows pick up the defaults on ALTER.
  - The ``description_embedding`` (pgvector) and ``seo_meta`` (JSONB) columns
    promised in the SPEC are deliberately deferred to Phases 11 and 12 —
    their final type depends on choices made in those phases, and adding
    placeholders now would just force a follow-up migration to retype them.

Revision ID: 0002_phase9_listing_metadata
Revises: 0001_phase8_baseline
Create Date: 2026-05-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_phase9_listing_metadata"
down_revision: str | None = "0001_phase8_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Order matches db/models.py for easy diffing.
_PROPERTY_NEW_COLUMNS: list[sa.Column] = [
    # Unique constraint added separately as a unique index — SQLite cannot
    # ALTER TABLE to add an inline unique constraint, but CREATE UNIQUE INDEX
    # works on both SQLite and PostgreSQL.
    sa.Column("slug", sa.String(length=160), nullable=True),
    sa.Column("listing_type", sa.String(length=8), nullable=True),
    sa.Column("price", sa.Numeric(precision=12, scale=2), nullable=True),
    sa.Column("currency", sa.String(length=3), nullable=True),
    sa.Column("price_period", sa.String(length=8), nullable=True),
    sa.Column("num_bedrooms", sa.SmallInteger(), nullable=True),
    sa.Column("num_bathrooms", sa.SmallInteger(), nullable=True),
    sa.Column("area_sqm", sa.Numeric(precision=8, scale=2), nullable=True),
    sa.Column("property_type", sa.String(length=16), nullable=True),
    sa.Column("furnishing", sa.String(length=16), nullable=True),
    sa.Column("available_from", sa.Date(), nullable=True),
    sa.Column("locality", sa.String(length=120), nullable=True),
    sa.Column("postal_code", sa.String(length=16), nullable=True),
    sa.Column("country_code", sa.String(length=2), nullable=True),
    sa.Column("latitude", sa.Numeric(precision=9, scale=6), nullable=True),
    sa.Column("longitude", sa.Numeric(precision=9, scale=6), nullable=True),
    sa.Column("owner_email", sa.String(length=255), nullable=True),
]

_IMAGE_NEW_COLUMNS: list[sa.Column] = [
    sa.Column("alt_text", sa.String(length=500), nullable=True),
    sa.Column("caption", sa.String(length=500), nullable=True),
    sa.Column(
        "is_primary",
        sa.Boolean(),
        nullable=False,
        server_default=sa.false(),
    ),
    sa.Column(
        "display_order",
        sa.SmallInteger(),
        nullable=False,
        server_default=sa.text("0"),
    ),
]


def upgrade() -> None:
    for col in _PROPERTY_NEW_COLUMNS:
        op.add_column("properties", col)

    op.create_index(
        "ix_properties_slug_unique",
        "properties",
        ["slug"],
        unique=True,
    )

    for col in _IMAGE_NEW_COLUMNS:
        op.add_column("images", col)

    # Partial index used by the Phase 11 search hot path: "find properties
    # whose detected_amenities row says (room=living_room, amenity=fireplace)
    # AND is_present is true". Both PostgreSQL and SQLite support partial
    # indexes via dialect-specific kwargs; the value differs because SQLite
    # encodes booleans as 0/1 while PostgreSQL has a real BOOLEAN type.
    op.create_index(
        "ix_detected_amenities_present_room_amenity",
        "detected_amenities",
        ["property_id", "room_type", "amenity_name"],
        sqlite_where=sa.text("is_present = 1"),
        postgresql_where=sa.text("is_present = true"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_detected_amenities_present_room_amenity",
        table_name="detected_amenities",
    )
    for col in reversed(_IMAGE_NEW_COLUMNS):
        op.drop_column("images", col.name)
    op.drop_index("ix_properties_slug_unique", table_name="properties")
    for col in reversed(_PROPERTY_NEW_COLUMNS):
        op.drop_column("properties", col.name)
