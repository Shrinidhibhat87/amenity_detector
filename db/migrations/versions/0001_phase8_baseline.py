"""Phase 8 baseline schema.

Captures the schema as it stood at the end of Phase 8 — three tables created
by ``Base.metadata.create_all`` against the ORM definitions before the Phase 9
listing-metadata expansion landed.

Two operational scenarios:

1. Fresh database (no rows, no schema): ``alembic upgrade head`` runs this
   migration first, then ``0002_phase9_listing_metadata`` to add the new
   columns. Result is a fully-current schema.

2. Existing PostgreSQL with data already created via ``Base.metadata.create_all``
   (which is how the live containers were running pre-Alembic): the operator
   runs ``alembic stamp 0001_phase8_baseline`` once to mark this revision as
   already applied without touching the schema, then ``alembic upgrade head``
   runs only ``0002`` to add the new Phase 9 columns. No data is lost.

Revision ID: 0001_phase8_baseline
Revises:
Create Date: 2026-05-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_phase8_baseline"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "properties",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("model_used", sa.String(length=100), nullable=True),
        sa.Column("extra_info", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "images",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "property_id",
            sa.String(length=36),
            sa.ForeignKey("properties.id"),
            nullable=False,
        ),
        sa.Column("file_path", sa.String(length=512), nullable=False),
        sa.Column("room_type", sa.String(length=100), nullable=True),
    )

    op.create_table(
        "detected_amenities",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "property_id",
            sa.String(length=36),
            sa.ForeignKey("properties.id"),
            nullable=False,
        ),
        sa.Column(
            "image_id",
            sa.String(length=36),
            sa.ForeignKey("images.id"),
            nullable=False,
        ),
        sa.Column("amenity_name", sa.String(length=200), nullable=False),
        sa.Column("room_type", sa.String(length=100), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("is_present", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_table("detected_amenities")
    op.drop_table("images")
    op.drop_table("properties")
