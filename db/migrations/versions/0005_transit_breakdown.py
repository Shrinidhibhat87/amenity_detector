"""Add transit_breakdown to locality insights.

Stores the per-mode transit composition ({"bus": 12, "rail": 2}) alongside the
opaque transit count, so the panel can show the real mix (e.g. a bus-only city)
instead of one number. Nullable — legacy rows written before subtypes existed
simply carry NULL.

Revision ID: 0005_transit_breakdown
Revises: 0004_locality_enrichment
Create Date: 2026-06-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0005_transit_breakdown"
down_revision: str | None = "0004_locality_enrichment"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Same dialect-aware JSON used by the ORM models.
_JSON_DOC = sa.JSON(none_as_null=True).with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column(
        "locality_insights",
        sa.Column("transit_breakdown", _JSON_DOC, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("locality_insights", "transit_breakdown")
