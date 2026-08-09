"""Add the publication status column to properties.

Before this migration every property row was public the moment it was created,
so an abandoned wizard draft appeared in browse, search, the sitemap, llms.txt
and the JSONL feed. The new column carries the publication state and every
public surface filters on it.

Existing rows are backfilled to 'published' so nothing that was already
visible disappears; new rows start as 'draft' via the server default.

Revision ID: 0006_listing_status
Revises: 0005_transit_breakdown
Create Date: 2026-08-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_listing_status"
down_revision: str | None = "0005_transit_breakdown"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # server_default is what backfills existing rows on ALTER; it also keeps
    # inserts that bypass the ORM (fixtures, psql) inside the lifecycle.
    op.add_column(
        "properties",
        sa.Column("status", sa.String(length=24), nullable=False, server_default="draft"),
    )
    # Rows that predate the lifecycle were already publicly visible — keep them
    # that way rather than silently unpublishing a live catalogue.
    op.execute("UPDATE properties SET status = 'published'")
    op.create_index("ix_properties_status", "properties", ["status"])


def downgrade() -> None:
    op.drop_index("ix_properties_status", table_name="properties")
    op.drop_column("properties", "status")
