"""Hybrid search infrastructure.

Adds the column + indexes used by the hybrid search pipeline (LLM-parsed
filters, SQL ``WHERE``, pgvector cosine rerank, blended with Postgres FTS):

  - ``properties.description_embedding`` — fixed-dim float vector.
    ``vector(1536)`` on PostgreSQL, JSON on SQLite (test path).
  - ``property_search_doc`` materialised view: per-property concatenated text
    (name + description + locality + amenity-room phrases) with a
    precomputed ``tsvector`` for ``ts_rank_cd``.
  - ivfflat cosine index on ``description_embedding``.
  - GIN index on ``property_search_doc.search_tsv``.
  - Composite btree on ``detected_amenities (room_type, amenity_name)``
    filtered to present rows, for the ``required_amenities`` EXISTS subquery.

The matview, GIN, ivfflat and vector type are PostgreSQL-only and gated on
``op.get_bind().dialect.name == 'postgresql'``. On SQLite the column is JSON
via the ``Embedding`` TypeDecorator and the search pipeline computes cosine
in Python against the stored arrays.

Revision ID: 0003_hybrid_search
Revises: 0002_phase9_listing_metadata
Create Date: 2026-05-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from db.types import Embedding

revision: str = "0003_hybrid_search"
down_revision: str | None = "0002_phase9_listing_metadata"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_PROPERTY_SEARCH_DOC_SQL = """
CREATE MATERIALIZED VIEW property_search_doc AS
SELECT
    p.id AS property_id,
    coalesce(p.name, '') AS name,
    coalesce(p.description, '') AS description,
    coalesce(p.locality, '') AS locality,
    coalesce(
        string_agg(
            DISTINCT da.amenity_name || ' in ' || coalesce(da.room_type, 'unknown'),
            ' '
        ),
        ''
    ) AS amenity_phrases,
    to_tsvector(
        'simple',
        coalesce(p.name, '') || ' ' ||
        coalesce(p.description, '') || ' ' ||
        coalesce(p.locality, '') || ' ' ||
        coalesce(
            string_agg(
                DISTINCT da.amenity_name || ' in ' || coalesce(da.room_type, 'unknown'),
                ' '
            ),
            ''
        )
    ) AS search_tsv
FROM properties p
LEFT JOIN detected_amenities da
       ON da.property_id = p.id
      AND da.is_present = true
GROUP BY p.id, p.name, p.description, p.locality
"""


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column(
        "properties",
        sa.Column("description_embedding", Embedding(), nullable=True),
    )

    if dialect != "postgresql":
        return

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_properties_desc_embedding_ivfflat "
        "ON properties USING ivfflat (description_embedding vector_cosine_ops) "
        "WITH (lists = 100)"
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_detected_amenities_room_amenity_present "
        "ON detected_amenities (room_type, amenity_name) "
        "WHERE is_present = true"
    )

    op.execute(_PROPERTY_SEARCH_DOC_SQL)
    op.execute(
        "CREATE UNIQUE INDEX ix_property_search_doc_property_id "
        "ON property_search_doc (property_id)"
    )
    op.execute(
        "CREATE INDEX ix_property_search_doc_tsv "
        "ON property_search_doc USING GIN (search_tsv)"
    )


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_property_search_doc_tsv")
        op.execute("DROP INDEX IF EXISTS ix_property_search_doc_property_id")
        op.execute("DROP MATERIALIZED VIEW IF EXISTS property_search_doc")
        op.execute("DROP INDEX IF EXISTS ix_detected_amenities_room_amenity_present")
        op.execute("DROP INDEX IF EXISTS ix_properties_desc_embedding_ivfflat")

    op.drop_column("properties", "description_embedding")
