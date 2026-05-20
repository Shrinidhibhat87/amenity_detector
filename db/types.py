"""Cross-dialect SQLAlchemy types used across the project.

``Embedding`` stores a fixed-dimension float vector. On PostgreSQL it is a
``vector(N)`` column (pgvector); on every other dialect it falls back to a
JSON-encoded list of floats so SQLite-backed tests work without Docker.

The dispatch happens at DDL emission time via ``load_dialect_impl``; the
application code always sees ``list[float] | None``.
"""

from typing import Any

from sqlalchemy import JSON
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator, TypeEngine

try:
    from pgvector.sqlalchemy import Vector as _PgVector
except ImportError:
    _PgVector = None  # type: ignore[assignment,misc]


class Embedding(TypeDecorator[list[float]]):
    """``vector(N)`` on PostgreSQL, JSON list-of-floats elsewhere.

    The dimension is fixed at the embedding model's output size; the default
    matches ``text-embedding-3-small``. Changing dimensionality requires a
    migration to recreate the column.
    """

    impl = JSON
    cache_ok = True

    DIMENSION = 1536

    def __init__(self, dimension: int = DIMENSION) -> None:
        super().__init__()
        self.dimension = dimension

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "postgresql" and _PgVector is not None:
            return dialect.type_descriptor(_PgVector(self.dimension))
        return dialect.type_descriptor(JSON())
