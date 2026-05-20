"""Unit tests for the embeddings backfill script.

We test the pure :func:`backfill` function against an in-memory SQLite
database, using a fake embedder so no network is required.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base, Property
from scripts.backfill_embeddings import backfill


@pytest.fixture()
def db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    session.add_all(
        [
            Property(
                id="p1",
                name="A",
                description="apartment with fireplace",
                created_at=datetime.now(UTC),
            ),
            Property(
                id="p2",
                name="B",
                description="villa with pool",
                created_at=datetime.now(UTC),
            ),
            # Already-embedded row — must be skipped by the backfill query.
            Property(
                id="p3",
                name="C",
                description="already indexed",
                description_embedding=[0.1, 0.2, 0.3],
                created_at=datetime.now(UTC),
            ),
            # No description — also skipped.
            Property(
                id="p4",
                name="D",
                description=None,
                created_at=datetime.now(UTC),
            ),
        ]
    )
    session.commit()
    return session


class TestBackfill:
    def test_embeds_only_pending_rows(self, db: Session) -> None:
        embedder = MagicMock()
        embedder.embed_batch.return_value = [[1.0, 2.0], [3.0, 4.0]]

        written = backfill(db, embedder, batch_size=10)

        assert written == 2
        p1 = db.get(Property, "p1")
        p2 = db.get(Property, "p2")
        p3 = db.get(Property, "p3")
        p4 = db.get(Property, "p4")
        assert p1 is not None and p2 is not None and p3 is not None and p4 is not None
        assert p1.description_embedding == [1.0, 2.0]
        assert p2.description_embedding == [3.0, 4.0]
        # Pre-existing embedding untouched.
        assert p3.description_embedding == [0.1, 0.2, 0.3]
        # No-description row stayed NULL.
        assert p4.description_embedding is None

    def test_dry_run_writes_nothing(self, db: Session) -> None:
        embedder = MagicMock()
        embedder.embed_batch.return_value = [[9.9]]

        written = backfill(db, embedder, batch_size=10, dry_run=True)

        assert written == 0
        embedder.embed_batch.assert_not_called()
        for prop_id in ("p1", "p2"):
            prop = db.get(Property, prop_id)
            assert prop is not None
            assert prop.description_embedding is None

    def test_paginates_across_batches(self, db: Session) -> None:
        embedder = MagicMock()
        # Two pending rows, batch size 1 — so the loop should iterate twice.
        embedder.embed_batch.side_effect = [[[0.1]], [[0.2]]]

        written = backfill(db, embedder, batch_size=1)

        assert written == 2
        assert embedder.embed_batch.call_count == 2

    def test_skips_empty_vectors(self, db: Session) -> None:
        embedder = MagicMock()
        # Embedder returns [] for the second row (e.g. blank text).
        embedder.embed_batch.return_value = [[1.0], []]

        written = backfill(db, embedder, batch_size=10)

        assert written == 1
        p1 = db.get(Property, "p1")
        p2 = db.get(Property, "p2")
        assert p1 is not None and p2 is not None
        assert p1.description_embedding == [1.0]
        assert p2.description_embedding is None
