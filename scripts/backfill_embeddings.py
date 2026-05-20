"""One-shot backfill of ``description_embedding`` for existing properties.

Walks every property whose embedding is still ``NULL`` and computes it
from the same text composition the matview uses (name + description +
locality + amenity-room phrases). Batched so the embedder sees up to
``BATCH_SIZE`` rows per HTTP call.

Idempotent:
  - rows that already have an embedding are skipped,
  - failures on one batch do not affect already-committed batches,
  - re-running the script picks up where it left off.

Usage:
    # Inside the api container (recommended — env vars + DB DNS pre-set):
    docker compose exec api uv run python -m scripts.backfill_embeddings

    # Locally with .env loaded:
    uv run python -m scripts.backfill_embeddings

Flags:
    --batch-size N   Override the batch size (default: 50).
    --dry-run        Print the row IDs that would be indexed, write nothing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Iterator

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from core.embeddings import EmbeddingsClient, EmbeddingsError
from core.search.pipeline import build_index_text, refresh_search_doc
from db.models import Property

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 50


def _iter_pending_properties(
    db: Session,
    batch_size: int,
    skip_ids: set[str],
) -> Iterator[list[Property]]:
    """Yield batches of properties missing their embedding.

    We re-query each pass so a row committed by the previous batch is no
    longer returned — the simplest way to make a long-running backfill
    survive transient failures without holding a giant cursor.

    ``skip_ids`` collects IDs that were already attempted but ended up with
    an empty vector (e.g. blank source text). Without this guard the loop
    would re-fetch the same row forever because its embedding column stays
    NULL.
    """
    while True:
        stmt = (
            select(Property)
            .where(Property.description_embedding.is_(None))
            .where(Property.description.isnot(None))
            .limit(batch_size)
        )
        if skip_ids:
            stmt = stmt.where(Property.id.notin_(skip_ids))
        rows = list(db.execute(stmt).scalars())
        if not rows:
            return
        yield rows


def backfill(
    db: Session,
    embedder: EmbeddingsClient,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
) -> int:
    """Embed every row missing a vector. Returns the number of rows written."""
    total_written = 0
    skip_ids: set[str] = set()

    for batch in _iter_pending_properties(db, batch_size, skip_ids):
        texts = [build_index_text(prop) for prop in batch]
        ids = [prop.id for prop in batch]

        if dry_run:
            for prop_id, text_ in zip(ids, texts, strict=True):
                logger.info("DRY-RUN would embed %s (%d chars)", prop_id, len(text_))
            # No write → stop after the first batch so we don't loop forever.
            return total_written

        try:
            vectors = embedder.embed_batch(texts)
        except EmbeddingsError as exc:
            logger.error("Batch of %d failed: %s — aborting backfill.", len(batch), exc)
            db.rollback()
            return total_written

        written_this_batch = 0
        for prop, vector in zip(batch, vectors, strict=True):
            if not vector:
                logger.warning("Empty vector for %s; leaving NULL", prop.id)
                skip_ids.add(prop.id)
                continue
            prop.description_embedding = vector
            total_written += 1
            written_this_batch += 1

        db.commit()
        logger.info(
            "Committed %d rows (running total: %d)",
            written_this_batch,
            total_written,
        )

    return total_written


def _build_session() -> Session:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL is required.")
    engine = create_engine(db_url)
    return sessionmaker(bind=engine)()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    embedder = EmbeddingsClient.from_env()
    db = _build_session()
    try:
        written = backfill(
            db,
            embedder,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
        if not args.dry_run and written > 0:
            refresh_search_doc(db)
            db.commit()
        logger.info("Done. %d rows embedded.", written)
    finally:
        db.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
