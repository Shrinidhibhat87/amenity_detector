"""End-to-end hybrid search pipeline + reindex helpers.

The pipeline glues the parser, the SQL builder and the scorer together:

  raw query
      └─► QueryParser.parse                              ─►  SearchFilter
                                                              │
      ┌──────────────────────────────────────────────────────┘
      ▼
  EmbeddingsClient.embed_text(free_text or query)        ─►  query_embedding
                                                              │
      ┌──────────────────────────────────────────────────────┘
      ▼
  build_candidate_query(filter).execute()                ─►  candidates (≤ 200)
      │
      ├─ on Postgres: SELECT ts_rank_cd from property_search_doc by id
      │
      ▼
  score_candidate per row, sort desc by score, slice to ``limit``.

The same module also exposes :func:`reindex_property` and
:func:`refresh_search_doc` — both are no-ops on non-PostgreSQL dialects so
the SQLite test path stays portable.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.search.filter import SearchFilter
from core.search.parser import fallback_regex_parse
from core.search.score import Candidate, score_candidate
from core.search.sql import build_candidate_query
from db.models import Property

logger = logging.getLogger(__name__)

DEFAULT_RESULT_LIMIT = 20


class _Parser(Protocol):
    def parse(self, query: str) -> SearchFilter: ...


class _Embedder(Protocol):
    def embed_text(self, text: str) -> list[float]: ...


class SearchPipeline:
    """Composable search runner; the parser and embedder are pluggable.

    Either dependency can be ``None`` and the pipeline still works — it
    falls back to the regex parser and a zero-vector query embedding
    respectively. This keeps the search UI functional in development
    environments where the LLM/embedding services aren't reachable.
    """

    def __init__(
        self,
        *,
        parser: _Parser | None = None,
        embedder: _Embedder | None = None,
    ) -> None:
        self._parser = parser
        self._embedder = embedder

    def search(
        self,
        db: Session,
        query: str,
        *,
        limit: int = DEFAULT_RESULT_LIMIT,
    ) -> list[Property]:
        cleaned = query.strip()
        if not cleaned:
            return []

        filter_ = (
            self._parser.parse(cleaned) if self._parser is not None else fallback_regex_parse(cleaned)
        )

        query_emb = self._safe_embed(filter_.free_text or cleaned)

        candidates: Sequence[Property] = (
            db.execute(build_candidate_query(filter_)).scalars().all()
        )
        if not candidates:
            return []

        fts_ranks = self._fts_ranks(db, [c.id for c in candidates], cleaned)

        scored: list[tuple[float, Property]] = []
        for prop in candidates:
            cand = Candidate(
                property_id=prop.id,
                description_embedding=prop.description_embedding,
                fts_rank=fts_ranks.get(prop.id, 0.0),
            )
            scored.append((score_candidate(filter_, cand, query_emb), prop))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [prop for _, prop in scored[:limit]]

    def _safe_embed(self, text_: str) -> list[float] | None:
        if self._embedder is None or not text_.strip():
            return None
        try:
            return self._embedder.embed_text(text_)
        except Exception as exc:
            logger.warning("query embedding failed (%s); ranking without cosine", exc)
            return None

    def _fts_ranks(
        self,
        db: Session,
        property_ids: list[str],
        query: str,
    ) -> dict[str, float]:
        if not _is_postgres(db) or not property_ids:
            return {}
        try:
            rows = db.execute(
                text(
                    "SELECT property_id, "
                    "       ts_rank_cd(search_tsv, plainto_tsquery('simple', :q)) AS rank "
                    "FROM property_search_doc "
                    "WHERE property_id = ANY(:ids)"
                ),
                {"q": query, "ids": property_ids},
            ).all()
        except Exception as exc:
            logger.warning("FTS rank lookup failed (%s); ranking without FTS", exc)
            return {}
        return {row.property_id: float(row.rank or 0.0) for row in rows}


# ── Indexing helpers ─────────────────────────────────────────────────────────


def build_index_text(prop: Property) -> str:
    """Concatenate the fields fed to the embedding model for a property.

    The same composition is mirrored in the ``property_search_doc`` matview
    so that lexical and semantic signals stay in sync.
    """
    parts: list[str] = []
    if prop.name:
        parts.append(prop.name)
    if prop.description:
        parts.append(prop.description)
    if prop.locality:
        parts.append(prop.locality)
    for amenity in prop.amenities:
        if not amenity.is_present:
            continue
        room = amenity.room_type or "unknown"
        parts.append(f"{amenity.amenity_name} in {room}")
    return " ".join(parts).strip()


def reindex_property(
    db: Session,
    embedder: _Embedder | None,
    property_id: str,
) -> None:
    """Recompute and persist ``description_embedding`` for a single property.

    No-op when ``embedder`` is ``None`` (e.g. local dev without LiteLLM
    creds). The matview refresh is also skipped on non-PostgreSQL — the
    SQLite test database has no matview to refresh.
    """
    if embedder is None:
        logger.info("reindex_property(%s): no embedder configured, skipping", property_id)
        return

    prop = db.get(Property, property_id)
    if prop is None:
        logger.warning("reindex_property(%s): property not found", property_id)
        return

    text_ = build_index_text(prop)
    if not text_:
        prop.description_embedding = None
        db.flush()
        return

    try:
        prop.description_embedding = embedder.embed_text(text_)
    except Exception as exc:
        logger.warning("reindex_property(%s): embedding failed: %s", property_id, exc)
        return

    db.flush()
    refresh_search_doc(db)


def refresh_search_doc(db: Session) -> None:
    """Refresh the FTS matview on PostgreSQL; no-op elsewhere."""
    if not _is_postgres(db):
        return
    try:
        db.execute(text("REFRESH MATERIALIZED VIEW property_search_doc"))
    except Exception as exc:
        logger.warning("REFRESH MATERIALIZED VIEW property_search_doc failed: %s", exc)


def _is_postgres(db: Session) -> bool:
    bind = db.get_bind()
    return bind.dialect.name == "postgresql" if bind is not None else False
