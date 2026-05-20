"""Hybrid scoring for search candidates.

The scorer blends three signals with fixed weights — the same weights the
SPEC promises (§11, "Pipeline" step 5):

    score = 0.5 * cosine(query, description)
          + 0.3 * ts_rank_cd(search_doc, query)
          + 0.2 * boost(optional_amenities + near)

Cosine is computed in pure Python so the same code works on SQLite (where
embeddings are stored as JSON lists) and on PostgreSQL (where pgvector
returns the column as a Python list of floats after registration).

FTS rank arrives precomputed — Postgres computes it via ``ts_rank_cd`` in
the candidate query; on SQLite we pass 0.0 and live with cosine + boost.

Boost is one tenth per matched optional hit, clamped at 1.0. It exists to
break ties when several properties have the same hard constraints but
different "nice-to-have" hits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.search.filter import RoomAmenity, SearchFilter

COSINE_WEIGHT = 0.5
FTS_WEIGHT = 0.3
BOOST_WEIGHT = 0.2


@dataclass(slots=True)
class Candidate:
    """A property that passed the SQL filter, ready to be scored."""

    property_id: str
    description_embedding: list[float] | None
    fts_rank: float = 0.0
    matched_optional: list[RoomAmenity] = field(default_factory=list)
    matched_near: list[str] = field(default_factory=list)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of two equal-length vectors, in [-1, 1].

    A zero vector or a length mismatch returns 0.0 — neither situation
    should crash a live search because of one bad row.
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def score_candidate(
    _filter: SearchFilter,
    candidate: Candidate,
    query_embedding: list[float] | None,
) -> float:
    """Compute the blended score for a single candidate.

    ``_filter`` is reserved for future per-field reweighting (e.g. boosting
    when ``listing_type`` matches strongly) — unused today, kept in the
    signature so callers don't need to change when we add it.
    """
    if query_embedding is not None and candidate.description_embedding is not None:
        cos = cosine_similarity(query_embedding, candidate.description_embedding)
    else:
        cos = 0.0

    fts = candidate.fts_rank

    boost = 0.1 * (len(candidate.matched_optional) + len(candidate.matched_near))
    boost = min(boost, 1.0)

    return COSINE_WEIGHT * cos + FTS_WEIGHT * fts + BOOST_WEIGHT * boost
