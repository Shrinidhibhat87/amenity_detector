"""Structured search filter — the contract between the NL parser and the
SQL builder.

The parser (LLM or regex fallback) emits a ``SearchFilter``; the SQL builder
turns it into a ``WHERE`` clause + EXISTS subqueries; the scorer uses the
remaining ``free_text`` and ``near`` hints to boost candidates after the
hard filters have run.

Unknown fields from the LLM are dropped by Pydantic (``extra="ignore"``) so
a hallucinated field name cannot crash the pipeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RoomAmenity(BaseModel):
    """An ``(amenity, room?)`` tuple — ``None`` room means "anywhere"."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    room_type: str | None = None
    amenity_name: str


class SearchFilter(BaseModel):
    """Structured filter extracted from a free-text query.

    All fields are optional — the parser emits only what it can confidently
    pull out of the query. The SQL builder ignores ``None`` fields, so
    leaving a field unset means "do not constrain on this dimension".
    """

    model_config = ConfigDict(extra="ignore")

    listing_type: Literal["rent", "sale"] | None = None

    min_bedrooms: int | None = None
    max_bedrooms: int | None = None
    min_bathrooms: int | None = None

    min_price: float | None = None
    max_price: float | None = None
    currency: str | None = None  # ISO 4217 (3-letter) when known.

    property_type: str | None = None
    furnishing: str | None = None

    locality: str | None = None
    country_code: str | None = None

    required_amenities: list[RoomAmenity] = Field(default_factory=list)
    optional_amenities: list[RoomAmenity] = Field(default_factory=list)

    # Locality / neighbourhood hints — e.g. ["park", "metro"]. Used only as a
    # scoring boost once the locality summary feed is wired in.
    near: list[str] = Field(default_factory=list)

    # Leftover phrasing the structured fields could not absorb. Embedded and
    # fed to cosine rerank.
    free_text: str = ""
