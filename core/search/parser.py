"""Natural-language query parser.

Two paths:

  - :class:`QueryParser` is the production path — wraps an OpenAI-SDK client
    pointed at OpenRouter and asks the model for a JSON ``SearchFilter``.
    On any failure (network, malformed JSON, hallucinated shape) it falls
    back to :func:`fallback_regex_parse` so the search UI keeps working.
  - :func:`fallback_regex_parse` is a deterministic regex parser kept in
    lockstep with the frontend mock in ``web/lib/nl-parser.ts``. Useful for
    offline development, integration tests, and graceful degradation.

The LLM is instructed to emit JSON matching :class:`core.search.SearchFilter`.
Pydantic with ``extra="ignore"`` discards stray fields, so a model that
adds a field outside the schema does not crash the parser — it just loses
that bit of context.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Self

from core.search.filter import RoomAmenity, SearchFilter

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "openai/gpt-4o-mini"


_SYSTEM_PROMPT = """You translate a property-search query into JSON.

The user types a free-text query like:
  "3BHK rent under 1500 EUR with fireplace in living room near a park"

You return ONLY a JSON object with the following keys (omit a key if you
are not confident about its value):

  - listing_type:        "rent" | "sale"
  - min_bedrooms:        integer
  - max_bedrooms:        integer
  - min_bathrooms:       integer
  - min_price:           number
  - max_price:           number
  - currency:            3-letter ISO 4217 code (e.g. "EUR", "USD", "INR")
  - property_type:       "apartment" | "house" | "villa" | "studio" | "other"
  - furnishing:          "furnished" | "semi_furnished" | "unfurnished"
  - locality:            free-text locality / neighbourhood name
  - country_code:        2-letter ISO 3166-1 alpha-2 (e.g. "DE", "IN")
  - required_amenities:  list of {"room_type": "<room>"|null, "amenity_name": "<amenity>"}
  - optional_amenities:  same shape as required_amenities (nice-to-have hits)
  - near:                list of nearby POI categories ("park", "metro", ...)
  - free_text:           leftover phrasing that did not fit the structured fields

Rules:
  - Output ONLY valid JSON. No prose, no markdown, no code fences.
  - If a value is missing or ambiguous, omit the key entirely.
  - "BHK" means bedrooms. "3BHK" => min_bedrooms = max_bedrooms = 3.
  - "buy" / "purchase" => listing_type = "sale".
  - room_type uses snake_case ("living_room", "kitchen", "master_bedroom").
"""


class QueryParser:
    """LLM-backed natural-language → ``SearchFilter`` translator.

    Construct via :meth:`from_env` in production; the unit tests inject a
    mocked OpenAI client through the constructor.
    """

    def __init__(self, *, openai_client: OpenAI, model: str) -> None:
        self._client = openai_client
        self._model = model

    @classmethod
    def from_env(cls) -> Self:
        from openai import OpenAI  # imported lazily so unit tests can mock it

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required to build a QueryParser; see .env.example."
            )
        timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30"))
        client = OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=api_key, timeout=timeout)
        model = os.getenv("SEARCH_PARSER_MODEL", _DEFAULT_MODEL)
        return cls(openai_client=client, model=model)

    def parse(self, query: str) -> SearchFilter:
        """Parse ``query`` via the LLM, falling back to regex on any failure."""
        cleaned = query.strip()
        if not cleaned:
            return SearchFilter(free_text="")

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": cleaned},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except Exception as exc:
            logger.warning("LLM parse failed (%s); using regex fallback", exc)
            return fallback_regex_parse(cleaned)

        raw = response.choices[0].message.content or ""
        try:
            payload: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("LLM emitted invalid JSON (%s); using regex fallback", exc)
            return fallback_regex_parse(cleaned)

        if not isinstance(payload, dict):
            logger.warning("LLM emitted non-object JSON (%r); using regex fallback", type(payload))
            return fallback_regex_parse(cleaned)

        try:
            return SearchFilter.model_validate(payload)
        except Exception as exc:
            logger.warning("LLM JSON failed schema validation (%s); using regex fallback", exc)
            return fallback_regex_parse(cleaned)


# ── Regex fallback ───────────────────────────────────────────────────────────

# Kept deliberately small. The frontend ``web/lib/nl-parser.ts`` is the
# canonical reference; this Python copy is wired to the same vocabulary so
# the regression risk lives in one place.
_AMENITY_VOCAB: tuple[str, ...] = (
    "pool",
    "wifi",
    "gym",
    "parking",
    "balcony",
    "fireplace",
    "garden",
    "terrace",
    "kitchen",
    "washer",
    "dryer",
    "dishwasher",
    "oven",
    "refrigerator",
    "microwave",
    "heating",
    "air conditioning",
    "elevator",
    "doorman",
)

_ROOM_VOCAB: tuple[tuple[str, str], ...] = (
    # (display phrase, normalised key)
    ("living room", "living_room"),
    ("kitchen", "kitchen"),
    ("bedroom", "bedroom"),
    ("master bedroom", "master_bedroom"),
    ("bathroom", "bathroom"),
    ("dining room", "dining_room"),
    ("balcony", "balcony"),
    ("garage", "garage"),
)

_WORD_NUMERALS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

_CURRENCY_SYMBOLS: dict[str, str] = {"$": "USD", "€": "EUR", "£": "GBP"}


def fallback_regex_parse(query: str) -> SearchFilter:
    """Deterministic regex extraction — the offline / failure-mode parser."""
    text = query.lower().strip()
    if not text:
        return SearchFilter(free_text="")

    out: dict[str, Any] = {"free_text": query}

    # Bedrooms.
    if (m := re.search(r"(\d+)\s*bhk", text)) or (
        m := re.search(r"(\d+)[-\s]?(?:bedroom|bedrooms|bed|beds)\b", text)
    ):
        n = int(m.group(1))
        out["min_bedrooms"] = n
        out["max_bedrooms"] = n
    else:
        for word, n in _WORD_NUMERALS.items():
            if re.search(rf"\b{word}\s+(?:bedroom|bedrooms|bed|beds)\b", text):
                out["min_bedrooms"] = n
                out["max_bedrooms"] = n
                break

    # Listing type.
    if re.search(r"\brent(?:al|als|ing)?\b", text):
        out["listing_type"] = "rent"
    elif re.search(r"\b(?:sale|buy|purchase|to\s+buy)\b", text):
        out["listing_type"] = "sale"

    # Price ceiling.
    price_match = re.search(
        r"\b(?:under|below|less\s+than|max(?:imum)?|up\s+to)\s+([€$£])?\s*([\d,]+)\s*([a-z]{3})?",
        text,
    )
    if price_match:
        out["max_price"] = float(price_match.group(2).replace(",", ""))
        iso = price_match.group(3)
        symbol = price_match.group(1)
        if iso:
            out["currency"] = iso.upper()
        elif symbol and symbol in _CURRENCY_SYMBOLS:
            out["currency"] = _CURRENCY_SYMBOLS[symbol]
    else:
        bare = re.search(r"([€$£])\s*([\d,]+)", text)
        if bare:
            out["max_price"] = float(bare.group(2).replace(",", ""))
            symbol = bare.group(1)
            if symbol in _CURRENCY_SYMBOLS:
                out["currency"] = _CURRENCY_SYMBOLS[symbol]

    # Amenities — "<amenity> in <room>" tuples first, then bare amenities.
    required: list[RoomAmenity] = []
    matched_amenities: set[str] = set()
    for amenity in _AMENITY_VOCAB:
        for room_phrase, room_key in _ROOM_VOCAB:
            pattern = rf"\b{re.escape(amenity)}\s+in\s+(?:the\s+)?{re.escape(room_phrase)}\b"
            if re.search(pattern, text):
                required.append(RoomAmenity(room_type=room_key, amenity_name=amenity))
                matched_amenities.add(amenity)

    for amenity in _AMENITY_VOCAB:
        if amenity in matched_amenities:
            continue
        if re.search(rf"\b{re.escape(amenity)}\b", text):
            required.append(RoomAmenity(room_type=None, amenity_name=amenity))
            matched_amenities.add(amenity)

    if required:
        out["required_amenities"] = required

    return SearchFilter.model_validate(out)
