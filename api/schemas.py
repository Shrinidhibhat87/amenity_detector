"""
Pydantic schemas for API request and response bodies.

Why separate schemas from ORM models?
  SQLAlchemy ORM models are tied to the database — they have relationship objects,
  lazy-loaded collections, and metadata that shouldn't leak into the API response.
  Pydantic schemas are pure data shapes: they define exactly what the API sends and
  receives, with validation and serialisation baked in.

  FastAPI uses these schemas to:
    - Validate incoming request bodies (input schemas)
    - Serialise outgoing response bodies (response schemas)
    - Auto-generate the OpenAPI docs at /docs

Naming convention:
  - *Response  — what the API sends to the client
  - *Create    — what the client sends when creating a resource (input)
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# ── Phase 9 listing-metadata enum-like literals ──────────────────────────────
# These are Pydantic Literals rather than DB-native ENUMs so the same schema
# round-trips through SQLite (tests) and PostgreSQL (production) without a
# dialect-specific ENUM type. Pydantic enforces the allowed value set at the
# API boundary.

ListingType = Literal["rent", "sale"]
PricePeriod = Literal["monthly", "weekly", "nightly", "total"]
PropertyType = Literal["apartment", "house", "villa", "studio", "other"]
Furnishing = Literal["furnished", "semi_furnished", "unfurnished"]
CountryCode = Annotated[str, StringConstraints(min_length=2, max_length=2, to_upper=True)]
NonNegativeDecimal = Annotated[Decimal, Field(ge=Decimal("0"))]

# ── Amenity schemas ──────────────────────────────────────────────────────────


class DetectedAmenityResponse(BaseModel):
    """
    One amenity detection result for a single image.

    Returned as part of PropertyImageResponse, which is part of PropertyResponse.
    """

    id: str
    amenity_name: str
    room_type: str | None
    is_present: bool
    confidence: float | None

    # model_config tells Pydantic to read data from SQLAlchemy ORM attributes.
    # Without this, Pydantic would treat the model as a plain dict and fail to
    # read from ORM objects (which use __getattr__ not __getitem__).
    model_config = ConfigDict(from_attributes=True)


# ── Image schemas ────────────────────────────────────────────────────────────


class PropertyImageResponse(BaseModel):
    """
    One uploaded image for a property, including its detected amenities.
    """

    id: str
    file_path: str
    room_type: str | None
    amenities: list[DetectedAmenityResponse]

    # Phase 9 SEO + ordering fields. alt_text is filled by the same VLM call
    # that detects amenities; is_primary marks the hero image used in JSON-LD.
    alt_text: str | None = None
    caption: str | None = None
    is_primary: bool = False
    display_order: int = 0

    model_config = ConfigDict(from_attributes=True)


# ── Property schemas ─────────────────────────────────────────────────────────


class _PropertyMetadataMixin(BaseModel):
    """Shared Phase 9 fields surfaced on both summary and detail responses.

    Every field is optional so legacy rows (created before Phase 9) load with
    ``None`` and the response stays valid.
    """

    slug: str | None = None
    listing_type: ListingType | None = None
    price: Decimal | None = None
    currency: str | None = None
    price_period: PricePeriod | None = None
    num_bedrooms: int | None = None
    num_bathrooms: int | None = None
    area_sqm: Decimal | None = None
    property_type: PropertyType | None = None
    furnishing: Furnishing | None = None
    available_from: date | None = None
    locality: str | None = None
    postal_code: str | None = None
    country_code: str | None = None
    latitude: Decimal | None = None
    longitude: Decimal | None = None
    owner_email: str | None = None


class PropertySummaryResponse(_PropertyMetadataMixin):
    """
    Lightweight property representation for list endpoints.

    Used in GET /api/v1/properties/ and GET /api/v1/properties/search
    to avoid sending the full amenity list (which can be large).
    """

    id: str
    name: str
    description: str | None
    model_used: str | None
    extra_info: str | None
    created_at: datetime
    image_count: int = 0  # Populated from len(property.images) in the router
    first_image_id: str | None = None  # Used by the Gradio browse card thumbnails

    model_config = ConfigDict(from_attributes=True)


class PropertyDetailResponse(_PropertyMetadataMixin):
    """
    Full property details including all images and detected amenities.

    Used in GET /api/v1/properties/{id}.
    """

    id: str
    name: str
    description: str | None
    model_used: str | None
    extra_info: str | None
    created_at: datetime
    images: list[PropertyImageResponse]

    model_config = ConfigDict(from_attributes=True)


# ── Upload response schema ────────────────────────────────────────────────────


class PropertyCreateRequest(BaseModel):
    """
    Request body for creating an empty property shell.

    Used by the Phase 5 UI before uploading images one-by-one. Phase 9 added
    the optional listing-metadata fields below; the API stays backward-compatible
    with the original ``{name, model_name, extra_info}`` body.
    """

    name: str
    model_name: str
    extra_info: str | None = None

    # Phase 9 listing metadata. All optional; ``None`` means "not provided" and
    # the corresponding column stays NULL on the row.
    listing_type: ListingType | None = None
    price: NonNegativeDecimal | None = None
    currency: str | None = None
    price_period: PricePeriod | None = None
    num_bedrooms: Annotated[int, Field(ge=0, le=50)] | None = None
    num_bathrooms: Annotated[int, Field(ge=0, le=50)] | None = None
    area_sqm: NonNegativeDecimal | None = None
    property_type: PropertyType | None = None
    furnishing: Furnishing | None = None
    available_from: date | None = None
    locality: str | None = None
    postal_code: str | None = None
    country_code: CountryCode | None = None
    latitude: Annotated[Decimal, Field(ge=-90, le=90)] | None = None
    longitude: Annotated[Decimal, Field(ge=-180, le=180)] | None = None
    # ``owner_email`` is a single-tenant stub today (no auth). Stored as plain
    # str rather than ``EmailStr`` to avoid pulling in the ``email-validator``
    # dependency for what is currently a free-text contact field. When real
    # auth lands in a later phase this will move onto a Users table.
    owner_email: str | None = None


class PropertyUpdateRequest(BaseModel):
    """
    Request body for PATCH /api/v1/properties/{id}.

    Same field set as :class:`PropertyCreateRequest`'s metadata block, minus
    ``name``, ``model_name``, ``extra_info`` (these are creation-time choices)
    and ``slug`` (immutable post-creation to keep public URLs stable).

    ``extra="forbid"`` rejects unknown keys with HTTP 422 — this is what
    blocks attempts to PATCH ``slug``.
    """

    listing_type: ListingType | None = None
    price: NonNegativeDecimal | None = None
    currency: str | None = None
    price_period: PricePeriod | None = None
    num_bedrooms: Annotated[int, Field(ge=0, le=50)] | None = None
    num_bathrooms: Annotated[int, Field(ge=0, le=50)] | None = None
    area_sqm: NonNegativeDecimal | None = None
    property_type: PropertyType | None = None
    furnishing: Furnishing | None = None
    available_from: date | None = None
    locality: str | None = None
    postal_code: str | None = None
    country_code: CountryCode | None = None
    latitude: Annotated[Decimal, Field(ge=-90, le=90)] | None = None
    longitude: Annotated[Decimal, Field(ge=-180, le=180)] | None = None
    owner_email: str | None = None
    # The wizard's describe step calls POST /describe to generate text, then
    # PATCHes that text back here to persist it on the property row.
    description: str | None = None

    model_config = ConfigDict(extra="forbid")


class PropertyCreateResponse(BaseModel):
    """Response returned after creating an empty property shell."""

    property_id: str
    message: str
    property: PropertyDetailResponse


class ImageDetectionResponse(BaseModel):
    """Response returned after processing one image for an existing property."""

    property_id: str
    image: PropertyImageResponse


# ── Health check schema ───────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str
    database: str  # "ok" or "unreachable"
    version: str = "1.0.0"


# ── Image patch schema ───────────────────────────────────────────────────────


class ImageUpdateRequest(BaseModel):
    """
    Request body for PATCH /api/v1/images/{id}.

    All fields are optional; only those actually present in the request are
    written. Setting ``is_primary=True`` flips the hero flag on this image and
    clears the flag on every other image in the same property (one hero per
    property is the contract). ``extra="forbid"`` rejects unknown keys.
    """

    alt_text: str | None = None
    caption: str | None = None
    is_primary: bool | None = None
    display_order: Annotated[int, Field(ge=0, le=999)] | None = None

    model_config = ConfigDict(extra="forbid")


# ── Describe endpoint schemas ─────────────────────────────────────────────────


class AmenityEditItem(BaseModel):
    """One amenity entry as edited by the user in the UI."""

    amenity_name: str
    room_type: str
    is_present: bool


class DescribeRequest(BaseModel):
    """
    Request body for POST /api/v1/properties/{id}/describe.

    The UI sends the user's edited amenity table so the VLM can regenerate
    the description based only on the amenities the user confirmed as present.
    """

    amenities: list[AmenityEditItem]
    model_name: str
    num_rooms: int | None = None
    has_kitchen: bool | None = None
    has_balcony: bool | None = None
    has_living_room: bool | None = None
    hints: dict[str, bool] | None = None


class DescribeResponse(BaseModel):
    """Response from the describe endpoint."""

    description: str


# ── Search endpoint schemas ──────────────────────────────────────────────────


class SearchRequest(BaseModel):
    """Body for ``POST /api/v1/search``.

    A single free-text query — the backend parses it into a structured
    ``SearchFilter`` and runs the hybrid pipeline (SQL filter + cosine
    rerank + FTS blend).
    """

    query: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    limit: Annotated[int, Field(ge=1, le=100)] = 20
