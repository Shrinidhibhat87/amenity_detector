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

from datetime import datetime

from pydantic import BaseModel, ConfigDict


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

    model_config = ConfigDict(from_attributes=True)


# ── Property schemas ─────────────────────────────────────────────────────────


class PropertySummaryResponse(BaseModel):
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

    model_config = ConfigDict(from_attributes=True)


class PropertyDetailResponse(BaseModel):
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


class UploadResponse(BaseModel):
    """
    Response returned after a successful property upload.

    Includes the full property details so the client can immediately display results.
    """

    property_id: str
    message: str
    property: PropertyDetailResponse


# ── Health check schema ───────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str
    database: str  # "ok" or "unreachable"
    version: str = "1.0.0"