"""
Tests for POST /api/v1/properties/{id}/describe.

We mock the VLM and DB so these run fast without external services.

Design notes:
  - For FastAPI dependency injection, we use `app.dependency_overrides` (the
    correct way) rather than `unittest.mock.patch`, which cannot intercept
    Depends() calls at the framework level.
  - The DB is an in-memory SQLite instance — no external services needed.
  - The VLM client is a MagicMock so no real inference is triggered.
"""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from api.dependencies import get_image_storage_dir, get_model_registry
from api.main import app
from db.models import Base, Property
from db.session import get_db
from models.base import VLMResponse
from models.registry import ModelRegistry

# ── In-memory DB fixture ─────────────────────────────────────────────────────


@pytest.fixture()
def in_memory_db() -> Generator[Session, None, None]:
    """
    SQLite in-memory database with all ORM tables created.

    We use StaticPool so that all connections share the same in-memory
    database. Without it, each new SQLAlchemy connection sees a blank DB
    (SQLite creates an independent in-memory DB per connection).

    Provides a clean, isolated database for each test without needing
    a running PostgreSQL instance.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        # StaticPool keeps a single connection open so all sessions share
        # the same in-memory database and see the tables we create here.
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


# ── VLM mock fixture ─────────────────────────────────────────────────────────


@pytest.fixture()
def mock_vlm() -> MagicMock:
    """
    A mock VLMClient that returns a canned description response.

    This prevents any real VLM inference (Ollama / Gemini) from being called.
    """
    mock = MagicMock()
    mock.generate.return_value = VLMResponse(
        raw_text="A lovely property with a sofa and a fridge.",
        model_name="gemini-2.0-flash",
    )
    return mock


# ── App-level fixture with dependency overrides ───────────────────────────────


@pytest.fixture()
def client(
    in_memory_db: Session, mock_vlm: MagicMock, tmp_path
) -> Generator[TestClient, None, None]:
    """
    TestClient for the full FastAPI app with all heavy dependencies overridden.

    - DB: in-memory SQLite (no PostgreSQL needed)
    - Registry: contains only the mock VLM
    - Image storage: a pytest-managed tmp directory
    """
    # Build a minimal registry that holds our mock VLM under the test model name
    registry = ModelRegistry()
    registry._clients["gemini-2.0-flash"] = mock_vlm

    # Override all three dependencies that the describe endpoint uses
    app.dependency_overrides[get_db] = lambda: in_memory_db
    app.dependency_overrides[get_model_registry] = lambda: registry
    app.dependency_overrides[get_image_storage_dir] = lambda: tmp_path / "images"

    # app.state.model_registry is read by the health endpoint; set it to avoid 503
    app.state.model_registry = registry

    yield TestClient(app)

    # Restore default dependencies so other tests are not affected
    app.dependency_overrides.clear()


# ── Helper ────────────────────────────────────────────────────────────────────


def _make_property(db: Session, name: str = "Test House") -> Property:
    """
    Insert a minimal Property row into the test DB and return it.

    Args:
        db:   The test SQLAlchemy session.
        name: Human-readable property name.

    Returns:
        The committed Property ORM instance.
    """
    prop = Property(
        name=name,
        model_used="gemini-2.0-flash",
        extra_info=None,
    )
    db.add(prop)
    db.commit()
    db.refresh(prop)
    return prop


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_describe_returns_description(client: TestClient, in_memory_db: Session) -> None:
    """Happy path: valid property + valid model → returns description string."""
    prop = _make_property(in_memory_db)

    response = client.post(
        f"/api/v1/properties/{prop.id}/describe",
        json={
            "amenities": [
                {"amenity_name": "Sofa", "room_type": "living_room", "is_present": True},
                {"amenity_name": "Dishwasher", "room_type": "kitchen", "is_present": False},
            ],
            "model_name": "gemini-2.0-flash",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["description"] == "A lovely property with a sofa and a fridge."


def test_describe_passes_sidebar_hints_to_prompt(
    client: TestClient,
    in_memory_db: Session,
    mock_vlm: MagicMock,
) -> None:
    """Phase 5 sidebar hints should be included in the VLM prompt."""
    prop = _make_property(in_memory_db)

    response = client.post(
        f"/api/v1/properties/{prop.id}/describe",
        json={
            "amenities": [
                {"amenity_name": "Sofa", "room_type": "living_room", "is_present": True},
            ],
            "model_name": "gemini-2.0-flash",
            "num_rooms": 2,
            "has_kitchen": True,
            "has_balcony": False,
            "has_living_room": True,
        },
    )

    assert response.status_code == 200
    prompt_used: str = mock_vlm.generate.call_args.kwargs["prompt"]
    assert "2 rooms" in prompt_used
    assert "has a kitchen" in prompt_used
    assert "no balcony" in prompt_used
    assert "has a living room" in prompt_used


def test_describe_returns_404_for_unknown_property(client: TestClient) -> None:
    """If property_id does not exist in DB, endpoint returns 404.

    The `client` fixture sets up an in-memory DB (with tables).
    The ID we send just won't match any row, triggering the 404 branch.
    """
    response = client.post(
        "/api/v1/properties/nonexistent-uuid/describe",
        json={
            "amenities": [],
            "model_name": "gemini-2.0-flash",
        },
    )

    assert response.status_code == 404


def test_describe_returns_400_for_unknown_model(client: TestClient, in_memory_db: Session) -> None:
    """If the model_name is not registered in the registry, endpoint returns 400."""
    prop = _make_property(in_memory_db)

    response = client.post(
        f"/api/v1/properties/{prop.id}/describe",
        json={
            "amenities": [
                {"amenity_name": "Sofa", "room_type": "living_room", "is_present": True},
            ],
            "model_name": "unknown-model-xyz",
        },
    )

    assert response.status_code == 400


def test_generate_description_from_amenities_filters_absent(tmp_path: Path) -> None:
    """
    generate_description_from_amenities must only include is_present=True
    amenities in the prompt sent to the VLM.

    This exercises the service layer (core/amenity_system.py) directly —
    no HTTP layer involved — so we bypass the API entirely.
    """
    from core.amenity_system import PropertyAmenitySystem

    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = VLMResponse(
        raw_text="A property with a sofa.", model_name="test"
    )

    system = PropertyAmenitySystem(
        vlm_client=mock_vlm,
        db=MagicMock(),
        image_storage_dir=tmp_path,
    )

    amenities = [
        {"amenity_name": "Sofa", "room_type": "living_room", "is_present": True},
        {"amenity_name": "Air Conditioning", "room_type": "living_room", "is_present": False},
    ]

    result = system.generate_description_from_amenities(amenities, "Test House")

    # Inspect the prompt the VLM was called with
    call_args = mock_vlm.generate.call_args
    # The generate() call uses keyword arguments: generate(image=..., prompt=...)
    prompt_used: str = call_args.kwargs.get("prompt") or call_args.args[1]

    # Present amenity must appear; absent one must NOT
    assert "Sofa" in prompt_used
    assert "Air Conditioning" not in prompt_used
    assert isinstance(result, str)


def test_generate_description_empty_amenities_returns_fallback(tmp_path: Path) -> None:
    """
    If every amenity in the list has is_present=False, the method should return
    the fallback sentinel string without calling the VLM at all.
    """
    from core.amenity_system import PropertyAmenitySystem

    system = PropertyAmenitySystem(
        vlm_client=MagicMock(),
        db=MagicMock(),
        image_storage_dir=tmp_path,
    )

    result = system.generate_description_from_amenities(
        amenities=[{"amenity_name": "Sofa", "room_type": "lr", "is_present": False}],
        property_name="Empty House",
    )

    assert result == "No amenities were confirmed as present."
