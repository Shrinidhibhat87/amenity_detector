"""
Integration test configuration and shared fixtures.

How integration tests differ from unit tests:
  - They test multiple components working together (API + DB + service layer)
  - They use a real (in-memory SQLite) database with the actual ORM models
  - They use FastAPI's TestClient, which sends real HTTP requests to the app
  - They mock ONLY the VLM clients (not the DB or API layers)

Why SQLite for integration tests (not PostgreSQL)?
  SQLite runs in-memory without any external process — perfect for CI.
  The tests still exercise the full SQLAlchemy ORM layer, so schema bugs
  and query logic errors are caught. PostgreSQL-specific behaviour (e.g.,
  full-text search, UUID types) would require a test Docker container,
  which we keep out of the fast test suite.

  If you want to run against a real PostgreSQL:
      TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/test_db pytest tests/integration/

Fixtures defined here:
  test_db_engine   — SQLAlchemy engine backed by in-memory SQLite
  db_session       — Session that uses the test engine (injected into TestClient)
  mock_vlm_client  — A fake VLMClient that returns canned JSON without calling a real VLM
  test_app         — The FastAPI app with DB and VLM dependencies overridden
  client           — FastAPI TestClient connected to test_app
"""

import io
import json
import os
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from api.main import app
from db.models import Base
from db.session import get_db
from models.base import VLMClient, VLMResponse
from models.registry import ModelRegistry

# Use in-memory SQLite for integration tests (no Docker needed)
# Override with TEST_DATABASE_URL environment variable for PostgreSQL testing
_TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:")


@pytest.fixture(scope="session")
def test_db_engine():
    """
    Create a single SQLite engine for the entire test session.

    scope="session" means this engine is shared across all integration tests.
    The tables are created once and cleaned up after all tests complete.
    """
    engine = create_engine(
        _TEST_DATABASE_URL,
        connect_args={"check_same_thread": False} if "sqlite" in _TEST_DATABASE_URL else {},
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def db_session(test_db_engine) -> Generator[Session, None, None]:
    """
    Create a fresh database session for each test, rolled back after.

    Using a rollback (instead of truncating tables) is faster and ensures
    each test starts with a clean state without any I/O.
    """
    connection = test_db_engine.connect()
    transaction = connection.begin()
    TestSession = sessionmaker(bind=connection)
    session = TestSession()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def mock_vlm_client() -> VLMClient:
    """
    A fake VLMClient that returns a pre-canned JSON amenity detection response.

    Using a fake here (not a mock) means integration tests verify the full
    pipeline logic (parsing, DB writes, response serialisation) without
    needing Ollama or a Gemini key.

    The response simulates a kitchen image with refrigerator and oven detected.
    """
    fake_response_json = json.dumps(
        {
            "refrigerator": True,
            "oven": True,
            "dishwasher": False,
            "microwave": False,
        }
    )

    client = MagicMock(spec=VLMClient)
    type(client).model_name = property(lambda self: "openai/gpt-4o-mini")
    client.generate.return_value = VLMResponse(
        raw_text=fake_response_json,
        model_name="openai/gpt-4o-mini",
    )
    return client


@pytest.fixture(scope="function")
def mock_registry(mock_vlm_client: VLMClient) -> ModelRegistry:
    """
    A ModelRegistry containing only the fake VLM client.
    """
    registry = ModelRegistry()
    registry._clients["openai/gpt-4o-mini"] = mock_vlm_client
    return registry


@pytest.fixture(scope="function")
def test_app(db_session: Session, mock_registry: ModelRegistry, tmp_path):
    """
    Configure the FastAPI app with test-specific dependency overrides.

    FastAPI's dependency_overrides allows us to swap out real dependencies
    with test fakes without changing the application code.
    """
    # Override the DB session with our test session
    app.dependency_overrides[get_db] = lambda: db_session

    # Override the model registry with our fake registry
    from api.dependencies import get_model_registry

    app.dependency_overrides[get_model_registry] = lambda: mock_registry

    # Override image storage to a pytest temp directory (cleaned up automatically)
    from api.dependencies import get_image_storage_dir

    app.dependency_overrides[get_image_storage_dir] = lambda: tmp_path / "images"

    # Store the registry on app.state (the health check and startup code expect it)
    app.state.model_registry = mock_registry

    yield app

    # Clean up overrides after each test
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def client(test_app) -> TestClient:
    """FastAPI TestClient connected to the test app."""
    return TestClient(test_app)


@pytest.fixture(scope="function")
def sample_image_bytes() -> bytes:
    """
    A minimal valid JPEG image as bytes — used as file upload content in tests.
    """
    buffer = io.BytesIO()
    img = PILImage.new("RGB", (100, 100), color=(100, 150, 200))
    img.save(buffer, format="JPEG")
    return buffer.getvalue()
