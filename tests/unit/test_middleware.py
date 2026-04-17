"""
Unit tests for api/middleware.py.

Tests the RequestLoggingMiddleware:
  - Logs request info for non-health requests
  - Skips logging for /health and / paths
  - Passes the response through unchanged
  - Works with different HTTP methods and status codes

We use FastAPI's TestClient with a minimal test app to avoid needing the full
application stack. The logger is mocked so we can inspect what was logged.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware import RequestLoggingMiddleware

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def test_app_with_middleware() -> FastAPI:
    """
    A minimal FastAPI app with RequestLoggingMiddleware attached.

    Has three routes to test different scenarios:
      GET /         → root (skipped)
      GET /health   → health check (skipped)
      GET /items    → normal route (logged)
      POST /items   → POST route (logged)
    """
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/")
    def root():
        return {"root": True}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/items")
    def get_items():
        return {"items": [1, 2, 3]}

    @app.post("/items")
    def create_item():
        return {"created": True}

    @app.get("/error")
    def error_route():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="not found")

    return app


@pytest.fixture
def client(test_app_with_middleware: FastAPI) -> TestClient:
    return TestClient(test_app_with_middleware)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestRequestLoggingMiddleware:
    def test_request_passes_through_correctly(self, client: TestClient):
        """The middleware should not change the response body or status code."""
        response = client.get("/items")
        assert response.status_code == 200
        assert response.json() == {"items": [1, 2, 3]}

    def test_health_path_is_skipped(self, client: TestClient):
        """GET /health should return 200 and not call the logger."""
        with patch("api.middleware.logger") as mock_logger:
            response = client.get("/health")
        assert response.status_code == 200
        mock_logger.info.assert_not_called()

    def test_root_path_is_skipped(self, client: TestClient):
        """GET / should return 200 and not call the logger."""
        with patch("api.middleware.logger") as mock_logger:
            response = client.get("/")
        assert response.status_code == 200
        mock_logger.info.assert_not_called()

    def test_normal_path_is_logged(self, client: TestClient):
        """GET /items should trigger a logger.info() call."""
        with patch("api.middleware.logger") as mock_logger:
            client.get("/items")
        mock_logger.info.assert_called_once()

    def test_log_includes_method_and_path(self, client: TestClient):
        """The log message should contain the HTTP method and path."""
        with patch("api.middleware.logger") as mock_logger:
            client.get("/items")
        call_args = mock_logger.info.call_args
        # First positional arg is the format string
        log_message = call_args[0][0] % call_args[0][1:]
        assert "GET" in log_message
        assert "/items" in log_message

    def test_log_includes_status_code(self, client: TestClient):
        """The logged extra dict should contain status_code."""
        with patch("api.middleware.logger") as mock_logger:
            client.get("/items")
        extra = mock_logger.info.call_args.kwargs.get("extra", {})
        assert extra.get("status_code") == 200

    def test_post_request_is_logged(self, client: TestClient):
        """POST requests should be logged just like GET."""
        with patch("api.middleware.logger") as mock_logger:
            client.post("/items")
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        log_message = call_args[0][0] % call_args[0][1:]
        assert "POST" in log_message

    def test_404_response_is_logged_with_correct_status(self, client: TestClient):
        """A 404 response should be logged with status_code=404."""
        with patch("api.middleware.logger") as mock_logger:
            response = client.get("/error")
        assert response.status_code == 404
        extra = mock_logger.info.call_args.kwargs.get("extra", {})
        assert extra.get("status_code") == 404

    def test_duration_is_logged(self, client: TestClient):
        """The log extra should contain a numeric duration_ms field."""
        with patch("api.middleware.logger") as mock_logger:
            client.get("/items")
        extra = mock_logger.info.call_args.kwargs.get("extra", {})
        assert "duration_ms" in extra
        assert isinstance(extra["duration_ms"], float)
        assert extra["duration_ms"] >= 0
