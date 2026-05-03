"""
HTTP request/response logging middleware — Phase 4 observability.

Why middleware?
  Middleware wraps every incoming request before it reaches a route and every
  outgoing response after the route finishes. This is the right place to log
  request metadata (method, path, query params) and response metadata (status code,
  duration) because the logic lives in one place rather than in every route handler.

The ``RequestLoggingMiddleware`` class:
  - Logs a single structured line per request containing:
      method, path, query_string, status_code, duration_ms
  - Uses ``time.perf_counter()`` for sub-millisecond precision.
  - Passes all extra fields as ``extra=`` kwargs so JSON logging (see logging_config.py)
    picks them up as top-level JSON keys for easy querying.

Usage:
  Registered in api/main.py via ``app.add_middleware(RequestLoggingMiddleware)``.
  It runs for EVERY request, including /health polls, so keep it lightweight.
"""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Starlette/FastAPI middleware that logs one line per HTTP request.

    Each log entry contains:
      - HTTP method (GET, POST, …)
      - URL path (/api/v1/properties/{property_id}/images)
      - Query string (if any)
      - Response status code (200, 201, 400, 500, …)
      - Processing duration in milliseconds

    The /health endpoint is excluded from logging to avoid drowning logs with
    Docker Compose health-check polls every 10 seconds.

    Errors during route processing are not swallowed — the exception propagates
    normally after the error is logged.
    """

    # Skip logging for these paths to reduce noise from health-check polls
    _SKIP_PATHS: frozenset[str] = frozenset({"/health", "/"})

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """
        Called by Starlette for every HTTP request.

        Args:
            request:   The incoming HTTP request.
            call_next: Callable that passes the request to the next middleware
                       or to the actual route handler.

        Returns:
            The HTTP response produced by the route handler (or an error response).
        """
        # Skip health-check endpoints to keep logs clean
        if request.url.path in self._SKIP_PATHS:
            # call_next is typed as Any by Starlette but it is always an async callable
            import typing

            next_callable = typing.cast(
                typing.Callable[[Request], typing.Awaitable[Response]], call_next
            )
            return await next_callable(request)

        start = time.perf_counter()

        # Type cast: Starlette types call_next as object in BaseHTTPMiddleware
        import typing

        next_callable = typing.cast(
            typing.Callable[[Request], typing.Awaitable[Response]], call_next
        )
        response = await next_callable(request)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "%s %s → %d  (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            extra={
                # These become top-level JSON keys when LOG_FORMAT=json
                "http_method": request.method,
                "http_path": request.url.path,
                "query_string": str(request.url.query),
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response
