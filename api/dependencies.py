"""
FastAPI dependency functions.

FastAPI's dependency injection (Depends()) is how we avoid passing the database
session, model registry, and config around as global variables.

How Depends() works:
  When FastAPI calls a route function, it first calls any dependency functions
  listed in the route's parameters. The return value (or yielded value) is then
  passed into the route function automatically.

  Example:
    @router.post("/upload")
    def upload(registry: ModelRegistry = Depends(get_model_registry)):
        ...
    FastAPI calls get_model_registry() first and injects the result.

Why this pattern?
  1. Testability: in tests, you can override dependencies with fakes/mocks.
  2. No globals: dependencies live on `app.state`, not module-level globals.
  3. Lifetime control: DB sessions live for one request; the model registry
     lives for the app's lifetime.
"""

import os
from pathlib import Path

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from db.session import get_db
from models.registry import ModelRegistry


def get_model_registry(request: Request) -> ModelRegistry:
    """
    Retrieve the ModelRegistry singleton stored on `app.state`.

    The registry is created once at startup in `api/main.py` and stored on
    `app.state.model_registry`. This function makes it available to any
    route that lists it as a dependency.

    Args:
        request: The incoming HTTP request (FastAPI injects this automatically).

    Returns:
        The application-level ModelRegistry instance.

    Raises:
        HTTPException 503: If the registry was not initialised at startup.
    """
    registry: ModelRegistry | None = getattr(request.app.state, "model_registry", None)
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="Model registry not initialised. The service is starting up.",
        )
    return registry


def get_image_storage_dir() -> Path:
    """
    Return the directory where uploaded images are stored.

    Reads IMAGE_STORAGE_DIR from the environment. In Docker Compose, this
    is set to a path that maps to a mounted volume so images survive restarts.

    Returns:
        Path to the image storage directory.
    """
    return Path(os.getenv("IMAGE_STORAGE_DIR", "./storage/images"))


# Re-export get_db so routers only need to import from this module
__all__ = ["get_db", "get_model_registry", "get_image_storage_dir"]