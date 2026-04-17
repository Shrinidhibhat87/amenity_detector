"""
Models router — handles /api/v1/models endpoints.

Endpoints:
  GET /api/v1/models   List all VLMs and their availability status

This is used by the Gradio UI dropdown (Phase 3) to populate the model selector
dynamically. The UI calls this endpoint at load time rather than hardcoding model names.
"""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import get_model_registry
from models.registry import ModelRegistry, SUPPORTED_MODELS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["models"])


class ModelInfo(BaseModel):
    """Information about a single VLM."""

    name: str
    available: bool   # True = registered and reachable; False = not configured/offline
    description: str  # Human-readable explanation for the UI


# Static descriptions for each model — shown in the UI dropdown tooltip
_MODEL_DESCRIPTIONS: dict[str, str] = {
    "qwen2.5vl:7b": (
        "Qwen 2.5 VL 7B (local via Ollama) — "
        "good accuracy, fits in 6GB VRAM with 4-bit quantisation"
    ),
    "llama3.2-vision:11b": (
        "LLaMA 3.2 Vision 11B (local via Ollama) — "
        "higher quality but may need CPU offloading on 6GB GPU (slower)"
    ),
    "gemini-2.0-flash": (
        "Gemini 2.0 Flash (Google API, free tier) — "
        "no local GPU needed, fastest option, 1500 req/day free"
    ),
}


@router.get("/", response_model=list[ModelInfo])
def list_models(registry: ModelRegistry = Depends(get_model_registry)) -> list[ModelInfo]:
    """
    List all supported VLMs and whether they are currently available.

    A model is 'available' if it was successfully registered at startup
    (Ollama server reachable + model pulled, or Gemini API key present).

    Returns:
        List of ModelInfo objects covering all supported models.
    """
    available = set(registry.available_models())
    return [
        ModelInfo(
            name=model_name,
            available=(model_name in available),
            description=_MODEL_DESCRIPTIONS.get(model_name, "No description available."),
        )
        for model_name in SUPPORTED_MODELS
    ]