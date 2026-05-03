"""
ModelRegistry — maps OpenRouter model ids to OpenRouterVLMClient instances.

Adding a model means appending one entry to AVAILABLE_MODELS — no code change
elsewhere required.
"""

from __future__ import annotations

import logging
from typing import TypedDict

from models.base import VLMClient
from models.openrouter_client import OpenRouterVLMClient

logger = logging.getLogger(__name__)


class ModelEntry(TypedDict):
    id: str
    label: str
    json_mode: bool


AVAILABLE_MODELS: list[ModelEntry] = [
    {"id": "openai/gpt-4o-mini", "label": "GPT-4o Mini", "json_mode": True},
    {"id": "google/gemini-pro-1.5", "label": "Gemini Pro 1.5", "json_mode": True},
    {
        "id": "meta-llama/llama-3.2-11b-vision-instruct",
        "label": "Llama 3.2 11B Vision",
        "json_mode": False,
    },
    {"id": "qwen/qwen2-vl-72b-instruct", "label": "Qwen2-VL 72B", "json_mode": False},
]

SUPPORTED_MODELS: list[str] = [model["id"] for model in AVAILABLE_MODELS]
DEFAULT_MODEL = "openai/gpt-4o-mini"


class ModelRegistry:
    """Lazy mapping of OpenRouter model id -> OpenRouterVLMClient."""

    def __init__(self) -> None:
        self._entries: dict[str, ModelEntry] = {model["id"]: model for model in AVAILABLE_MODELS}
        self._clients: dict[str, VLMClient] = {}

    @classmethod
    def from_env(cls) -> ModelRegistry:
        """Build a registry. Clients are constructed on first .get() call."""
        return cls()

    def get(self, model_id: str) -> VLMClient:
        if model_id not in self._entries:
            raise KeyError(
                f"Model '{model_id}' is not supported. Available: {list(self._entries)}."
            )
        if model_id not in self._clients:
            entry = self._entries[model_id]
            self._clients[model_id] = OpenRouterVLMClient(
                model=entry["id"],
                json_mode=entry["json_mode"],
            )
            logger.info("Instantiated OpenRouter client for %s", model_id)
        return self._clients[model_id]

    def available_models(self) -> list[str]:
        return list(self._entries)

    def all_supported_models(self) -> list[str]:
        return SUPPORTED_MODELS
