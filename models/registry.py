"""
ModelRegistry — maps model name strings to VLMClient instances.

Why a registry?
  The API receives the desired model as a string from the UI (e.g., "qwen2.5vl:7b").
  The registry translates that string into the correct client object without the
  router needing to know how each client is constructed.

  It also acts as a single place to see all supported models — adding a new VLM
  means adding one entry here.

Supported model names (as they appear in the UI dropdown):
  - "qwen2.5vl:7b"        → OllamaClient("qwen2.5vl:7b")
  - "llama3.2-vision:11b" → OllamaClient("llama3.2-vision:11b")
  - "gemini-2.0-flash"    → GeminiClient()
"""

import logging
import os

from models.base import VLMClient
from models.gemini_client import GeminiClient
from models.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# The canonical list of model identifiers the UI presents to the user.
# This is the source of truth — if a model is not here it cannot be selected.
SUPPORTED_MODELS: list[str] = [
    "qwen2.5vl:7b",
    "llama3.2-vision:11b",
    "gemini-2.0-flash",
]

# Default model used when none is specified (e.g., in scripts or tests)
DEFAULT_MODEL = "gemini-2.0-flash"


class ModelRegistry:
    """
    Holds a dictionary of model name → VLMClient instance.

    Clients are created lazily on first access so we don't fail at startup if,
    for example, the Gemini API key is missing but the user only wants Ollama.

    Example usage:
        registry = ModelRegistry.from_env()
        client = registry.get("qwen2.5vl:7b")
        response = client.generate(image, prompt)
    """

    def __init__(self) -> None:
        # Internal cache: populated lazily when get() is first called
        self._clients: dict[str, VLMClient] = {}

    @classmethod
    def from_env(cls) -> "ModelRegistry":
        """
        Create a registry pre-loaded with all models that can be initialised
        from the current environment (API keys, Ollama URL, etc.).

        Models that fail to initialise (e.g., missing API key) are logged as
        warnings but do not prevent the registry from being created — so the
        app can still start even if one backend is unavailable.

        Returns:
            A ModelRegistry instance. Call .available_models() to see what loaded.
        """
        registry = cls()
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Try to register Ollama models (no key needed, just needs server running)
        for ollama_model in ["qwen2.5vl:7b", "llama3.2-vision:11b"]:
            try:
                registry._clients[ollama_model] = OllamaClient(
                    model=ollama_model, base_url=ollama_base_url
                )
                logger.info("Registered Ollama model: %s at %s", ollama_model, ollama_base_url)
            except Exception as e:
                logger.warning("Could not register Ollama model %s: %s", ollama_model, e)

        # Try to register Gemini (needs GEMINI_API_KEY)
        try:
            registry._clients["gemini-2.0-flash"] = GeminiClient()
            logger.info("Registered Gemini model: gemini-2.0-flash")
        except ValueError as e:
            # ValueError means missing API key — expected in environments without a key
            logger.warning("Could not register Gemini: %s", e)

        return registry

    def get(self, model_name: str) -> VLMClient:
        """
        Retrieve a VLMClient by model name.

        Args:
            model_name: One of the strings in SUPPORTED_MODELS.

        Returns:
            The corresponding VLMClient instance.

        Raises:
            KeyError: If the model name is not registered or failed to initialise.
        """
        if model_name not in self._clients:
            raise KeyError(
                f"Model '{model_name}' is not available. "
                f"Available models: {list(self._clients.keys())}. "
                "For Ollama models, ensure the server is running and the model is pulled. "
                "For Gemini, ensure GEMINI_API_KEY is set in your .env file."
            )
        return self._clients[model_name]

    def available_models(self) -> list[str]:
        """
        Return the names of all currently registered (available) models.

        Returns:
            List of model name strings, in the order they were registered.
        """
        return list(self._clients.keys())

    def all_supported_models(self) -> list[str]:
        """
        Return all model names in the supported list (registered or not).

        Useful for the UI dropdown — shows all options even if some are offline.

        Returns:
            List of all model name strings from SUPPORTED_MODELS.
        """
        return SUPPORTED_MODELS
