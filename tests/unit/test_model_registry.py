"""
Unit tests for models/registry.py — ModelRegistry.

We mock out the actual VLM clients so these tests:
  - Run without Ollama or a Gemini API key
  - Verify the registry's logic (not the clients themselves)

Key things verified:
  1. from_env() creates client entries for available models
  2. get() returns the right client
  3. get() raises KeyError for unknown/unavailable models
  4. available_models() lists only registered models
  5. Registry survives if some models fail to initialise
"""

from unittest.mock import MagicMock, patch

import pytest

from models.base import VLMClient, VLMResponse
from models.registry import ModelRegistry, SUPPORTED_MODELS


def _make_fake_client(name: str) -> VLMClient:
    """Helper to create a mock VLMClient with a given model_name."""
    client = MagicMock(spec=VLMClient)
    type(client).model_name = property(lambda self: name)
    return client


class TestModelRegistryBasic:
    def test_empty_registry_has_no_models(self):
        registry = ModelRegistry()
        assert registry.available_models() == []

    def test_get_unknown_model_raises_key_error(self):
        registry = ModelRegistry()
        with pytest.raises(KeyError, match="not available"):
            registry.get("nonexistent-model")

    def test_get_returns_registered_client(self):
        registry = ModelRegistry()
        fake_client = _make_fake_client("test-model")
        registry._clients["test-model"] = fake_client

        result = registry.get("test-model")
        assert result is fake_client

    def test_available_models_returns_registered_names(self):
        registry = ModelRegistry()
        registry._clients["model-a"] = _make_fake_client("model-a")
        registry._clients["model-b"] = _make_fake_client("model-b")

        available = registry.available_models()
        assert "model-a" in available
        assert "model-b" in available

    def test_all_supported_models_returns_known_list(self):
        registry = ModelRegistry()
        supported = registry.all_supported_models()
        assert supported == SUPPORTED_MODELS  # Must match the hardcoded list


class TestModelRegistryFromEnv:
    def test_from_env_registers_ollama_models(self, monkeypatch):
        """When OllamaClient initialises without error, both models should be registered."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://fake-ollama:11434")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with patch("models.registry.OllamaClient") as MockOllama, patch(
            "models.registry.GeminiClient", side_effect=ValueError("no key")
        ):
            MockOllama.return_value = _make_fake_client("qwen2.5vl:7b")
            registry = ModelRegistry.from_env()

        # Both Ollama models should have been attempted
        assert MockOllama.call_count == 2

    def test_from_env_registers_gemini_when_key_present(self, monkeypatch):
        """When GEMINI_API_KEY is set, Gemini should be registered."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://fake-ollama:11434")

        with patch("models.registry.OllamaClient") as MockOllama, patch(
            "models.registry.GeminiClient"
        ) as MockGemini:
            MockOllama.return_value = _make_fake_client("ollama-model")
            MockGemini.return_value = _make_fake_client("gemini-2.0-flash")
            registry = ModelRegistry.from_env()

        MockGemini.assert_called_once()

    def test_from_env_survives_missing_gemini_key(self, monkeypatch):
        """
        If GEMINI_API_KEY is absent, from_env() should NOT raise — just skip Gemini.
        This allows the app to start in Ollama-only mode.
        """
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with patch("models.registry.OllamaClient") as MockOllama, patch(
            "models.registry.GeminiClient", side_effect=ValueError("no key")
        ):
            MockOllama.return_value = _make_fake_client("ollama")
            # This should NOT raise
            registry = ModelRegistry.from_env()

        assert "gemini-2.0-flash" not in registry.available_models()

    def test_from_env_survives_ollama_failure(self, monkeypatch):
        """
        If OllamaClient raises (e.g., bad URL), from_env() should skip it and continue.
        Gemini should still register if the API key is present.
        """
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("models.registry.OllamaClient", side_effect=Exception("bad init")), patch(
            "models.registry.GeminiClient"
        ) as MockGemini:
            MockGemini.return_value = _make_fake_client("gemini-2.0-flash")
            registry = ModelRegistry.from_env()

        assert "gemini-2.0-flash" in registry.available_models()
