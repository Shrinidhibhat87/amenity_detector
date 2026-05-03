"""Unit tests for ModelRegistry."""

import pytest

from models.openrouter_client import OpenRouterVLMClient
from models.registry import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    ModelRegistry,
)


def test_available_models_lists_four_ids():
    assert {m["id"] for m in AVAILABLE_MODELS} == {
        "openai/gpt-4o-mini",
        "google/gemini-pro-1.5",
        "meta-llama/llama-3.2-11b-vision-instruct",
        "qwen/qwen2-vl-72b-instruct",
    }


def test_supported_models_matches_available():
    assert SUPPORTED_MODELS == [m["id"] for m in AVAILABLE_MODELS]


def test_default_model_is_gpt_4o_mini():
    assert DEFAULT_MODEL == "openai/gpt-4o-mini"


def test_get_returns_openrouter_client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    registry = ModelRegistry.from_env()
    client = registry.get("openai/gpt-4o-mini")
    assert isinstance(client, OpenRouterVLMClient)
    assert client.model_name == "openai/gpt-4o-mini"


def test_get_unknown_id_raises_key_error():
    registry = ModelRegistry.from_env()
    with pytest.raises(KeyError):
        registry.get("not/a-real-model")


def test_available_and_all_supported_are_full_set():
    registry = ModelRegistry.from_env()
    assert set(registry.available_models()) == set(SUPPORTED_MODELS)
    assert registry.all_supported_models() == SUPPORTED_MODELS
