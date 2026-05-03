"""Unit tests for OpenRouterVLMClient."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from models.base import VLMResponse
from models.openrouter_client import OpenRouterVLMClient


def _img() -> Image.Image:
    return Image.new("RGB", (8, 8), color="red")


def test_constructor_uses_explicit_api_key():
    client = OpenRouterVLMClient(model="openai/gpt-4o-mini", api_key="explicit-key")
    assert client.model_name == "openai/gpt-4o-mini"


def test_constructor_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    client = OpenRouterVLMClient(model="openai/gpt-4o-mini")
    assert client.model_name == "openai/gpt-4o-mini"


def test_constructor_without_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenRouterVLMClient(model="openai/gpt-4o-mini")


def test_generate_returns_vlm_response(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    fake_choice = MagicMock()
    fake_choice.message.content = '{"room_type":"kitchen","amenities":{}}'
    fake_completion = MagicMock(choices=[fake_choice])

    with patch("models.openrouter_client.OpenAI") as openai_cls:
        openai_cls.return_value.chat.completions.create.return_value = fake_completion
        client = OpenRouterVLMClient(model="openai/gpt-4o-mini")
        out = client.generate(_img(), "prompt")

    assert isinstance(out, VLMResponse)
    assert out.raw_text.startswith("{")
    assert out.model_name == "openai/gpt-4o-mini"


def test_json_mode_true_sends_response_format(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    with patch("models.openrouter_client.OpenAI") as openai_cls:
        completions = openai_cls.return_value.chat.completions
        completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=""))]
        )
        client = OpenRouterVLMClient(model="openai/gpt-4o-mini", json_mode=True)
        client.generate(_img(), "Return a JSON object.")

    kwargs = completions.create.call_args.kwargs
    assert kwargs.get("response_format") == {"type": "json_object"}


def test_json_mode_true_omits_response_format_for_non_json_prompt(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    with patch("models.openrouter_client.OpenAI") as openai_cls:
        completions = openai_cls.return_value.chat.completions
        completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=""))]
        )
        client = OpenRouterVLMClient(model="openai/gpt-4o-mini", json_mode=True)
        client.generate(_img(), "Write a warm property description.")

    kwargs = completions.create.call_args.kwargs
    assert "response_format" not in kwargs


def test_json_mode_false_omits_response_format(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    with patch("models.openrouter_client.OpenAI") as openai_cls:
        completions = openai_cls.return_value.chat.completions
        completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=""))]
        )
        client = OpenRouterVLMClient(model="qwen/qwen2-vl-72b-instruct", json_mode=False)
        client.generate(_img(), "p")

    kwargs = completions.create.call_args.kwargs
    assert "response_format" not in kwargs


def test_sdk_error_becomes_runtime_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    with patch("models.openrouter_client.OpenAI") as openai_cls:
        openai_cls.return_value.chat.completions.create.side_effect = Exception("boom")
        client = OpenRouterVLMClient(model="openai/gpt-4o-mini")
        with pytest.raises(RuntimeError):
            client.generate(_img(), "p")


def test_generate_sends_image_as_data_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    with patch("models.openrouter_client.OpenAI") as openai_cls:
        completions = openai_cls.return_value.chat.completions
        completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=""))]
        )
        OpenRouterVLMClient(model="openai/gpt-4o-mini").generate(_img(), "p")
    content = completions.create.call_args.kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": "p"}
