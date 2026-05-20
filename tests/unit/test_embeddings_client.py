"""Unit tests for ``core.embeddings.EmbeddingsClient``.

The client wraps the OpenAI SDK pointed at a custom base URL (the user's
LiteLLM proxy in production). Tests mock the ``OpenAI`` constructor so they
never hit the network and run on any machine.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.embeddings import EmbeddingsClient, EmbeddingsError


def _fake_response(*vectors: list[float]) -> Any:
    """Build a fake OpenAI embeddings.create() response.

    The SDK returns an object whose ``.data`` is a list of items each with an
    ``.embedding`` attribute. We mimic that with simple namespaces so the
    client's ``[item.embedding for item in resp.data]`` walk works unchanged.
    """
    items = [MagicMock(embedding=v) for v in vectors]
    resp = MagicMock()
    resp.data = items
    return resp


class TestEmbeddingsClientConstruction:
    def test_explicit_config_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDINGS_BASE_URL", "https://env.example/v1")
        monkeypatch.setenv("EMBEDDINGS_API_KEY", "env-key")
        monkeypatch.setenv("EMBEDDINGS_MODEL", "env-model")

        with patch("core.embeddings.OpenAI") as openai_ctor:
            EmbeddingsClient(
                base_url="https://explicit.example/v1",
                api_key="explicit-key",
                model="explicit-model",
            )

        openai_ctor.assert_called_once_with(
            base_url="https://explicit.example/v1",
            api_key="explicit-key",
        )

    def test_from_env_reads_required_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDINGS_BASE_URL", "https://litellm.example/v1")
        monkeypatch.setenv("EMBEDDINGS_API_KEY", "sk-litellm")
        monkeypatch.setenv("EMBEDDINGS_MODEL", "text-embedding-3-small")

        with patch("core.embeddings.OpenAI") as openai_ctor:
            client = EmbeddingsClient.from_env()

        openai_ctor.assert_called_once_with(
            base_url="https://litellm.example/v1",
            api_key="sk-litellm",
        )
        assert client.model == "text-embedding-3-small"

    def test_from_env_raises_when_required_vars_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EMBEDDINGS_BASE_URL", raising=False)
        monkeypatch.delenv("EMBEDDINGS_API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDINGS_MODEL", raising=False)

        with pytest.raises(EmbeddingsError, match="EMBEDDINGS_"):
            EmbeddingsClient.from_env()


class TestEmbedText:
    def test_embed_text_returns_vector(self) -> None:
        with patch("core.embeddings.OpenAI") as openai_ctor:
            openai_ctor.return_value.embeddings.create.return_value = _fake_response(
                [0.1, 0.2, 0.3]
            )
            client = EmbeddingsClient(base_url="x", api_key="x", model="m")

            vec = client.embed_text("a kitchen with a fireplace")

        assert vec == [0.1, 0.2, 0.3]
        openai_ctor.return_value.embeddings.create.assert_called_once_with(
            model="m",
            input=["a kitchen with a fireplace"],
        )

    def test_embed_text_strips_and_rejects_empty(self) -> None:
        with patch("core.embeddings.OpenAI"):
            client = EmbeddingsClient(base_url="x", api_key="x", model="m")
            with pytest.raises(EmbeddingsError, match="empty"):
                client.embed_text("   ")

    def test_embed_text_wraps_sdk_errors(self) -> None:
        with patch("core.embeddings.OpenAI") as openai_ctor:
            openai_ctor.return_value.embeddings.create.side_effect = RuntimeError("boom")
            client = EmbeddingsClient(base_url="x", api_key="x", model="m")
            with pytest.raises(EmbeddingsError, match="boom"):
                client.embed_text("hello")


class TestEmbedBatch:
    def test_embed_batch_single_call_preserves_order(self) -> None:
        with patch("core.embeddings.OpenAI") as openai_ctor:
            openai_ctor.return_value.embeddings.create.return_value = _fake_response(
                [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]
            )
            client = EmbeddingsClient(base_url="x", api_key="x", model="m")

            vecs = client.embed_batch(["a", "b", "c"])

        assert vecs == [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        openai_ctor.return_value.embeddings.create.assert_called_once_with(
            model="m",
            input=["a", "b", "c"],
        )

    def test_embed_batch_empty_input_short_circuits(self) -> None:
        with patch("core.embeddings.OpenAI") as openai_ctor:
            client = EmbeddingsClient(base_url="x", api_key="x", model="m")
            assert client.embed_batch([]) == []
            openai_ctor.return_value.embeddings.create.assert_not_called()

    def test_embed_batch_skips_blank_strings(self) -> None:
        with patch("core.embeddings.OpenAI") as openai_ctor:
            openai_ctor.return_value.embeddings.create.return_value = _fake_response(
                [0.1], [0.2]
            )
            client = EmbeddingsClient(base_url="x", api_key="x", model="m")
            vecs = client.embed_batch(["alpha", "   ", "beta"])

        assert vecs == [[0.1], [], [0.2]]
        openai_ctor.return_value.embeddings.create.assert_called_once_with(
            model="m",
            input=["alpha", "beta"],
        )
