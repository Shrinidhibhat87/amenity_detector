"""Thin wrapper around the OpenAI embeddings API.

The wrapper exists for three reasons:

  1. The production deployment routes through an OpenAI-compatible LiteLLM
     proxy with its own ``base_url`` + key — separate from OpenRouter (which
     handles chat/VLM calls). Centralising that here keeps the rest of the
     codebase from caring.
  2. We need ``embed_text`` and ``embed_batch`` for both the indexing pipeline
     (one row at a time on writes) and the backfill script (batched).
  3. SDK errors should bubble up as a project-specific ``EmbeddingsError`` so
     callers can catch one type instead of every possible OpenAI exception.

Configuration is via environment variables:

  - ``EMBEDDINGS_BASE_URL`` — OpenAI-compatible URL ending in ``/v1``.
  - ``EMBEDDINGS_API_KEY``  — auth token sent as ``Authorization: Bearer``.
  - ``EMBEDDINGS_MODEL``    — usually ``text-embedding-3-small`` (1536 dims).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Self

from openai import OpenAI

logger = logging.getLogger(__name__)


class EmbeddingsError(RuntimeError):
    """Raised when the embeddings client cannot return a usable vector."""


@dataclass(slots=True)
class _Config:
    base_url: str
    api_key: str
    model: str


class EmbeddingsClient:
    """OpenAI-compatible embeddings client (LiteLLM-proxied in production).

    Construct directly when you know the config or call :meth:`from_env` to
    pull from ``EMBEDDINGS_*`` environment variables. Either way the heavy
    lifting is delegated to the ``openai`` SDK.
    """

    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    @classmethod
    def from_env(cls) -> Self:
        cfg = _read_env()
        return cls(base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string. Whitespace is stripped; empty input errors."""
        cleaned = text.strip()
        if not cleaned:
            raise EmbeddingsError("cannot embed empty/whitespace-only text")

        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=[cleaned],
            )
        except Exception as exc:  # SDK raises many specific types; wrap them all.
            raise EmbeddingsError(f"embedding request failed: {exc}") from exc

        try:
            return list(response.data[0].embedding)
        except (AttributeError, IndexError) as exc:
            raise EmbeddingsError(f"unexpected response shape: {exc}") from exc

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings in a single API call.

        Blank entries (empty / whitespace-only) are returned as ``[]`` so the
        caller can map the result back to the original positions without
        having to track which slots were skipped. This matters for the
        backfill script which iterates over rows that may have NULL/empty
        descriptions and must keep its index aligned.
        """
        if not texts:
            return []

        non_blank: list[tuple[int, str]] = [
            (i, t.strip()) for i, t in enumerate(texts) if t and t.strip()
        ]
        if not non_blank:
            return [[] for _ in texts]

        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=[t for _, t in non_blank],
            )
        except Exception as exc:
            raise EmbeddingsError(f"batch embedding request failed: {exc}") from exc

        results: list[list[float]] = [[] for _ in texts]
        for (idx, _), item in zip(non_blank, response.data, strict=True):
            results[idx] = list(item.embedding)
        return results


def _read_env() -> _Config:
    missing: list[str] = []
    base_url = os.getenv("EMBEDDINGS_BASE_URL")
    api_key = os.getenv("EMBEDDINGS_API_KEY")
    model = os.getenv("EMBEDDINGS_MODEL")
    if not base_url:
        missing.append("EMBEDDINGS_BASE_URL")
    if not api_key:
        missing.append("EMBEDDINGS_API_KEY")
    if not model:
        missing.append("EMBEDDINGS_MODEL")
    if missing:
        raise EmbeddingsError(
            f"missing required env vars: {', '.join(missing)}. See .env.example for the full list."
        )
    # mypy: each of the three is non-None after the guard above.
    assert base_url is not None and api_key is not None and model is not None
    return _Config(base_url=base_url, api_key=api_key, model=model)
