"""
OpenRouter VLM client.

Uses the openai SDK against https://openrouter.ai/api/v1 so every supported
model — GPT-4o-mini, Gemini Pro 1.5, Llama 3.2 Vision, Qwen2-VL — is reached
through the same OpenAI-compatible chat completions endpoint.

Environment variables:
  OPENROUTER_API_KEY — required. Obtain at https://openrouter.ai/.
"""

from __future__ import annotations

import base64
import io
import logging
import os

from openai import OpenAI
from PIL.Image import Image

from models.base import VLMClient, VLMResponse

logger = logging.getLogger(__name__)

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterVLMClient(VLMClient):
    """Single OpenRouter-backed client used for every registered model."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        json_mode: bool = False,
    ) -> None:
        resolved = api_key or os.getenv("OPENROUTER_API_KEY")
        if not resolved:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY in the environment "
                "or pass api_key=... to OpenRouterVLMClient."
            )
        self._model = model
        self._json_mode = json_mode
        self._client = OpenAI(api_key=resolved, base_url=_BASE_URL)

    @property
    def model_name(self) -> str:
        return self._model

    def _image_to_data_url(self, image: Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def generate(self, image: Image, prompt: str) -> VLMResponse:
        data_url = self._image_to_data_url(image)
        kwargs: dict[str, object] = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            completion = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise RuntimeError(f"OpenRouter call failed for '{self._model}': {e}") from e

        raw = completion.choices[0].message.content or ""
        logger.debug("OpenRouter response for '%s': %s", self._model, raw[:200])
        return VLMResponse(raw_text=raw, model_name=self._model)
