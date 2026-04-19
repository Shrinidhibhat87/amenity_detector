"""
Ollama VLM client — wraps the Ollama REST API for local model inference.

Why Ollama?
  Ollama handles model downloading, 4-bit quantization, and serving behind a
  simple HTTP API. We don't need to manage GPU memory or HuggingFace weights
  ourselves — Ollama does it all. The unified endpoint means Qwen2.5-VL-7B and
  LLaMA 3.2 Vision are accessed identically.

Endpoint used: POST /api/chat  (preferred over /api/generate for message history)

Environment variables:
  OLLAMA_BASE_URL — defaults to http://localhost:11434
                    In Docker Compose this becomes http://ollama:11434
"""

import base64
import io
import logging
import os

import requests
from PIL.Image import Image

from models.base import VLMClient, VLMResponse

logger = logging.getLogger(__name__)


class OllamaClient(VLMClient):
    """
    Calls a locally running Ollama server to run vision-language inference.

    Ollama must be running with the target model already pulled, e.g.:
        ollama pull qwen2.5vl:7b
        ollama pull llama3.2-vision:11b
    """

    def __init__(self, model: str, base_url: str | None = None) -> None:
        """
        Initialise the Ollama client.

        Args:
            model:    The Ollama model tag to use, e.g. "qwen2.5vl:7b".
                      Must match exactly what `ollama list` shows.
            base_url: The Ollama server URL. Defaults to the OLLAMA_BASE_URL
                      environment variable, then http://localhost:11434.
        """
        self._model = model
        self._base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        ).rstrip("/")

    @property
    def model_name(self) -> str:
        """The Ollama model tag used by this client instance."""
        return self._model

    def _image_to_base64(self, image: Image) -> str:
        """
        Convert a PIL Image to a base64-encoded JPEG string.

        Ollama's /api/chat endpoint expects images as a list of base64 strings.
        We use JPEG (not PNG) to reduce payload size — quality 85 is a good
        balance between fidelity and network speed for local inference.

        Args:
            image: A PIL Image object in any mode.

        Returns:
            A base64-encoded string (no newlines, not a data URI).
        """
        buffer = io.BytesIO()
        # Convert to RGB first — JPEG doesn't support transparency (RGBA/P modes)
        image.convert("RGB").save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def generate(self, image: Image, prompt: str) -> VLMResponse:
        """
        Send an image + prompt to Ollama and return the text response.

        Uses the /api/chat endpoint with stream=false so we get a single JSON
        response rather than a stream of tokens. This keeps the code simple at
        the cost of not being able to show a progress indicator.

        Args:
            image:  PIL Image to analyse.
            prompt: Text instruction for the model.

        Returns:
            VLMResponse with the model's text output and the model name.

        Raises:
            RuntimeError: If Ollama returns a non-200 status or is unreachable.
        """
        img_b64 = self._image_to_base64(image)

        # num_gpu=-1 tells Ollama to offload as many layers as possible to GPU.
        # On a 6GB card like GTX 1660 Ti the full 7B model won't fit, but Ollama
        # will still place partial layers on GPU, which is faster than pure CPU.
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    # Ollama's chat API accepts images as a list of base64 strings
                    "images": [img_b64],
                }
            ],
            "stream": False,  # Wait for the full response before returning
            "options": {"num_gpu": -1},  # use GPU layers where VRAM allows
        }

        # 600s timeout: when the model falls back to CPU (e.g. VRAM too small for
        # all layers), inference takes 3–5 minutes. 10 minutes gives enough headroom.
        _timeout = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))

        try:
            response = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=_timeout,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Is the Ollama server running? Try: OLLAMA_HOST=0.0.0.0 ollama serve"
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                f"Ollama request timed out after {_timeout}s for model '{self._model}'. "
                "The model may be running on CPU — this is slow on low-VRAM hardware. "
                "Try a smaller model or increase OLLAMA_TIMEOUT_SECONDS."
            ) from e
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code}: {response.text}"
            ) from e

        data = response.json()

        # The chat response nests the text inside message.content
        raw_text: str = data.get("message", {}).get("content", "")
        logger.debug("Ollama response for model '%s': %s", self._model, raw_text[:200])

        return VLMResponse(raw_text=raw_text, model_name=self._model)
