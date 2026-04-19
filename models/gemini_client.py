"""
Gemini VLM client — wraps the Google GenAI SDK.

Why Gemini 2.0 Flash?
  It's free up to 1500 requests/day, requires no local GPU, and is the fastest
  option for testing the pipeline end-to-end without Ollama running. It's also
  useful as a quality baseline to compare against local models.

SDK used: google-genai (the new official SDK, replacing the deprecated google-generativeai)
  The new SDK is at google.genai and uses a Client-based approach.

Environment variables:
  GEMINI_API_KEY — required; obtain from https://aistudio.google.com/

Model used: gemini-2.0-flash
"""

import logging
import os

from google import genai
from PIL.Image import Image

from models.base import VLMClient, VLMResponse

logger = logging.getLogger(__name__)

# The model name to use with the Gemini API.
GEMINI_MODEL_ID = "gemini-2.0-flash"


class GeminiClient(VLMClient):
    """
    Calls the Gemini 2.0 Flash API via Google's genai SDK.

    The client creates a genai.Client on initialisation (validates the API key).
    All calls go to Google's servers — no local GPU required.

    Note on free tier: 1500 requests/day, 15 requests/minute as of April 2025.
    Uploading multiple images for a single property counts as multiple requests.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialise the Gemini client.

        Args:
            api_key: Your Gemini API key. If not provided, reads from the
                     GEMINI_API_KEY environment variable. Raises ValueError
                     if neither is set, so the error surfaces early.

        Raises:
            ValueError: If no API key is available.
        """
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Gemini API key not found. Set the GEMINI_API_KEY environment variable "
                "or pass it explicitly to GeminiClient(api_key=...)."
            )

        # The new google-genai SDK uses a Client object.
        # API key is passed here — not stored on self to avoid accidental logging.
        # http_options timeout is in milliseconds; 90 s is generous for a single
        # image call while still failing clearly if the API hangs or quota throttles.
        self._client = genai.Client(
            api_key=resolved_key,
            http_options={"timeout": 90_000},
        )

    @property
    def model_name(self) -> str:
        """The Gemini model identifier."""
        return GEMINI_MODEL_ID

    def generate(self, image: Image, prompt: str) -> VLMResponse:
        """
        Send an image + prompt to Gemini and return the text response.

        The new google-genai SDK accepts PIL Image objects in the contents list.
        We put the image first so the model sees it before the instruction.

        Args:
            image:  PIL Image to analyse.
            prompt: Text instruction for the model.

        Returns:
            VLMResponse with Gemini's text output and the model name.

        Raises:
            RuntimeError: If the API call fails (network error, quota exceeded, etc.)
        """
        try:
            # The new SDK: client.models.generate_content() with a mixed-content list
            response = self._client.models.generate_content(
                model=GEMINI_MODEL_ID,
                contents=[image, prompt],  # type: ignore[arg-type]
            )
            raw_text: str = response.text or ""
        except Exception as e:
            # Catch all SDK exceptions and re-raise as RuntimeError so callers
            # don't need to import Google's exception hierarchy.
            raise RuntimeError(f"Gemini API call failed: {e}") from e

        logger.debug("Gemini response: %s", raw_text[:200])
        return VLMResponse(raw_text=raw_text, model_name=GEMINI_MODEL_ID)
