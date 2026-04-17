"""
Abstract base classes for all VLM (Vision-Language Model) clients.

Why an abstract base class?
  The rest of the codebase (AmenityDetector, FastAPI endpoints) should not care
  which model is running. They call `client.generate(image, prompt)` and get back
  a VLMResponse. Swapping models becomes a config change, not a code change.

How to add a new model:
  1. Create a new file (e.g., models/my_new_client.py)
  2. Subclass VLMClient and implement `generate()`
  3. Register it in models/registry.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL.Image import Image


@dataclass
class VLMResponse:
    """
    Standardised response returned by every VLM client.

    Attributes:
        raw_text:   The raw text output from the model (not yet parsed).
        model_name: The model identifier that produced this response
                    (e.g., "qwen2.5vl:7b" or "gemini-2.0-flash").
    """

    raw_text: str
    model_name: str


class VLMClient(ABC):
    """
    Abstract base class that every VLM backend must implement.

    Concrete implementations live in:
      - models/ollama_client.py  (Qwen2.5-VL-7B, LLaMA 3.2 Vision via Ollama)
      - models/gemini_client.py  (Gemini 2.0 Flash via Google API)
    """

    @abstractmethod
    def generate(self, image: Image, prompt: str) -> VLMResponse:
        """
        Send an image + text prompt to the model and return the response.

        Args:
            image:  A PIL Image object. The client is responsible for converting
                    it to whatever format the backend expects (e.g., base64 bytes).
            prompt: The text instruction — e.g., "List all visible amenities as JSON."

        Returns:
            A VLMResponse containing the model's raw text output and model name.

        Raises:
            RuntimeError: If the backend is unreachable or returns an error.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        The canonical model identifier string (e.g., "qwen2.5vl:7b").

        Used by ModelRegistry and stored in the database alongside results
        so you always know which model produced a given detection.
        """
        ...
