"""
Unit tests for models/gemini_client.py.

We mock the google-genai SDK (Client) so these tests:
  - Run without a real Gemini API key
  - Run in CI without network access
  - Are fast (no actual HTTP calls)

The key things we verify:
  1. The client refuses to initialise without an API key
  2. generate() calls the SDK and returns a VLMResponse
  3. SDK exceptions are re-raised as RuntimeError (not leaked as Google exceptions)
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from models.base import VLMResponse
from models.gemini_client import GEMINI_MODEL_ID, GeminiClient


@pytest.fixture
def fake_image() -> PILImage.Image:
    """10x10 test image — same helper as in test_ollama_client.py."""
    return PILImage.new("RGB", (10, 10), color=(0, 255, 0))


class TestGeminiClientInit:
    def test_raises_if_no_api_key(self, monkeypatch):
        """
        GeminiClient should raise ValueError early if no API key is available.
        This prevents confusing errors later when generate() is called.
        """
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with patch("models.gemini_client.genai.Client"):
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                GeminiClient(api_key=None)

    def test_reads_api_key_from_env(self, monkeypatch):
        """
        If no explicit key is given, it should read GEMINI_API_KEY from the environment.
        """
        monkeypatch.setenv("GEMINI_API_KEY", "env-test-key")

        with patch("models.gemini_client.genai.Client") as MockClient:
            client = GeminiClient()
            MockClient.assert_called_once_with(api_key="env-test-key")
            assert client is not None

    def test_explicit_key_overrides_env(self, monkeypatch):
        """An explicitly passed key should take precedence over the environment."""
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")

        with patch("models.gemini_client.genai.Client") as MockClient:
            GeminiClient(api_key="explicit-key")
            MockClient.assert_called_once_with(api_key="explicit-key")


class TestGeminiClientGenerate:
    @pytest.fixture
    def client(self, monkeypatch) -> GeminiClient:
        """
        GeminiClient with the SDK Client fully mocked.
        """
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("models.gemini_client.genai.Client") as MockClient:
            client = GeminiClient()
            # The underlying _client is the mock instance
            client._client = MockClient.return_value
        return client

    def test_model_name_property(self, client: GeminiClient):
        assert client.model_name == GEMINI_MODEL_ID

    def test_successful_generate_returns_vlm_response(
        self, client: GeminiClient, fake_image: PILImage.Image
    ):
        """A successful API call should return a VLMResponse with the model's text."""
        mock_api_response = MagicMock()
        mock_api_response.text = '{"pool": true, "gym": false}'
        client._client.models.generate_content.return_value = mock_api_response

        result = client.generate(fake_image, "List visible amenities as JSON.")

        assert isinstance(result, VLMResponse)
        assert result.raw_text == '{"pool": true, "gym": false}'
        assert result.model_name == GEMINI_MODEL_ID

    def test_passes_image_and_prompt_to_sdk(self, client: GeminiClient, fake_image: PILImage.Image):
        """The SDK's generate_content() should receive both the image and the prompt."""
        mock_api_response = MagicMock()
        mock_api_response.text = "ok"
        client._client.models.generate_content.return_value = mock_api_response

        client.generate(fake_image, "describe the room")

        call_kwargs = client._client.models.generate_content.call_args
        contents = call_kwargs.kwargs["contents"]
        # The contents list should contain [image, prompt_text]
        assert fake_image in contents
        assert "describe the room" in contents

    def test_sdk_exception_wrapped_as_runtime_error(
        self, client: GeminiClient, fake_image: PILImage.Image
    ):
        """
        Any exception from the Gemini SDK should be caught and re-raised as
        RuntimeError so callers don't need to import Google's exception classes.
        """
        client._client.models.generate_content.side_effect = Exception("quota exceeded")

        with pytest.raises(RuntimeError, match="Gemini API call failed"):
            client.generate(fake_image, "test")
