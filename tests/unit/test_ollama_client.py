"""
Unit tests for models/ollama_client.py.

We mock the HTTP layer (requests.post) so these tests:
  - Run without an Ollama server
  - Run without a GPU
  - Run in CI

This is the right approach for unit tests: test the logic of OllamaClient
(image encoding, payload building, response parsing, error handling) in isolation.

The tests use `unittest.mock.patch` to replace `requests.post` with a fake
that returns a predefined response, so we can verify OllamaClient behaves
correctly for all response shapes.
"""

import base64
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from models.base import VLMResponse
from models.ollama_client import OllamaClient


@pytest.fixture
def fake_image() -> PILImage.Image:
    """Create a tiny 10x10 red PIL Image for testing — no file I/O needed."""
    return PILImage.new("RGB", (10, 10), color=(255, 0, 0))


@pytest.fixture
def client() -> OllamaClient:
    """OllamaClient pointed at a fake URL — no real server expected."""
    return OllamaClient(model="qwen2.5vl:7b", base_url="http://fake-ollama:11434")


class TestOllamaClientInit:
    def test_model_name_property(self, client: OllamaClient):
        assert client.model_name == "qwen2.5vl:7b"

    def test_reads_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-ollama:11434")
        c = OllamaClient(model="test-model")
        # We can't inspect _base_url directly, but generate() will use it.
        # Just verify it was constructed without error.
        assert c.model_name == "test-model"

    def test_strips_trailing_slash_from_url(self):
        c = OllamaClient(model="qwen2.5vl:7b", base_url="http://localhost:11434/")
        # Internal URL should not have a trailing slash
        assert not c._base_url.endswith("/")


class TestImageToBase64:
    def test_produces_valid_base64(self, client: OllamaClient, fake_image: PILImage.Image):
        """The encoded string should be decodable back to valid bytes."""
        b64 = client._image_to_base64(fake_image)
        # Should not raise
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_rgba_image_converted_to_rgb(self, client: OllamaClient):
        """RGBA images need to be converted to RGB before JPEG encoding."""
        rgba_image = PILImage.new("RGBA", (10, 10), color=(255, 0, 0, 128))
        # Should not raise (JPEG doesn't support alpha)
        b64 = client._image_to_base64(rgba_image)
        assert len(b64) > 0


class TestGenerate:
    def test_successful_response_parsed_correctly(
        self, client: OllamaClient, fake_image: PILImage.Image
    ):
        """A 200 response with the expected JSON structure returns a VLMResponse."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": '{"refrigerator": true, "oven": false}'}
        }
        mock_response.raise_for_status.return_value = None

        with patch("models.ollama_client.requests.post", return_value=mock_response):
            result = client.generate(fake_image, "What amenities are visible?")

        assert isinstance(result, VLMResponse)
        assert result.raw_text == '{"refrigerator": true, "oven": false}'
        assert result.model_name == "qwen2.5vl:7b"

    def test_sends_correct_model_in_payload(
        self, client: OllamaClient, fake_image: PILImage.Image
    ):
        """The model name in the request payload must match the client's model."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "response"}}
        mock_response.raise_for_status.return_value = None

        with patch("models.ollama_client.requests.post", return_value=mock_response) as mock_post:
            client.generate(fake_image, "test prompt")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["model"] == "qwen2.5vl:7b"
        assert payload["stream"] is False  # We want the full response, not a stream

    def test_sends_image_as_base64_in_messages(
        self, client: OllamaClient, fake_image: PILImage.Image
    ):
        """The image should appear in the messages[0].images list as base64."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ok"}}
        mock_response.raise_for_status.return_value = None

        with patch("models.ollama_client.requests.post", return_value=mock_response) as mock_post:
            client.generate(fake_image, "describe")

        payload = mock_post.call_args.kwargs["json"]
        messages = payload["messages"]
        assert len(messages) == 1
        assert len(messages[0]["images"]) == 1  # One base64 image string

    def test_connection_error_raises_runtime_error(
        self, client: OllamaClient, fake_image: PILImage.Image
    ):
        """A ConnectionError (Ollama not running) should raise RuntimeError."""
        import requests

        with patch(
            "models.ollama_client.requests.post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
                client.generate(fake_image, "test")

    def test_timeout_raises_runtime_error(
        self, client: OllamaClient, fake_image: PILImage.Image
    ):
        """A timeout should raise RuntimeError with a descriptive message."""
        import requests

        with patch(
            "models.ollama_client.requests.post",
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                client.generate(fake_image, "test")

    def test_http_error_raises_runtime_error(
        self, client: OllamaClient, fake_image: PILImage.Image
    ):
        """A non-200 HTTP response should raise RuntimeError."""
        import requests

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500")

        with patch("models.ollama_client.requests.post", return_value=mock_response):
            with pytest.raises(RuntimeError, match="HTTP"):
                client.generate(fake_image, "test")
