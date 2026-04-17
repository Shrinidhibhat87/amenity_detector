"""
Unit tests for models/base.py — VLMClient ABC and VLMResponse dataclass.

These tests verify:
  1. VLMResponse stores its fields correctly
  2. VLMClient cannot be instantiated directly (it's abstract)
  3. A concrete subclass that implements all abstract methods can be instantiated
  4. A concrete subclass that forgets an abstract method raises TypeError
"""

import pytest
from PIL.Image import Image
from unittest.mock import MagicMock

from models.base import VLMClient, VLMResponse


class TestVLMResponse:
    """Tests for the VLMResponse dataclass."""

    def test_stores_fields(self):
        """Fields set at creation should be accessible."""
        resp = VLMResponse(raw_text="hello world", model_name="test-model")
        assert resp.raw_text == "hello world"
        assert resp.model_name == "test-model"

    def test_empty_raw_text_is_valid(self):
        """An empty string is a valid (if unhelpful) model response."""
        resp = VLMResponse(raw_text="", model_name="test-model")
        assert resp.raw_text == ""

    def test_is_a_dataclass(self):
        """VLMResponse should be comparable by value (dataclass equality)."""
        r1 = VLMResponse(raw_text="abc", model_name="m")
        r2 = VLMResponse(raw_text="abc", model_name="m")
        assert r1 == r2

    def test_different_values_not_equal(self):
        """Different raw_text should mean different objects."""
        r1 = VLMResponse(raw_text="a", model_name="m")
        r2 = VLMResponse(raw_text="b", model_name="m")
        assert r1 != r2


class TestVLMClientAbstract:
    """Tests that VLMClient enforces its abstract interface."""

    def test_cannot_instantiate_directly(self):
        """VLMClient is abstract — instantiating it directly should raise TypeError."""
        with pytest.raises(TypeError):
            VLMClient()  # type: ignore[abstract]

    def test_concrete_subclass_without_generate_raises(self):
        """Forgetting to implement `generate` should make instantiation fail."""

        class IncompleteClient(VLMClient):
            # Missing generate() — only implements model_name
            @property
            def model_name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteClient()  # type: ignore[abstract]

    def test_concrete_subclass_without_model_name_raises(self):
        """Forgetting to implement `model_name` should also fail."""

        class IncompleteClient(VLMClient):
            # Missing model_name — only implements generate()
            def generate(self, image: Image, prompt: str) -> VLMResponse:
                return VLMResponse(raw_text="", model_name="x")

        with pytest.raises(TypeError):
            IncompleteClient()  # type: ignore[abstract]

    def test_valid_concrete_subclass_can_be_instantiated(self):
        """A fully implemented subclass should work fine."""

        class FakeClient(VLMClient):
            @property
            def model_name(self) -> str:
                return "fake-model"

            def generate(self, image: Image, prompt: str) -> VLMResponse:
                return VLMResponse(raw_text="fake response", model_name=self.model_name)

        client = FakeClient()
        assert client.model_name == "fake-model"

    def test_generate_returns_vlm_response(self):
        """generate() on a valid subclass should return a VLMResponse."""

        class FakeClient(VLMClient):
            @property
            def model_name(self) -> str:
                return "fake-model"

            def generate(self, image: Image, prompt: str) -> VLMResponse:
                return VLMResponse(raw_text="detected: pool", model_name=self.model_name)

        client = FakeClient()
        mock_image = MagicMock(spec=Image)
        result = client.generate(mock_image, "What amenities are present?")

        assert isinstance(result, VLMResponse)
        assert result.raw_text == "detected: pool"
        assert result.model_name == "fake-model"
