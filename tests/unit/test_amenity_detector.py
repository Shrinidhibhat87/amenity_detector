"""
Unit tests for core/amenity_detector.py.

These tests cover:
  - Prompt building (detection prompt, description prompt)
  - JSON parsing from VLM output:
      * New structured format  {"present": bool, "confidence": float}
      * Legacy boolean format  {amenity: bool}
      * Markdown code-block wrapping
      * Malformed / missing JSON (graceful fallback)
  - Full detect_from_image() flow (mocked VLM, checks return structure)
  - generate_description() (mocked VLM)
  - Error handling: VLM RuntimeError does not propagate

All VLM calls are mocked so these tests run without Ollama or a Gemini key.
"""

from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage

from core.amenity_detector import AmenityDetector
from models.base import VLMClient, VLMResponse

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def amenity_schema() -> dict[str, list[str]]:
    """A minimal amenity schema covering two room types."""
    return {
        "kitchen": ["refrigerator", "oven", "dishwasher"],
        "bedroom": ["bed", "wardrobe"],
    }


@pytest.fixture
def fake_vlm() -> MagicMock:
    """A mock VLMClient that can be configured per test."""
    mock = MagicMock(spec=VLMClient)
    type(mock).model_name = property(lambda self: "fake-model")
    return mock


@pytest.fixture
def detector(amenity_schema: dict[str, list[str]], fake_vlm: MagicMock) -> AmenityDetector:
    """An AmenityDetector wired to the fake VLM and test schema."""
    return AmenityDetector(vlm_client=fake_vlm, amenity_schema=amenity_schema)


@pytest.fixture
def small_image() -> PILImage.Image:
    """A tiny PIL Image — no file I/O needed."""
    return PILImage.new("RGB", (16, 16), color=(200, 100, 50))


# ── Prompt building ──────────────────────────────────────────────────────────


class TestBuildDetectionPrompt:
    def test_includes_all_amenities(self, detector: AmenityDetector):
        """All amenity names from the schema should appear in the prompt."""
        prompt = detector._build_detection_prompt(["refrigerator", "oven", "bed"])
        assert "refrigerator" in prompt
        assert "oven" in prompt
        assert "bed" in prompt

    def test_prompt_requests_json(self, detector: AmenityDetector):
        """The prompt must instruct the model to respond with JSON."""
        prompt = detector._build_detection_prompt(["refrigerator"])
        assert "JSON" in prompt or "json" in prompt.lower()

    def test_prompt_mentions_confidence(self, detector: AmenityDetector):
        """Phase 4: prompt should ask for confidence scores."""
        prompt = detector._build_detection_prompt(["refrigerator"])
        assert "confidence" in prompt.lower()


class TestBuildDescriptionPrompt:
    def test_includes_present_amenities(self, detector: AmenityDetector):
        """Detected amenity names should appear in the description prompt."""
        prompt = detector._build_description_prompt(
            {"refrigerator": True, "oven": False, "dishwasher": True}
        )
        assert "refrigerator" in prompt
        assert "dishwasher" in prompt
        # Absent amenity should NOT be listed as present
        assert "oven" not in prompt

    def test_empty_amenities_gives_generic_prompt(self, detector: AmenityDetector):
        """No detected amenities should produce a generic fallback prompt."""
        prompt = detector._build_description_prompt({})
        assert len(prompt) > 0
        assert "brief" in prompt.lower() or "description" in prompt.lower()


# ── JSON parsing ─────────────────────────────────────────────────────────────


class TestParseJsonFromText:
    def test_parses_new_structured_format(self, detector: AmenityDetector):
        """New format with present + confidence should be parsed correctly."""
        text = '{"refrigerator": {"present": true, "confidence": 0.95}}'
        result = detector._parse_json_from_text(text)
        assert "refrigerator" in result
        is_present, confidence = result["refrigerator"]
        assert is_present is True
        assert abs(confidence - 0.95) < 0.001

    def test_parses_legacy_boolean_format(self, detector: AmenityDetector):
        """Legacy bool format should produce is_present=True, confidence=1.0."""
        text = '{"refrigerator": true, "oven": false}'
        result = detector._parse_json_from_text(text)
        assert result["refrigerator"] == (True, 1.0)
        assert result["oven"] == (False, 0.0)

    def test_strips_markdown_code_block(self, detector: AmenityDetector):
        """JSON wrapped in ```json ... ``` should still be parsed."""
        text = '```json\n{"bed": {"present": true, "confidence": 0.8}}\n```'
        result = detector._parse_json_from_text(text)
        assert "bed" in result
        assert result["bed"][0] is True

    def test_ignores_surrounding_commentary(self, detector: AmenityDetector):
        """JSON buried in prose should still be extracted."""
        text = 'Sure, here is the JSON: {"refrigerator": false} Done.'
        result = detector._parse_json_from_text(text)
        assert "refrigerator" in result
        assert result["refrigerator"][0] is False

    def test_returns_empty_dict_for_invalid_json(self, detector: AmenityDetector):
        """Totally malformed text should return an empty dict (no exception)."""
        result = detector._parse_json_from_text("I cannot identify any amenities here.")
        assert result == {}

    def test_returns_empty_dict_for_empty_string(self, detector: AmenityDetector):
        """Empty string input should not raise, just return empty."""
        result = detector._parse_json_from_text("")
        assert result == {}

    def test_clamps_confidence_above_1(self, detector: AmenityDetector):
        """Confidence > 1.0 from the model should be clamped to 1.0."""
        text = '{"oven": {"present": true, "confidence": 1.5}}'
        result = detector._parse_json_from_text(text)
        _, confidence = result["oven"]
        assert confidence <= 1.0

    def test_clamps_confidence_below_0(self, detector: AmenityDetector):
        """Negative confidence should be clamped to 0.0."""
        text = '{"oven": {"present": false, "confidence": -0.2}}'
        result = detector._parse_json_from_text(text)
        _, confidence = result["oven"]
        assert confidence >= 0.0

    def test_handles_single_quotes(self, detector: AmenityDetector):
        """Single-quoted JSON (Python dict repr) should be normalised."""
        text = "{'refrigerator': true}"
        result = detector._parse_json_from_text(text)
        assert "refrigerator" in result


# ── detect_from_image ────────────────────────────────────────────────────────


class TestDetectFromImage:
    def test_returns_three_dicts(
        self,
        detector: AmenityDetector,
        fake_vlm: MagicMock,
        small_image: PILImage.Image,
    ):
        """detect_from_image must return a 3-tuple of dicts."""
        fake_vlm.generate.return_value = VLMResponse(
            raw_text='{"refrigerator": {"present": true, "confidence": 0.9}}',
            model_name="fake-model",
        )
        result = detector.detect_from_image(small_image)
        assert len(result) == 3
        amenities_by_room, flat_amenities, flat_confidences = result
        assert isinstance(amenities_by_room, dict)
        assert isinstance(flat_amenities, dict)
        assert isinstance(flat_confidences, dict)

    def test_flat_amenities_match_schema(
        self,
        detector: AmenityDetector,
        fake_vlm: MagicMock,
        small_image: PILImage.Image,
    ):
        """flat_amenities values should all be booleans."""
        fake_vlm.generate.return_value = VLMResponse(
            raw_text='{"refrigerator": {"present": true, "confidence": 0.9}, "bed": {"present": false, "confidence": 0.1}}',
            model_name="fake-model",
        )
        _, flat_amenities, _ = detector.detect_from_image(small_image)
        for key in flat_amenities:
            # Each value in the flat dict must be a bool
            assert isinstance(flat_amenities[key], bool)

    def test_amenities_by_room_uses_schema_keys(
        self,
        detector: AmenityDetector,
        fake_vlm: MagicMock,
        amenity_schema: dict[str, list[str]],
        small_image: PILImage.Image,
    ):
        """amenities_by_room should have the same room-type keys as the schema."""
        fake_vlm.generate.return_value = VLMResponse(
            raw_text='{"refrigerator": true}',
            model_name="fake-model",
        )
        amenities_by_room, _, _ = detector.detect_from_image(small_image)
        assert set(amenities_by_room.keys()) == set(amenity_schema.keys())

    def test_vlm_error_returns_empty_dicts(
        self,
        detector: AmenityDetector,
        fake_vlm: MagicMock,
        small_image: PILImage.Image,
    ):
        """A RuntimeError from the VLM should not propagate — return empty dicts."""
        fake_vlm.generate.side_effect = RuntimeError("Ollama not running")
        amenities_by_room, flat_amenities, flat_confidences = detector.detect_from_image(
            small_image
        )
        assert flat_amenities == {}
        assert flat_confidences == {}
        # amenities_by_room should still have room-type keys (defaulting to False)
        assert "kitchen" in amenities_by_room
        assert all(not v for v in amenities_by_room["kitchen"].values())

    def test_confidence_propagated_correctly(
        self,
        detector: AmenityDetector,
        fake_vlm: MagicMock,
        small_image: PILImage.Image,
    ):
        """Confidence values from the VLM response should appear in flat_confidences."""
        fake_vlm.generate.return_value = VLMResponse(
            raw_text='{"refrigerator": {"present": true, "confidence": 0.92}}',
            model_name="fake-model",
        )
        _, _, flat_confidences = detector.detect_from_image(small_image)
        assert "refrigerator" in flat_confidences
        assert abs(flat_confidences["refrigerator"] - 0.92) < 0.001


# ── generate_description ─────────────────────────────────────────────────────


class TestGenerateDescription:
    def test_returns_string(
        self,
        detector: AmenityDetector,
        fake_vlm: MagicMock,
        small_image: PILImage.Image,
    ):
        """generate_description should always return a non-empty string."""
        fake_vlm.generate.return_value = VLMResponse(
            raw_text="A modern kitchen with stainless steel appliances.",
            model_name="fake-model",
        )
        result = detector.generate_description(small_image, {"refrigerator": True})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_strips_whitespace(
        self,
        detector: AmenityDetector,
        fake_vlm: MagicMock,
        small_image: PILImage.Image,
    ):
        """Leading/trailing whitespace in VLM output should be stripped."""
        fake_vlm.generate.return_value = VLMResponse(
            raw_text="  Description with spaces.  ",
            model_name="fake-model",
        )
        result = detector.generate_description(small_image, {})
        assert result == "Description with spaces."

    def test_vlm_error_returns_fallback_message(
        self,
        detector: AmenityDetector,
        fake_vlm: MagicMock,
        small_image: PILImage.Image,
    ):
        """A VLM RuntimeError should produce a graceful fallback message."""
        fake_vlm.generate.side_effect = RuntimeError("timeout")
        result = detector.generate_description(small_image, {"refrigerator": True})
        assert isinstance(result, str)
        # The fallback message is defined in the implementation
        assert len(result) > 0
