"""
AmenityDetector — orchestrates VLM calls and result parsing.

This module is responsible for two things:
  1. Building the right prompts to ask the VLM about an image.
  2. Parsing the VLM's free-text response back into structured data.

Importantly, it does NOT know which VLM is being used — it just calls
`client.generate(image, prompt)` and gets a VLMResponse back. The VLMClient
implementation (Ollama, Gemini) is injected from outside (dependency injection).

Why prompt engineering matters here:
  We ask the model to respond in JSON so we can parse it reliably. The prompts
  are carefully written to minimise hallucination and get consistent structure.
  If output parsing fails, we fall back to an empty detection result rather than
  crashing — the caller sees a graceful degradation.
"""

import json
import logging
import re

from PIL.Image import Image

from core.preprocessing import preprocess_image
from models.base import VLMClient


class AmenityDetector:
    """
    Detects amenities in property images and generates descriptions.

    Injected with a VLMClient so it's model-agnostic. The same code works
    whether you're running Qwen2.5-VL-7B locally via Ollama or Gemini Flash
    via the cloud API.
    """

    def __init__(
        self,
        vlm_client: VLMClient,
        amenity_schema: dict[str, list[str]],
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise the detector.

        Args:
            vlm_client:     Any VLMClient implementation (Ollama, Gemini, etc.)
            amenity_schema: Dict mapping room type → list of amenity names.
                            e.g. {"kitchen": ["refrigerator", "oven"], "bedroom": [...]}
            logger:         Optional logger. Defaults to module logger.
        """
        self.client = vlm_client
        self.amenity_schema = amenity_schema
        self.logger = logger or logging.getLogger(__name__)

    def _build_detection_prompt(self, all_amenities: list[str]) -> str:
        """
        Build the prompt used to ask the VLM which amenities are visible.

        The prompt asks for a strict JSON response to make parsing deterministic.
        Each amenity entry includes both a boolean ``present`` flag and a ``confidence``
        score (0.0–1.0) so the UI and search features can surface high-confidence results
        first. We include a format example so even smaller models understand the structure.

        Args:
            all_amenities: Flattened, deduplicated list of amenity names to check.

        Returns:
            A prompt string ready to send to the VLM.
        """
        amenity_list = ", ".join(all_amenities)
        return (
            "You are analysing one property photo. Identify the most likely room type "
            "(for example kitchen, bedroom, bathroom, living_room, balcony, dining_room, "
            "or unknown), then check which listed amenities are visible.\n\n"
            f"Amenities to check: {amenity_list}\n\n"
            "Respond ONLY with one JSON object using this exact shape:\n"
            "{\n"
            '  "room_type": "kitchen",\n'
            '  "amenities": {\n'
            '    "refrigerator": {"present": true, "confidence": 0.95},\n'
            '    "oven": {"present": false, "confidence": 0.10}\n'
            "  }\n"
            "}\n\n"
            "Rules:\n"
            "- Use amenity names exactly as listed above.\n"
            "- Set present=true only when the amenity is clearly visible.\n"
            "- Confidence must be a number from 0.0 to 1.0.\n"
            "- Do not add commentary, markdown, or keys outside room_type and amenities.\n\n"
            "JSON response:"
        )

    def _build_description_prompt(self, detected_amenities: dict[str, bool]) -> str:
        """
        Build the prompt to ask the VLM for a natural-language property description.

        Args:
            detected_amenities: Flat dict of amenity_name → bool (True = present).

        Returns:
            A prompt string for description generation.
        """
        present = [name for name, present in detected_amenities.items() if present]

        if not present:
            return (
                "You are describing a property image. "
                "Based on what you can see, write a brief 2-3 sentence description "
                "of the room, even if no specific amenities are clearly visible."
            )

        amenity_list = ", ".join(present)
        return (
            f"You are describing a property listing. "
            f"The following amenities are visible in the image: {amenity_list}.\n\n"
            f"Write a natural, appealing 2-3 sentence description of this room "
            f"that highlights these amenities. Also identify the room type "
            f"(kitchen, bedroom, bathroom, living room, etc.) and include it naturally "
            f"in the description. Do NOT use the word 'amenity' — write like a real listing."
        )

    def _extract_json_string(self, text: str) -> str | None:
        """
        Find and return the raw JSON substring from VLM free-text output.

        VLMs often wrap JSON in markdown code blocks or add surrounding commentary.
        Two extraction strategies are tried in order:
          1. A fenced code block  (```json ... ``` or ``` ... ```)
          2. The first brace-delimited span in the text  ({ ... })

        Args:
            text: Raw text from the VLM response.

        Returns:
            The extracted JSON string, or None if nothing brace-delimited was found.
        """
        # Strategy 1: markdown fenced code block
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block:
            return code_block.group(1)

        # Strategy 2: first brace-delimited object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        return text[start:end]

    def _normalise_json_string(self, json_str: str) -> str:
        """
        Fix common formatting quirks produced by VLMs before passing to json.loads.

        Normalises:
          - Single quotes → double quotes  (Python-style dicts)
          - Python True/False → JSON true/false
          - Escaped underscores  (\\_  →  _)

        Args:
            json_str: A raw string that looks like JSON but may have quirks.

        Returns:
            The normalised string.
        """
        json_str = json_str.replace("'", '"')
        json_str = json_str.replace("True", "true").replace("False", "false")
        json_str = json_str.replace("\\_", "_")
        return json_str

    def _parse_amenity_entry(self, key: str, value: object) -> tuple[bool, float]:
        """
        Convert a single value from the parsed JSON into (is_present, confidence).

        The prompt asks for the new structured format:
            {"present": true, "confidence": 0.9}
        But older/smaller models may still return a plain boolean.
        Both formats are handled here so the parser is robust.

        Args:
            key:   Amenity name (used only for logging on unexpected values).
            value: The JSON value — either a bool or a dict with present/confidence.

        Returns:
            Tuple of (is_present: bool, confidence: float in 0.0–1.0).
        """
        if isinstance(value, bool):
            # Legacy boolean format — default confidence to 1.0 / 0.0
            return value, (1.0 if value else 0.0)

        if isinstance(value, dict):
            present = bool(value.get("present", False))
            # Clamp confidence to [0.0, 1.0] in case the model overflows
            raw_conf = value.get("confidence", 1.0 if present else 0.0)
            confidence = max(0.0, min(1.0, float(raw_conf)))
            return present, confidence

        # Unexpected value type — log and fall back to absent/zero-confidence
        self.logger.warning(
            "Unexpected JSON value for amenity '%s': %r — treating as not present.", key, value
        )
        return False, 0.0

    def _parse_raw_json_object(self, text: str) -> dict[str, object] | None:
        """
        Extract and decode a JSON object from VLM free text.

        Returns:
            The decoded object, or None when extraction/decoding fails.
        """
        json_str = self._extract_json_string(text)
        if json_str is None:
            self.logger.warning("No JSON object found in VLM response. Raw: %s", text[:300])
            return None

        json_str = self._normalise_json_string(json_str)

        try:
            raw = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.warning("JSON parse failed (%s). Raw text: %s", e, text[:300])
            return None

        if not isinstance(raw, dict):
            self.logger.warning("VLM JSON response was not an object. Raw: %s", text[:300])
            return None
        return raw

    def _parse_json_from_text(self, text: str) -> dict[str, tuple[bool, float]]:
        """
        Extract amenity detection results from VLM free-text output.

        Supports two JSON formats produced by the model:

        New format (preferred — requested by the detection prompt):
            {"refrigerator": {"present": true, "confidence": 0.95}, ...}

        Legacy boolean format (produced by older/smaller models):
            {"refrigerator": true, "oven": false}

        Both formats are normalised into a dict of amenity_name → (is_present, confidence).

        Args:
            text: Raw text from the VLM response.

        Returns:
            Dict mapping amenity name → (is_present, confidence).
            Returns an empty dict if no valid JSON is found.
        """
        raw = self._parse_raw_json_object(text)
        if raw is None:
            return {}

        # Phase 5 response shape nests amenity entries under "amenities".
        if isinstance(raw.get("amenities"), dict):
            raw = raw["amenities"]  # type: ignore[assignment]

        return {key: self._parse_amenity_entry(key, val) for key, val in raw.items()}

    def _parse_detection_response(
        self, text: str
    ) -> tuple[str | None, dict[str, tuple[bool, float]]]:
        """
        Parse the preferred Phase 5 detection response.

        The preferred shape is:
            {"room_type": "kitchen", "amenities": {"oven": {"present": true, ...}}}

        For compatibility with existing tests and less obedient models, the older
        flat amenity object is still accepted and returns room_type=None.
        """
        raw = self._parse_raw_json_object(text)
        if raw is None:
            return None, {}

        room_type = raw.get("room_type")
        parsed_room = (
            str(room_type).strip() if isinstance(room_type, str) and room_type.strip() else None
        )

        amenities_obj = raw.get("amenities")
        if isinstance(amenities_obj, dict):
            parsed = {
                key: self._parse_amenity_entry(key, val) for key, val in amenities_obj.items()
            }
            return parsed_room, parsed

        parsed = {key: self._parse_amenity_entry(key, val) for key, val in raw.items()}
        return None, parsed

    def detect_from_image(
        self, image: Image
    ) -> tuple[dict[str, dict[str, bool]], dict[str, bool], dict[str, float]]:
        """
        Run amenity detection on a PIL Image.

        This is the primary detection method used by the API. It:
          1. Builds a prompt listing all amenities from the schema
          2. Calls the VLM and handles timeout / connection errors gracefully
          3. Parses the JSON response (supports both new structured and legacy boolean formats)
          4. Organises results by room type (for the DB) and as flat dicts

        Args:
            image: A PIL Image of the room to analyse.

        Returns:
            Tuple of three dicts:

            amenities_by_room: {room_type: {amenity_name: bool}}
                Used to infer the room type and structure DB records.
                e.g. {"kitchen": {"refrigerator": True, "oven": False}, ...}

            flat_amenities: {amenity_name: bool}
                Convenience dict of all detected amenities across all rooms.
                e.g. {"refrigerator": True, "oven": False, "bed": False, ...}

            flat_confidences: {amenity_name: float}
                Model confidence per amenity (0.0 = not sure, 1.0 = certain).
                e.g. {"refrigerator": 0.95, "oven": 0.1, ...}

            On VLM error, all three dicts are empty rather than raising an exception,
            so a single bad image does not abort the whole upload pipeline.
        """
        # Flatten and deduplicate all amenity names across all room types
        all_amenities = sorted(
            {amenity for amenities in self.amenity_schema.values() for amenity in amenities}
        )

        prompt = self._build_detection_prompt(all_amenities)

        # Call the VLM — catch RuntimeError (connection/timeout/HTTP errors from clients)
        processed = preprocess_image(image)
        try:
            response = self.client.generate(processed, prompt)
            detected_room, parsed = self._parse_detection_response(response.raw_text)
        except RuntimeError as e:
            self.logger.error("VLM detection failed: %s", e)
            detected_room = None
            parsed = {}

        # Split the parsed dict into separate bool and float dicts for clarity
        flat_amenities: dict[str, bool] = {k: v[0] for k, v in parsed.items()}
        flat_confidences: dict[str, float] = {k: v[1] for k, v in parsed.items()}

        # Prefer the room label supplied by the Phase 5 structured response. For
        # legacy flat responses, keep the older schema-wide structure so callers
        # can still infer a room by counting present amenities.
        amenities_by_room: dict[str, dict[str, bool]] = {}
        if detected_room:
            amenities_by_room[detected_room] = dict(flat_amenities)
        else:
            for room_type, amenity_list in self.amenity_schema.items():
                amenities_by_room[room_type] = {
                    amenity: flat_amenities.get(amenity, False) for amenity in amenity_list
                }

        return amenities_by_room, flat_amenities, flat_confidences

    def generate_description(self, image: Image, detected_amenities: dict[str, bool]) -> str:
        """
        Generate a natural-language description for the property image.

        Args:
            image:               PIL Image of the room.
            detected_amenities:  Flat dict of amenity_name → bool from detect_from_image().

        Returns:
            A 2-3 sentence natural language description string.
            Falls back to a generic message if the VLM call fails.
        """
        prompt = self._build_description_prompt(detected_amenities)

        try:
            response = self.client.generate(image, prompt)
            return response.raw_text.strip()
        except RuntimeError as e:
            self.logger.error("VLM description generation failed: %s", e)
            return "Could not generate a description for this image."
