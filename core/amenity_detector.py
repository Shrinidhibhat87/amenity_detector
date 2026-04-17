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
        We include a format example so even smaller models understand the structure.

        Args:
            all_amenities: Flattened, deduplicated list of amenity names to check.

        Returns:
            A prompt string ready to send to the VLM.
        """
        amenity_list = ", ".join(all_amenities)
        return (
            f"You are analysing a property image. "
            f"For each of the following amenities, determine whether it is VISIBLE in the image.\n\n"
            f"Amenities to check: {amenity_list}\n\n"
            f"Respond ONLY with a JSON object. "
            f"Keys are amenity names (exactly as listed above). "
            f"Values are true (present and visible) or false (not present or not visible).\n"
            f'Example format: {{"refrigerator": true, "oven": false, "dishwasher": true}}\n\n'
            f"JSON response:"
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

    def _parse_json_from_text(self, text: str) -> dict[str, bool]:
        """
        Extract a JSON object from VLM free-text output.

        VLMs often wrap JSON in markdown code blocks or add surrounding commentary.
        This method tries progressively looser strategies to find valid JSON.

        Strategy:
          1. Look for a JSON code block (```json ... ```)
          2. Look for the first { ... } span in the text
          3. Fall back to empty dict with a warning logged

        Args:
            text: Raw text from the VLM response.

        Returns:
            A dict of amenity_name → bool. Empty dict on parse failure.
        """
        # Strategy 1: markdown code block
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block:
            json_str = code_block.group(1)
        else:
            # Strategy 2: first brace-delimited object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                self.logger.warning("No JSON object found in VLM response. Raw: %s", text[:300])
                return {}
            json_str = text[start:end]

        # Normalise common VLM quirks before parsing:
        #   - Single quotes → double quotes
        #   - Python True/False → JSON true/false
        #   - Escaped underscores (some models add \_)
        json_str = json_str.replace("'", '"')
        json_str = json_str.replace("True", "true").replace("False", "false")
        json_str = json_str.replace("\\_", "_")

        try:
            result: dict[str, bool] = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            self.logger.warning(
                "JSON parse failed (%s). Raw text: %s", e, text[:300]
            )
            return {}

    def detect_from_image(
        self, image: Image
    ) -> tuple[dict[str, dict[str, bool]], dict[str, bool]]:
        """
        Run amenity detection on a PIL Image.

        This is the primary detection method used by the API. It:
          1. Builds a prompt listing all amenities from the schema
          2. Calls the VLM
          3. Parses the JSON response
          4. Organises results by room type (for the DB) and as a flat dict

        Args:
            image: A PIL Image of the room to analyse.

        Returns:
            Tuple of:
              amenities_by_room: {room_type: {amenity_name: bool}}
                  e.g. {"kitchen": {"refrigerator": True, "oven": False}, ...}
              flat_amenities: {amenity_name: bool}
                  e.g. {"refrigerator": True, "oven": False, "bed": False, ...}

            On VLM error, returns empty dicts rather than raising.
        """
        # Flatten and deduplicate all amenity names across all room types
        all_amenities = sorted(
            {amenity for amenities in self.amenity_schema.values() for amenity in amenities}
        )

        prompt = self._build_detection_prompt(all_amenities)

        try:
            response = self.client.generate(image, prompt)
            flat_amenities = self._parse_json_from_text(response.raw_text)
        except RuntimeError as e:
            self.logger.error("VLM detection failed: %s", e)
            flat_amenities = {}

        # Organise flat results back into the room-type structure
        amenities_by_room: dict[str, dict[str, bool]] = {}
        for room_type, amenity_list in self.amenity_schema.items():
            amenities_by_room[room_type] = {
                amenity: flat_amenities.get(amenity, False)
                for amenity in amenity_list
            }

        return amenities_by_room, flat_amenities

    def generate_description(
        self, image: Image, detected_amenities: dict[str, bool]
    ) -> str:
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