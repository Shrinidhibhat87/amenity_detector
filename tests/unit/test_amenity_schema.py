"""
Unit tests for core/amenity_schema.py

These tests verify that the amenity schema is structured correctly and that
the helper functions return the expected output.

Run with:
    uv run pytest tests/unit/test_amenity_schema.py -v
"""

import json
from pathlib import Path

from core.amenity_schema import AMENITY_SCHEMA, get_all_amenities, load_amenity_schema


class TestAmenitySchema:
    """Tests for the default AMENITY_SCHEMA constant."""

    def test_schema_has_expected_room_types(self) -> None:
        """All six expected room categories should exist as keys."""
        expected_rooms = {"kitchen", "living_room", "bedroom", "bathroom", "outdoor", "common"}
        assert set(AMENITY_SCHEMA.keys()) == expected_rooms

    def test_each_room_has_at_least_one_amenity(self) -> None:
        """Every room type must have a non-empty list of amenities."""
        for room, amenities in AMENITY_SCHEMA.items():
            assert len(amenities) > 0, f"Room '{room}' has no amenities defined"

    def test_all_amenities_are_strings(self) -> None:
        """Amenity names must be plain strings (not None, int, etc.)."""
        for room, amenities in AMENITY_SCHEMA.items():
            for amenity in amenities:
                assert isinstance(amenity, str), (
                    f"Amenity '{amenity}' in room '{room}' is not a string"
                )

    def test_no_empty_amenity_names(self) -> None:
        """Amenity names must not be empty strings."""
        for room, amenities in AMENITY_SCHEMA.items():
            for amenity in amenities:
                assert amenity.strip() != "", f"Empty amenity name found in room '{room}'"


class TestGetAllAmenities:
    """Tests for the get_all_amenities() helper function."""

    def test_returns_sorted_list(self) -> None:
        """Output should be alphabetically sorted for consistent behaviour."""
        result = get_all_amenities(AMENITY_SCHEMA)
        assert result == sorted(result)

    def test_no_duplicates(self) -> None:
        """Each amenity should appear exactly once in the flat list."""
        result = get_all_amenities(AMENITY_SCHEMA)
        assert len(result) == len(set(result))

    def test_contains_known_amenities(self) -> None:
        """Spot-check that well-known amenities are present in the output."""
        result = get_all_amenities(AMENITY_SCHEMA)
        for expected in ["bed", "toilet", "wifi", "pool", "refrigerator"]:
            assert expected in result, f"Expected amenity '{expected}' not found"

    def test_empty_schema_returns_empty_list(self) -> None:
        """Passing an empty schema should produce an empty list, not an error."""
        result = get_all_amenities({})
        assert result == []


class TestLoadAmenitySchema:
    """Tests for the load_amenity_schema() function."""

    def test_returns_default_schema_when_no_path_given(self) -> None:
        """Calling with no argument should return the built-in AMENITY_SCHEMA."""
        result = load_amenity_schema()
        assert result == AMENITY_SCHEMA

    def test_returns_default_schema_on_missing_file(self) -> None:
        """If the given file path does not exist, fall back to the default schema."""
        result = load_amenity_schema("/nonexistent/path/schema.json")
        assert result == AMENITY_SCHEMA

    def test_loads_custom_schema_from_file(self, tmp_path: Path) -> None:
        """A valid JSON file should be loaded and returned as a dict."""
        # Create a minimal custom schema in a temporary directory
        custom_schema = {"test_room": ["item_a", "item_b"]}
        schema_file = tmp_path / "custom_schema.json"
        schema_file.write_text(json.dumps(custom_schema))

        result = load_amenity_schema(str(schema_file))
        assert result == custom_schema
