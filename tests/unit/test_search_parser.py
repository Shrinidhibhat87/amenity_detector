"""Unit tests for the natural-language query parser.

The parser turns free text into a structured ``SearchFilter`` so the search
pipeline can build a SQL ``WHERE``. We test both the regex fallback (which
runs offline) and the LLM-backed path (with the OpenRouter client mocked).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from core.search import RoomAmenity, SearchFilter
from core.search.parser import QueryParser, fallback_regex_parse


class TestRegexFallback:
    """The fallback parser is what happens when the LLM is unavailable.

    It is intentionally narrow — extracts the same handful of fields the
    frontend mock parser does so the search UI keeps working in dev without
    an OpenRouter key.
    """

    def test_extracts_bedrooms_and_listing_type(self) -> None:
        f = fallback_regex_parse("3 bhk apartment to rent")
        assert f.min_bedrooms == 3
        assert f.max_bedrooms == 3
        assert f.listing_type == "rent"

    def test_extracts_price_ceiling_with_currency(self) -> None:
        f = fallback_regex_parse("flat under 1500 EUR")
        assert f.max_price == 1500.0
        assert f.currency == "EUR"

    def test_extracts_room_amenity_tuples(self) -> None:
        f = fallback_regex_parse("apartment with fireplace in living room")
        assert RoomAmenity(room_type="living_room", amenity_name="fireplace") in f.required_amenities

    def test_extracts_bare_amenities_without_room(self) -> None:
        f = fallback_regex_parse("flat with wifi and parking")
        names = {ra.amenity_name for ra in f.required_amenities}
        assert "wifi" in names
        assert "parking" in names
        assert all(ra.room_type is None for ra in f.required_amenities)

    def test_buy_maps_to_sale(self) -> None:
        f = fallback_regex_parse("villa for sale")
        assert f.listing_type == "sale"
        f2 = fallback_regex_parse("looking to buy a house")
        assert f2.listing_type == "sale"

    def test_empty_input_returns_empty_filter(self) -> None:
        f = fallback_regex_parse("")
        assert f == SearchFilter(free_text="")


def _mock_chat_completion(payload: dict) -> MagicMock:
    """Build a fake OpenAI chat completion result with a JSON message."""
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = json.dumps(payload)
    return completion


class TestQueryParserLLM:
    def test_parse_uses_llm_response_when_valid(self) -> None:
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _mock_chat_completion(
            {
                "listing_type": "rent",
                "min_bedrooms": 3,
                "max_bedrooms": 3,
                "max_price": 1500.0,
                "currency": "EUR",
                "required_amenities": [
                    {"room_type": "living_room", "amenity_name": "fireplace"}
                ],
                "near": ["park"],
                "free_text": "near a park",
            }
        )

        parser = QueryParser(openai_client=fake_client, model="openai/gpt-4o-mini")
        result = parser.parse("3BHK rent under 1500 EUR with fireplace in living room near a park")

        assert result.listing_type == "rent"
        assert result.min_bedrooms == 3
        assert result.max_price == 1500.0
        assert result.currency == "EUR"
        assert RoomAmenity(room_type="living_room", amenity_name="fireplace") in (
            result.required_amenities
        )
        assert "park" in result.near

    def test_parse_falls_back_to_regex_when_llm_errors(self) -> None:
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("rate limited")

        parser = QueryParser(openai_client=fake_client, model="openai/gpt-4o-mini")
        result = parser.parse("3 bhk rent under 1500 EUR")

        # The regex fallback should still extract the basics.
        assert result.min_bedrooms == 3
        assert result.listing_type == "rent"
        assert result.max_price == 1500.0
        assert result.currency == "EUR"

    def test_parse_falls_back_when_llm_returns_invalid_json(self) -> None:
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="not json at all"))]
        )

        parser = QueryParser(openai_client=fake_client, model="openai/gpt-4o-mini")
        result = parser.parse("2 bedroom apartment to rent")

        assert result.min_bedrooms == 2
        assert result.listing_type == "rent"

    def test_parse_strips_unknown_fields(self) -> None:
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _mock_chat_completion(
            {
                "listing_type": "rent",
                "min_bedrooms": 2,
                # The LLM hallucinated an extra field — Pydantic must ignore it.
                "hallucinated_field": "garbage",
            }
        )

        parser = QueryParser(openai_client=fake_client, model="openai/gpt-4o-mini")
        result = parser.parse("2 bed rent")
        assert result.listing_type == "rent"
        assert result.min_bedrooms == 2

    def test_empty_query_returns_empty_filter_without_llm_call(self) -> None:
        fake_client = MagicMock()
        parser = QueryParser(openai_client=fake_client, model="openai/gpt-4o-mini")
        result = parser.parse("   ")

        assert result == SearchFilter(free_text="")
        fake_client.chat.completions.create.assert_not_called()
