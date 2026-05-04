"""Unit tests for expanded amenity hint payload mapping."""

from ui.hints import HINT_KEYS, hints_payload, legacy_flags


def test_hints_payload_maps_yes_no_and_drops_unspecified() -> None:
    payload = hints_payload(
        {
            "kitchen": "Yes",
            "garage": "No",
            "elevator": "Not specified",
            "balcony": None,
        }
    )

    assert payload == {"kitchen": True, "garage": False}


def test_legacy_flags_derive_from_expanded_payload() -> None:
    flags = legacy_flags({"kitchen": True, "balcony": False, "living_room": True})

    assert flags == {
        "has_kitchen": True,
        "has_balcony": False,
        "has_living_room": True,
    }


def test_hint_key_set_contains_expected_18_items() -> None:
    assert len(HINT_KEYS) == 18
    assert {"kitchen", "bathroom", "garage", "elevator", "furnished"} <= set(HINT_KEYS)
