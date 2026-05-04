"""Amenity hint definitions and payload helpers for the Gradio UI."""

from __future__ import annotations

CHOICES = ("Not specified", "Yes", "No")

AMENITY_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Rooms",
        (
            ("kitchen", "Kitchen"),
            ("living_room", "Living room"),
            ("bedroom", "Bedroom"),
            ("bathroom", "Bathroom"),
            ("dining_room", "Dining room"),
        ),
    ),
    (
        "Outdoor",
        (
            ("balcony", "Balcony"),
            ("terrace", "Terrace"),
            ("garden", "Garden"),
            ("pool", "Pool"),
        ),
    ),
    (
        "Storage & access",
        (
            ("garage", "Garage"),
            ("parking", "Parking"),
            ("elevator", "Elevator"),
            ("storage", "Storage"),
        ),
    ),
    (
        "Features",
        (
            ("fireplace", "Fireplace"),
            ("air_conditioning", "Air conditioning"),
            ("heating", "Heating"),
            ("furnished", "Furnished"),
            ("pet_friendly", "Pet friendly"),
        ),
    ),
)

HINT_KEYS: tuple[str, ...] = tuple(key for _group, items in AMENITY_GROUPS for key, _label in items)
HINT_LABELS: dict[str, str] = {
    key: label for _group, items in AMENITY_GROUPS for key, label in items
}


def hint_to_bool(value: str | None) -> bool | None:
    """Translate UI tri-state strings to API booleans."""
    if value == "Yes":
        return True
    if value == "No":
        return False
    return None


def hints_payload(values: dict[str, str | None]) -> dict[str, bool]:
    """Map tri-state UI values to the sparse API hints dict."""
    payload: dict[str, bool] = {}
    for key, value in values.items():
        parsed = hint_to_bool(value)
        if parsed is not None:
            payload[key] = parsed
    return payload


def legacy_flags(hints: dict[str, bool]) -> dict[str, bool | None]:
    """Derive legacy describe fields from the expanded hint dict."""
    return {
        "has_kitchen": hints.get("kitchen"),
        "has_balcony": hints.get("balcony"),
        "has_living_room": hints.get("living_room"),
    }
