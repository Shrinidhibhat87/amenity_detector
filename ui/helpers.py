"""
Pure UI helper functions for the Gradio frontend.

These functions do not import Gradio, which keeps them cheap to unit test.
"""

from typing import Any

WARNING_PREFIX = "[warning] "

_KITCHEN_TERMS = {
    "kitchen",
    "refrigerator",
    "fridge",
    "oven",
    "stove",
    "cooktop",
    "microwave",
    "dishwasher",
    "sink",
}
_BALCONY_TERMS = {"balcony", "terrace", "patio", "outdoor seating"}
_LIVING_ROOM_TERMS = {"living_room", "living room", "sofa", "couch", "tv", "television"}


def reconcile_amenities(sidebar: dict[str, Any], rows: list[list[Any]]) -> list[list[Any]]:
    """
    Mark amenity rows that contradict user-provided sidebar hints.

    Args:
        sidebar: Dict with optional keys num_rooms, has_kitchen, has_balcony,
                 and has_living_room.
        rows:    Gradio Dataframe rows in the shape
                 [room_type, amenity_name, is_present, confidence].

    Returns:
        A copied row list where mismatched amenity names are prefixed with
        WARNING_PREFIX. Input rows are not mutated.
    """
    num_rooms = _positive_int_or_none(sidebar.get("num_rooms"))
    has_kitchen = sidebar.get("has_kitchen")
    has_balcony = sidebar.get("has_balcony")
    has_living_room = sidebar.get("has_living_room")

    room_order = _distinct_present_rooms(rows)
    allowed_rooms = set(room_order[:num_rooms]) if num_rooms is not None else set(room_order)

    reconciled: list[list[Any]] = []
    for row in rows:
        copied = list(row)
        room = _normalise_text(copied[0] if len(copied) > 0 else "")
        amenity = _normalise_text(copied[1] if len(copied) > 1 else "")
        present = bool(copied[2]) if len(copied) > 2 else False

        mismatch = False
        if present:
            if has_kitchen is False and _matches_any(room, amenity, _KITCHEN_TERMS):
                mismatch = True
            if has_balcony is False and _matches_any(room, amenity, _BALCONY_TERMS):
                mismatch = True
            if has_living_room is False and _matches_any(room, amenity, _LIVING_ROOM_TERMS):
                mismatch = True
            if num_rooms is not None and room and room not in allowed_rooms:
                mismatch = True

        if mismatch and len(copied) > 1:
            copied[1] = _with_warning(str(copied[1]))
        elif len(copied) > 1:
            copied[1] = _without_warning(str(copied[1]))

        reconciled.append(copied)

    return reconciled


def _distinct_present_rooms(rows: list[list[Any]]) -> list[str]:
    """Return distinct room labels with at least one present amenity, preserving order."""
    rooms: list[str] = []
    for row in rows:
        if len(row) < 3 or not bool(row[2]):
            continue
        room = _normalise_text(row[0])
        if room and room not in rooms:
            rooms.append(room)
    return rooms


def _matches_any(room: str, amenity: str, terms: set[str]) -> bool:
    """Return True when a room or amenity label contains any domain term."""
    haystack = f"{room} {amenity}"
    return any(term in haystack for term in terms)


def _normalise_text(value: Any) -> str:
    """Normalise row values for simple case-insensitive comparisons."""
    return str(value or "").strip().lower().replace("-", "_")


def _positive_int_or_none(value: Any) -> int | None:
    """Convert Gradio slider values into an optional positive integer."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _with_warning(value: str) -> str:
    """Prefix a value once."""
    clean = _without_warning(value)
    return f"{WARNING_PREFIX}{clean}"


def _without_warning(value: str) -> str:
    """Remove the warning prefix if a previous reconciliation added it."""
    return value.removeprefix(WARNING_PREFIX)
