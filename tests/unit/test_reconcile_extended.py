"""Unit tests for expanded amenity hint reconciliation."""

from ui.helpers import WARNING_PREFIX, reconcile_amenities


def test_reconcile_flags_garage_contradiction_from_expanded_hints() -> None:
    rows = [["garage", "garage door", True, "0.90"]]

    out = reconcile_amenities({"hints": {"garage": False}}, rows)

    assert out[0][1].startswith(WARNING_PREFIX)


def test_reconcile_flags_elevator_contradiction_from_expanded_hints() -> None:
    rows = [["hallway", "elevator", True, "0.85"]]

    out = reconcile_amenities({"hints": {"elevator": False}}, rows)

    assert out[0][1].startswith(WARNING_PREFIX)


def test_reconcile_does_not_flag_positive_expanded_hint() -> None:
    rows = [["bedroom", "bed", True, "0.95"]]

    out = reconcile_amenities({"hints": {"bedroom": True}}, rows)

    assert not out[0][1].startswith(WARNING_PREFIX)
