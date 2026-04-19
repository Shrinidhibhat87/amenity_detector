"""
Tests for ui/app.py helper functions.

These are pure-Python functions that transform API response data into
Gradio display formats. No Gradio runtime needed — just plain unit tests.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests


def _make_image(filename: str, room: str, amenities: list[dict]) -> dict:
    """Helper to build a mock image dict matching the API response shape."""
    return {
        "file_path": f"/storage/{filename}",
        "room_type": room,
        "amenities": amenities,
    }


def test_format_amenities_table_basic() -> None:
    """Should return one row per amenity across all images."""
    from ui.app import _format_amenities_table

    images = [
        _make_image(
            "kitchen.jpg",
            "kitchen",
            [
                {"amenity_name": "refrigerator", "is_present": True, "confidence": 0.95},
                {"amenity_name": "dishwasher", "is_present": False, "confidence": 0.30},
            ],
        )
    ]

    rows = _format_amenities_table(images)

    assert len(rows) == 2
    assert rows[0] == ["kitchen", "refrigerator", True, "0.95"]
    assert rows[1] == ["kitchen", "dishwasher", False, "0.30"]


def test_format_amenities_table_multiple_images() -> None:
    """Should flatten amenities from all images into a single list."""
    from ui.app import _format_amenities_table

    images = [
        _make_image(
            "lr.jpg",
            "living_room",
            [{"amenity_name": "sofa", "is_present": True, "confidence": 0.9}],
        ),
        _make_image(
            "bed.jpg", "bedroom", [{"amenity_name": "bed", "is_present": True, "confidence": 0.88}]
        ),
    ]

    rows = _format_amenities_table(images)
    assert len(rows) == 2
    rooms = [r[0] for r in rows]
    assert "living_room" in rooms
    assert "bedroom" in rooms


def test_format_amenities_table_missing_confidence() -> None:
    """If confidence is None, the cell should show '—' not crash."""
    from ui.app import _format_amenities_table

    images = [
        _make_image(
            "x.jpg",
            "kitchen",
            [{"amenity_name": "oven", "is_present": True, "confidence": None}],
        )
    ]

    rows = _format_amenities_table(images)
    assert rows[0][3] == "—"


def test_get_available_models_fallback() -> None:
    """If the backend is unreachable, returns the hardcoded fallback list."""
    with patch("ui.app.requests.get", side_effect=requests.exceptions.ConnectionError):
        from ui.app import _get_available_models

        models = _get_available_models()

    assert isinstance(models, list)
    assert len(models) > 0
    assert "gemini-2.0-flash" in models


def test_get_available_models_filters_unavailable() -> None:
    """Only models with available=True are returned."""
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"name": "gemini-2.0-flash", "available": True},
        {"name": "qwen2.5vl:7b", "available": False},
    ]

    with patch("ui.app.requests.get", return_value=mock_response):
        from ui.app import _get_available_models

        models = _get_available_models()

    assert models == ["gemini-2.0-flash"]
    assert "qwen2.5vl:7b" not in models


def test_reconcile_flags_kitchen_when_user_says_no_kitchen() -> None:
    """Kitchen detections contradicting the sidebar should be marked."""
    from ui.helpers import WARNING_PREFIX, reconcile_amenities

    rows = [["kitchen", "refrigerator", True, "0.95"]]
    out = reconcile_amenities({"has_kitchen": False}, rows)

    assert out[0][1].startswith(WARNING_PREFIX)
    assert rows[0][1] == "refrigerator"


def test_reconcile_flags_balcony_when_user_says_no_balcony() -> None:
    """Balcony-like rooms or amenities should be marked when balcony is false."""
    from ui.helpers import WARNING_PREFIX, reconcile_amenities

    rows = [["balcony", "outdoor seating", True, "0.80"]]
    out = reconcile_amenities({"has_balcony": False}, rows)

    assert out[0][1].startswith(WARNING_PREFIX)


def test_reconcile_flags_living_room_when_user_says_no_living_room() -> None:
    """Living-room signals should be marked when the user says there is none."""
    from ui.helpers import WARNING_PREFIX, reconcile_amenities

    rows = [["living_room", "sofa", True, "0.90"]]
    out = reconcile_amenities({"has_living_room": False}, rows)

    assert out[0][1].startswith(WARNING_PREFIX)


def test_reconcile_flags_rooms_over_user_room_count() -> None:
    """Rows outside the user-provided room count should be marked."""
    from ui.helpers import WARNING_PREFIX, reconcile_amenities

    rows = [
        ["kitchen", "refrigerator", True, "0.95"],
        ["bedroom", "bed", True, "0.90"],
    ]
    out = reconcile_amenities({"num_rooms": 1}, rows)

    assert not out[0][1].startswith(WARNING_PREFIX)
    assert out[1][1].startswith(WARNING_PREFIX)


def test_reconcile_does_not_flag_absent_amenities() -> None:
    """Only present detections can contradict property-level hints."""
    from ui.helpers import WARNING_PREFIX, reconcile_amenities

    rows = [["kitchen", "refrigerator", False, "0.10"]]
    out = reconcile_amenities({"has_kitchen": False}, rows)

    assert not out[0][1].startswith(WARNING_PREFIX)


def test_reconcile_removes_old_warning_when_hint_changes() -> None:
    """Reconciliation is repeatable and removes stale warning prefixes."""
    from ui.helpers import WARNING_PREFIX, reconcile_amenities

    rows = [["kitchen", f"{WARNING_PREFIX}refrigerator", True, "0.95"]]
    out = reconcile_amenities({"has_kitchen": True}, rows)

    assert out[0][1] == "refrigerator"


def test_reconcile_from_hints_accepts_dataframe() -> None:
    """Regression: Gradio can pass a pandas DataFrame, and `df or []` raises
    ``ValueError: The truth value of a DataFrame is ambiguous``. The handler
    must normalise to a list of rows before reconciling."""
    pd = pytest.importorskip("pandas")
    from ui.app import reconcile_table_from_hints

    df = pd.DataFrame(
        [["kitchen", "refrigerator", True, "0.95"]],
        columns=["Room", "Amenity", "Present", "Confidence"],
    )

    # Should not raise — previously threw ValueError on `df or []`.
    out = reconcile_table_from_hints(0, "Not specified", "Not specified", "Not specified", df)

    assert out == [["kitchen", "refrigerator", True, "0.95"]]


def test_reconcile_from_hints_handles_none() -> None:
    """Empty table / first render passes None — must not raise."""
    from ui.app import reconcile_table_from_hints

    assert (
        reconcile_table_from_hints(0, "Not specified", "Not specified", "Not specified", None) == []
    )


def test_build_app_smoke_returns_blocks() -> None:
    """Smoke test for Phase 6 @gr.render wiring."""
    import gradio as gr

    from ui.app import build_app

    app = build_app()

    assert isinstance(app, gr.Blocks)
