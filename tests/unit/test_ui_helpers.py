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

    assert models == [
        "openai/gpt-4o-mini",
        "google/gemini-pro-1.5",
        "meta-llama/llama-3.2-11b-vision-instruct",
        "qwen/qwen2-vl-72b-instruct",
    ]


def test_get_available_models_filters_unavailable() -> None:
    """Only models with available=True are returned."""
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"name": "openai/gpt-4o-mini", "available": True},
        {"name": "qwen/qwen2-vl-72b-instruct", "available": False},
    ]

    with patch("ui.app.requests.get", return_value=mock_response):
        from ui.app import _get_available_models

        models = _get_available_models()

    assert models == ["openai/gpt-4o-mini"]
    assert "qwen/qwen2-vl-72b-instruct" not in models


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


# ── Per-image timeout / skip-on-failure status helpers ──────────────────────


def test_build_progress_status_no_skipped() -> None:
    """Without skipped images, the status reads cleanly."""
    from ui.app import _build_progress_status

    text = _build_progress_status(2, 5, [])

    assert "2/5 images processed" in text
    assert "Skipped" not in text


def test_build_progress_status_includes_skipped_filenames() -> None:
    """Skipped images are listed by filename so the user can tell what failed."""
    from ui.app import _build_progress_status

    skipped = [
        {"filename": "blurry.jpg", "reason": "timed out after 5s"},
        {"filename": "huge.png", "reason": "HTTP 502"},
    ]

    text = _build_progress_status(3, 5, skipped)

    assert "blurry.jpg" in text
    assert "huge.png" in text
    assert "Skipped 2 image" in text


def test_build_final_status_no_skipped() -> None:
    """Final banner just summarises detected counts when nothing was skipped."""
    from ui.app import _build_final_status

    text = _build_final_status("Sunny Flat", 7, [])

    assert "Detected 7 amenities for 'Sunny Flat'" in text
    assert "not processed" not in text


def test_build_final_status_calls_out_skipped_with_reason() -> None:
    """When images time out the final banner spells out which file and why."""
    from ui.app import _build_final_status

    skipped = [{"filename": "balcony.jpg", "reason": "timed out after 5s"}]

    text = _build_final_status("City Loft", 4, skipped)

    assert "1 image(s) were not processed" in text
    assert "balcony.jpg" in text
    assert "timed out after 5s" in text


def test_per_image_timeout_constant_is_five_seconds() -> None:
    """Spec lock-in: per-image HTTP timeout must stay at 5 seconds.

    If you bump it, update SPEC and the user-facing messaging at the same
    time. The flow and tests both rely on this number being in the status
    text, so a silent change would mislead users.
    """
    from ui.app import _PER_IMAGE_TIMEOUT_SECONDS

    assert _PER_IMAGE_TIMEOUT_SECONDS == 5


def test_upload_and_detect_skips_image_on_timeout(monkeypatch) -> None:
    """A 5-second timeout on one image must not abort the whole upload."""
    from unittest.mock import MagicMock

    import requests as _requests

    from ui import app as ui_app

    create_response = MagicMock()
    create_response.json.return_value = {"property_id": "prop-1"}
    create_response.raise_for_status = MagicMock()

    ok_image_response = MagicMock()
    ok_image_response.json.return_value = {
        "image": {
            "id": "img-1",
            "file_path": "/storage/ok.jpg",
            "room_type": "kitchen",
            "amenities": [{"amenity_name": "refrigerator", "is_present": True, "confidence": 0.9}],
        }
    }
    ok_image_response.raise_for_status = MagicMock()

    call_log: list[str] = []

    def fake_post(url: str, **kwargs):  # type: ignore[no-untyped-def]
        if url.endswith("/api/v1/properties/"):
            call_log.append("create")
            return create_response
        # Per-image POST: first call times out, second succeeds.
        files = kwargs.get("files") or {}
        filename = files["file"][0] if "file" in files else ""
        call_log.append(f"image:{filename}")
        if filename == "slow.jpg":
            raise _requests.exceptions.Timeout("simulated 5s timeout")
        return ok_image_response

    monkeypatch.setattr(ui_app.requests, "post", fake_post)

    file_obj_slow = MagicMock()
    file_obj_slow.name = "/tmp/slow.jpg"
    file_obj_ok = MagicMock()
    file_obj_ok.name = "/tmp/ok.jpg"

    def fake_open_upload_file(file_obj):  # type: ignore[no-untyped-def]
        if file_obj is file_obj_slow:
            return ("slow.jpg", b"x", "image/jpeg")
        return ("ok.jpg", b"y", "image/jpeg")

    monkeypatch.setattr(ui_app, "_open_upload_file", fake_open_upload_file)

    yields = list(
        ui_app.upload_and_detect(
            files=[file_obj_slow, file_obj_ok],
            property_name="Test Loft",
            model_name="openai/gpt-4o-mini",
            extra_info="",
            num_rooms=0,
            kitchen_hint="Not specified",
            balcony_hint="Not specified",
            living_room_hint="Not specified",
            expanded_hints=None,
            listing_metadata=None,
            progress=MagicMock(),
        )
    )

    assert call_log == ["create", "image:slow.jpg", "image:ok.jpg"]
    final_status = yields[-1][1]
    assert "1 image(s) were not processed" in final_status
    assert "slow.jpg" in final_status
    assert "timed out after 5s" in final_status
    assert "Detected" in final_status
