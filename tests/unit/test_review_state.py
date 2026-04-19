"""Unit tests for ``ui.review_state`` — pure helpers behind the Phase 6 review panel.

Each helper is a pure function: given a state (list of room dicts) and an
operation, return a new state. These tests lock in the contract so the
``@gr.render`` handlers in ``ui/app.py`` can rely on it without re-testing
the mutation logic through the Gradio runtime.
"""

from __future__ import annotations

from ui.review_state import (
    add_item,
    amenities_for_describe,
    apply_reconciled_names,
    cancel_edit,
    confirm_item,
    flatten_for_reconcile,
    from_detections,
    reject_item,
    save_edit,
    start_edit,
)


def _image(filename: str, room: str, amenities: list[dict]) -> dict:
    """Shape that matches the FastAPI single-image response ``image`` block."""
    return {"file_path": f"/storage/{filename}", "room_type": room, "amenities": amenities}


# ── from_detections ──────────────────────────────────────────────────────────


def test_from_detections_groups_by_room() -> None:
    state = from_detections(
        [
            _image(
                "k.jpg",
                "kitchen",
                [
                    {"amenity_name": "refrigerator", "is_present": True, "confidence": 0.95},
                    {"amenity_name": "oven", "is_present": True, "confidence": 0.88},
                ],
            ),
            _image(
                "l.jpg",
                "living_room",
                [{"amenity_name": "sofa", "is_present": True, "confidence": 0.91}],
            ),
        ]
    )

    assert [room["room"] for room in state] == ["kitchen", "living_room"]
    assert len(state[0]["items"]) == 2
    assert state[0]["items"][0]["name"] == "refrigerator"
    assert state[0]["items"][0]["status"] == "pending"
    assert state[0]["items"][0]["confidence"] == 0.95
    # Every item gets a stable id
    assert all(item["id"] for room in state for item in room["items"])


def test_from_detections_merges_duplicate_room_blocks() -> None:
    """Two images both classified as 'kitchen' collapse into one kitchen group."""
    state = from_detections(
        [
            _image(
                "k1.jpg",
                "kitchen",
                [{"amenity_name": "oven", "is_present": True, "confidence": 0.9}],
            ),
            _image(
                "k2.jpg",
                "kitchen",
                [{"amenity_name": "sink", "is_present": True, "confidence": 0.8}],
            ),
        ]
    )
    assert len(state) == 1
    assert state[0]["room"] == "kitchen"
    assert [i["name"] for i in state[0]["items"]] == ["oven", "sink"]


def test_from_detections_normalises_missing_room_type() -> None:
    state = from_detections(
        [_image("x.jpg", "", [{"amenity_name": "lamp", "is_present": True, "confidence": 0.5}])]
    )
    assert state[0]["room"] == "unknown"


# ── confirm_item ─────────────────────────────────────────────────────────────


def test_confirm_item_flips_status_and_leaves_others() -> None:
    state = _sample_state()
    target = state[0]["items"][0]["id"]

    new_state = confirm_item(state, target)

    assert new_state[0]["items"][0]["status"] == "confirmed"
    assert new_state[0]["items"][1]["status"] == "pending"
    # Original state was not mutated
    assert state[0]["items"][0]["status"] == "pending"


def test_confirm_item_unknown_id_is_noop() -> None:
    state = _sample_state()
    assert confirm_item(state, "does-not-exist") == state


# ── reject_item ──────────────────────────────────────────────────────────────


def test_reject_item_removes_row() -> None:
    state = _sample_state()
    target = state[0]["items"][0]["id"]

    new_state = reject_item(state, target)

    assert [i["id"] for i in new_state[0]["items"]] == [state[0]["items"][1]["id"]]


def test_reject_item_removes_room_when_last_item_gone() -> None:
    state = from_detections(
        [
            _image(
                "x.jpg",
                "balcony",
                [{"amenity_name": "chair", "is_present": True, "confidence": 0.7}],
            )
        ]
    )
    target = state[0]["items"][0]["id"]

    new_state = reject_item(state, target)

    assert new_state == []


# ── start_edit / save_edit / cancel_edit ─────────────────────────────────────


def test_start_edit_flips_status() -> None:
    state = _sample_state()
    target = state[0]["items"][0]["id"]

    new_state = start_edit(state, target)

    assert new_state[0]["items"][0]["status"] == "editing"


def test_save_edit_writes_new_name_and_confirms() -> None:
    state = start_edit(_sample_state(), _sample_state()[0]["items"][0]["id"])
    target = state[0]["items"][0]["id"]

    new_state = save_edit(state, target, new_name=" Dishwasher ", present=False)

    assert new_state[0]["items"][0]["name"] == "Dishwasher"  # trimmed
    assert new_state[0]["items"][0]["present"] is False
    assert new_state[0]["items"][0]["status"] == "confirmed"


def test_save_edit_with_blank_name_cancels_instead_of_saving() -> None:
    state = _sample_state()
    target = state[0]["items"][0]["id"]
    state = start_edit(state, target)

    new_state = save_edit(state, target, new_name="   ", present=True)

    # Blank name is treated as cancel: the item reverts to its prior pending state.
    assert new_state[0]["items"][0]["status"] == "pending"
    assert new_state[0]["items"][0]["name"] == "refrigerator"


def test_cancel_edit_reverts_pending_row() -> None:
    state = _sample_state()
    target = state[0]["items"][0]["id"]
    state = start_edit(state, target)

    new_state = cancel_edit(state, target)

    assert new_state[0]["items"][0]["status"] == "pending"


def test_cancel_edit_removes_never_detected_row() -> None:
    """A row that was added by the user and never had a detection should
    disappear when they cancel, not linger in editing state forever."""
    state = add_item(_sample_state(), room="kitchen")
    new_id = state[0]["items"][-1]["id"]

    new_state = cancel_edit(state, new_id)

    assert all(i["id"] != new_id for i in new_state[0]["items"])


# ── add_item ─────────────────────────────────────────────────────────────────


def test_add_item_appends_editing_row_under_room() -> None:
    state = _sample_state()

    new_state = add_item(state, room="kitchen")

    new_item = new_state[0]["items"][-1]
    assert new_item["status"] == "editing"
    assert new_item["name"] == ""
    assert new_item["present"] is True
    assert new_item["confidence"] is None


def test_add_item_creates_room_if_missing() -> None:
    state = _sample_state()

    new_state = add_item(state, room="bedroom")

    rooms = [r["room"] for r in new_state]
    assert "bedroom" in rooms
    assert new_state[-1]["items"][0]["status"] == "editing"


# ── flatten_for_reconcile ────────────────────────────────────────────────────


def test_flatten_for_reconcile_uses_current_names_and_present() -> None:
    state = _sample_state()

    rows = flatten_for_reconcile(state)

    assert ["kitchen", "refrigerator", True, "0.95"] in rows
    assert ["kitchen", "oven", True, "0.88"] in rows


def test_flatten_for_reconcile_skips_editing_rows() -> None:
    state = add_item(_sample_state(), room="kitchen")

    rows = flatten_for_reconcile(state)

    assert len(rows) == 2  # the brand-new editing row is not in the output


# ── amenities_for_describe ───────────────────────────────────────────────────


def test_amenities_for_describe_includes_confirmed_and_present_pending() -> None:
    state = _sample_state()
    state = confirm_item(state, state[0]["items"][0]["id"])

    payload = amenities_for_describe(state)

    assert {"room_type": "kitchen", "amenity_name": "refrigerator", "is_present": True} in payload
    assert {"room_type": "kitchen", "amenity_name": "oven", "is_present": True} in payload


def test_amenities_for_describe_excludes_editing_rows() -> None:
    state = add_item(_sample_state(), room="kitchen")
    payload = amenities_for_describe(state)
    # The brand-new editing row has an empty name; it must not be sent to /describe.
    assert all(entry["amenity_name"] != "" for entry in payload)


def test_amenities_for_describe_excludes_pending_not_present() -> None:
    state = _sample_state()
    # The second pending item is present=False
    state[0]["items"][1]["present"] = False

    payload = amenities_for_describe(state)

    # Only the present pending item (refrigerator) should survive
    names = [entry["amenity_name"] for entry in payload]
    assert "refrigerator" in names
    assert "oven" not in names


def test_amenities_for_describe_strips_warning_prefix() -> None:
    from ui.helpers import WARNING_PREFIX

    state = _sample_state()
    state[0]["items"][0]["name"] = f"{WARNING_PREFIX}refrigerator"

    payload = amenities_for_describe(state)

    assert {"room_type": "kitchen", "amenity_name": "refrigerator", "is_present": True} in payload


# ── apply_reconciled_names ───────────────────────────────────────────────────


def test_apply_reconciled_names_writes_warning_prefix_back_into_state() -> None:
    from ui.helpers import WARNING_PREFIX

    state = _sample_state()
    rows = flatten_for_reconcile(state)
    rows[0][1] = f"{WARNING_PREFIX}{rows[0][1]}"

    new_state = apply_reconciled_names(state, rows)

    assert new_state[0]["items"][0]["name"] == f"{WARNING_PREFIX}refrigerator"
    assert new_state[0]["items"][1]["name"] == "oven"  # untouched


def test_apply_reconciled_names_skips_editing_rows() -> None:
    state = add_item(_sample_state(), room="kitchen")  # adds an editing row
    rows = flatten_for_reconcile(state)  # 2 rows (editing row skipped)

    new_state = apply_reconciled_names(state, rows)

    # Editing row still present, unchanged
    editing_rows = [i for block in new_state for i in block["items"] if i["status"] == "editing"]
    assert len(editing_rows) == 1


# ── helpers ──────────────────────────────────────────────────────────────────


def _sample_state() -> list[dict]:
    return from_detections(
        [
            _image(
                "k.jpg",
                "kitchen",
                [
                    {"amenity_name": "refrigerator", "is_present": True, "confidence": 0.95},
                    {"amenity_name": "oven", "is_present": True, "confidence": 0.88},
                ],
            )
        ]
    )
