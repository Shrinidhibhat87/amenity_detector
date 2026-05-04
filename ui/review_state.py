"""Pure state helpers for the Phase 6 per-room review panel.

The Gradio UI (see ``ui/app.py``) holds a single ``gr.State`` whose value is
a list of ``{"room": str, "items": list[dict]}`` dicts. Every mutation goes
through one of the functions here and returns a *new* list — we never edit
the caller's state in place. This keeps Gradio's state diffing predictable
and makes the helpers trivial to unit-test without the Gradio runtime.

State shape
-----------
```
[
  {
    "room": "kitchen",
    "items": [
      {
        "id": "3f9a2c",          # stable key for @gr.render diffing
        "name": "refrigerator",
        "present": True,          # bool — what the UI shows
        "confidence": 0.95,       # float | None (None for user-added rows)
        "status": "pending",      # "pending" | "editing" | "confirmed"
      },
      ...
    ],
  },
  ...
]
```

``status`` is the single source of truth for how each row renders:

- ``pending``   — detected by the model, awaiting user action.
- ``editing``   — user clicked ✎ (or + Add amenity). Name is a textbox;
                  Present/Confidence are shown but disabled.
- ``confirmed`` — user clicked ✓. The row collapses to a green chip;
                  Present/Confidence are hidden.

Rejected rows are removed outright — there is no ``"rejected"`` status.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any

from ui.helpers import WARNING_PREFIX

# Type aliases for readability. These are also what ``ui/app.py`` imports.
Item = dict[str, Any]
RoomBlock = dict[str, Any]
State = list[RoomBlock]


# ── Construction ─────────────────────────────────────────────────────────────


def from_detections(images: list[dict[str, Any]]) -> State:
    """Build an initial review state from the API's per-image response list.

    Two images with the same ``room_type`` collapse into one room block so the
    review panel shows one card per physical room rather than one per image.
    """
    state: State = []
    for image in images:
        room = (image.get("room_type") or "unknown").strip() or "unknown"
        block = _find_room(state, room)
        if block is None:
            block = {"room": room, "items": []}
            state.append(block)
        existing_names = {i["name"].lower().strip() for i in block["items"]}
        for amenity in image.get("amenities", []):
            key = amenity["amenity_name"].lower().strip()
            if key in existing_names:
                continue
            existing_names.add(key)
            block["items"].append(
                {
                    "id": _new_id(),
                    "name": amenity["amenity_name"],
                    "present": bool(amenity["is_present"]),
                    "confidence": amenity.get("confidence"),
                    "status": "pending",
                }
            )
    return state


# ── Single-item mutations ────────────────────────────────────────────────────


def confirm_item(state: State, item_id: str) -> State:
    """Mark the item as user-confirmed. No-op if ``item_id`` isn't found."""
    return _map_item(state, item_id, lambda i: {**i, "status": "confirmed"})


def reject_item(state: State, item_id: str) -> State:
    """Remove the item. If the room is then empty, remove the room too."""
    new_state: State = []
    for block in state:
        kept = [i for i in block["items"] if i["id"] != item_id]
        if kept:
            new_state.append({"room": block["room"], "items": kept})
    return new_state


def start_edit(state: State, item_id: str) -> State:
    """Flip status to ``"editing"`` so the renderer shows the textbox row."""
    return _map_item(state, item_id, lambda i: {**i, "status": "editing"})


def save_edit(state: State, item_id: str, new_name: str, present: bool) -> State:
    """Persist the edit and flip status to ``"confirmed"``.

    A blank ``new_name`` is treated as a cancel: the user wiped the textbox,
    so we don't want to confirm an empty amenity. See ``cancel_edit`` for the
    shape of that branch.
    """
    cleaned = new_name.strip()
    if not cleaned:
        return cancel_edit(state, item_id)
    return _map_item(
        state,
        item_id,
        lambda i: {**i, "name": cleaned, "present": bool(present), "status": "confirmed"},
    )


def cancel_edit(state: State, item_id: str) -> State:
    """Undo an in-progress edit.

    For a row that was added by the user (no ``confidence`` recorded and a
    blank name), cancelling removes the row entirely — otherwise the user
    would be stuck with a nameless editing row they can't dismiss any other
    way.
    """
    new_state: State = []
    for block in state:
        kept: list[Item] = []
        for item in block["items"]:
            if item["id"] == item_id and _is_never_detected(item):
                continue  # drop it
            if item["id"] == item_id:
                kept.append({**item, "status": "pending"})
            else:
                kept.append(item)
        if kept:
            new_state.append({"room": block["room"], "items": kept})
    return new_state


def add_item(state: State, room: str) -> State:
    """Append a blank editing row under the given room (creating the room if
    it doesn't yet exist)."""
    new_state = copy.deepcopy(state)
    block = _find_room(new_state, room)
    new_item: Item = {
        "id": _new_id(),
        "name": "",
        "present": True,
        "confidence": None,
        "status": "editing",
    }
    if block is None:
        new_state.append({"room": room, "items": [new_item]})
    else:
        block["items"].append(new_item)
    return new_state


# ── Reads for upstream flows ─────────────────────────────────────────────────


def flatten_for_reconcile(state: State) -> list[list[Any]]:
    """Return rows compatible with ``ui.helpers.reconcile_amenities``.

    The reconcile helper expects ``[room, name, present, confidence_str]``.
    We skip ``"editing"`` rows because their name is typically empty and
    reconcile can't meaningfully flag them.
    """
    rows: list[list[Any]] = []
    for block in state:
        for item in block["items"]:
            if item["status"] == "editing":
                continue
            conf = item.get("confidence")
            rows.append(
                [
                    block["room"],
                    item["name"],
                    bool(item["present"]),
                    f"{conf:.2f}" if isinstance(conf, (int, float)) else "—",
                ]
            )
    return rows


def apply_reconciled_names(state: State, reconciled_rows: list[list[Any]]) -> State:
    """Write back the ``name`` field for each non-editing item from reconciled rows.

    The reconcile helper returns rows with ``[warning] `` prepended to names
    that contradict sidebar hints. ``flatten_for_reconcile`` produces rows in
    the same order as non-editing items in state, so we zip them back in the
    same order. Editing rows are left untouched (they were skipped on flatten).
    """
    new_state: State = []
    row_idx = 0
    for block in state:
        new_items: list[Item] = []
        for item in block["items"]:
            if item["status"] == "editing":
                new_items.append(item)
                continue
            if row_idx < len(reconciled_rows):
                reconciled_row = reconciled_rows[row_idx]
                if len(reconciled_row) > 1:
                    new_items.append({**item, "name": str(reconciled_row[1])})
                else:
                    new_items.append(item)
                row_idx += 1
            else:
                new_items.append(item)
        new_state.append({"room": block["room"], "items": new_items})
    return new_state


def amenities_for_describe(state: State) -> list[dict[str, Any]]:
    """Produce the ``amenities`` payload for ``POST /properties/{id}/describe``.

    Rules:
      - confirmed rows are always sent;
      - pending rows are sent only when ``present`` is True (the user tacitly
        accepts model predictions that say 'present' by hitting the global
        Confirm button);
      - editing rows are never sent (empty / half-typed names);
      - any ``[warning] `` prefix from reconcile is stripped before send.
    """
    payload: list[dict[str, Any]] = []
    for block in state:
        for item in block["items"]:
            if item["status"] == "editing":
                continue
            if item["status"] == "pending" and not item["present"]:
                continue
            name = str(item["name"]).removeprefix(WARNING_PREFIX).strip()
            if not name:
                continue
            payload.append(
                {
                    "room_type": block["room"],
                    "amenity_name": name,
                    "is_present": bool(item["present"]),
                }
            )
    return payload


# ── Internals ────────────────────────────────────────────────────────────────


def _find_room(state: State, room: str) -> RoomBlock | None:
    for block in state:
        if block["room"] == room:
            return block
    return None


def _map_item(state: State, item_id: str, fn: Any) -> State:
    """Return a new state where the item matching ``item_id`` is replaced
    with ``fn(item)``. Structure-preserving; no-op if ``item_id`` is absent."""
    new_state: State = []
    for block in state:
        new_items: list[Item] = []
        for item in block["items"]:
            new_items.append(fn(item) if item["id"] == item_id else item)
        new_state.append({"room": block["room"], "items": new_items})
    return new_state


def _is_never_detected(item: Item) -> bool:
    """True for items the user added via + and then abandoned."""
    return item.get("confidence") is None and not str(item.get("name", "")).strip()


def _new_id() -> str:
    """Short stable id; 8 hex chars is plenty for at most a few hundred rows."""
    return uuid.uuid4().hex[:8]
