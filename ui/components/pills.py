"""Segmented pill HTML helpers."""

from __future__ import annotations

import html

from ui.hints import CHOICES


def segmented_pill_html(
    group_id: str, choices: tuple[str, ...] = CHOICES, default: str = "Not specified"
) -> str:
    """Render one JS-wired segmented pill group."""
    buttons = []
    for choice in choices:
        active = " active" if choice == default else ""
        buttons.append(
            '<button type="button" '
            f'class="pill{active}" data-value="{html.escape(choice)}">'
            f"{html.escape(choice)}</button>"
        )
    return (
        f'<div class="pillgroup" data-group="{html.escape(group_id)}">'
        '<span class="pillbg"></span>'
        f"{''.join(buttons)}</div>"
    )


def amenity_hint_heading(group_name: str, label: str) -> str:
    """Render a compact hint label above each segmented pill group."""
    return (
        '<div class="hint-label">'
        f"<span>{html.escape(group_name)}</span>"
        f"<strong>{html.escape(label)}</strong>"
        "</div>"
    )


def hints_grid_html() -> str:
    """Render all amenity hint groups as a 4-column CSS grid (Issue 8)."""
    from ui.hints import AMENITY_GROUPS

    cols: list[str] = []
    for group_name, hints in AMENITY_GROUPS:
        items: list[str] = [f'<div class="room-col-header">{html.escape(group_name)}</div>']
        for key, label in hints:
            items.append(f'<div class="hint-label"><strong>{html.escape(label)}</strong></div>')
            items.append(segmented_pill_html(key))
        cols.append(f'<div class="room-col">{"".join(items)}</div>')
    return f'<div class="rooms-grid">{"".join(cols)}</div>'
