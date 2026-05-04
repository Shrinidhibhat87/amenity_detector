"""Browse page property card HTML helpers."""

from __future__ import annotations

import html
from typing import Any


def property_card_html(prop: dict[str, Any], api_base_url: str) -> str:
    """Render a property summary as one browse card."""
    name = html.escape(str(prop.get("name") or "Untitled property"))
    description = html.escape(str(prop.get("description") or "No description generated yet."))
    model = html.escape(str(prop.get("model_used") or "Unknown model"))
    image_count = int(prop.get("image_count") or 0)
    first_image_id = prop.get("first_image_id")
    prop_id = html.escape(str(prop.get("id") or ""))
    thumb = (
        f'<img loading="lazy" src="{api_base_url}/api/v1/images/{html.escape(str(first_image_id))}" alt="">'
        if first_image_id
        else '<div class="prop-thumb-empty">No image</div>'
    )
    return f"""
<article class="prop-card" data-name="{name.lower()}" data-id="{prop_id}">
  <div class="prop-thumb">{thumb}</div>
  <div class="prop-body">
    <div class="prop-top"><h3>{name}</h3><span>{image_count} images</span></div>
    <p>{description[:180]}</p>
    <div class="prop-meta"><span>{model}</span><code>{prop_id}</code></div>
  </div>
</article>
"""


def cards_grid_html(properties: list[dict[str, Any]], api_base_url: str) -> str:
    """Render a full property grid or empty state."""
    if not properties:
        return '<div class="empty-state">No properties match this view.</div>'
    cards = "".join(property_card_html(prop, api_base_url) for prop in properties)
    return f'<div class="prop-grid">{cards}</div>'
