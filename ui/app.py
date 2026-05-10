"""
Gradio frontend for the Amenity Detector — Phase 3 (redesigned).

Architecture:
  - gr.HTML block at the top renders the landing/hero page.
  - Two gr.Tab blocks below handle Upload and Browse.
  - UI communicates with the FastAPI backend via HTTP only (no imports from api/).
  - gr.State stores the property_id and model between upload and describe calls.

Environment variables:
  API_BASE_URL  — FastAPI backend URL (default: http://localhost:8000)
  GRADIO_PORT   — Port to bind on (default: 7860)
"""

import html
import json
import logging
import os
from typing import Any, cast

# Keep ``build_app()`` smoke tests deterministic and avoid Gradio background
# telemetry threads in CI/local test runs.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import requests

from core.logging_config import setup_logging
from ui.components.cards import cards_grid_html
from ui.components.orbs import bg_orbs_html
from ui.components.pills import hints_grid_html
from ui.components.stepper import step_html
from ui.helpers import WARNING_PREFIX, reconcile_amenities
from ui.hints import AMENITY_GROUPS, HINT_KEYS, hints_payload
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
from ui.theme import CSS, HEAD
from ui.view_router import BROWSE, HOME, UPLOAD, go_to

# Configure structured logging so UI exceptions are captured in `docker compose
# logs ui` with a level, module and full traceback — the UI process is a
# separate container from the API, so API Prometheus metrics do not see these
# errors. See docs/error_logs.md (question 4) for a longer explanation.
setup_logging()
logger = logging.getLogger("ui")

# ── Configuration ──────────────────────────────────────────────────────────────
_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
_TIMEOUT_SECONDS = 300  # VLM inference can be slow on local GPU


# ── Helper functions ───────────────────────────────────────────────────────────


def _get_available_models() -> list[str]:
    """
    Fetch available model names from the FastAPI /api/v1/models/ endpoint.

    Falls back to a hardcoded list if the backend is unreachable (e.g. cold start).

    Returns:
        List of model name strings, e.g. ["openai/gpt-4o-mini", "google/gemini-pro-1.5"].
    """
    try:
        response = requests.get(f"{_API_BASE_URL}/api/v1/models/", timeout=10)
        response.raise_for_status()
        models_data: list[dict[str, Any]] = response.json()
        return [m["name"] for m in models_data if m.get("available")]
    except Exception:
        return [
            "openai/gpt-4o-mini",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3.2-11b-vision-instruct",
            "qwen/qwen2-vl-72b-instruct",
        ]


def _format_amenities_table(images: list[dict[str, Any]]) -> list[list[Any]]:
    """
    Flatten detected amenities from the upload API response into table rows.

    Each row represents one amenity detection for one image. The 'Present'
    column is a bool so Gradio renders it as a checkbox (interactive=True).

    Args:
        images: The ``images`` list from the PropertyDetailResponse JSON.

    Returns:
        List of rows: [room_type, amenity_name, is_present (bool), confidence (str)]
    """
    rows: list[list[Any]] = []
    for img in images:
        room = img.get("room_type") or "unknown"
        for amenity in img.get("amenities", []):
            confidence = amenity.get("confidence")
            rows.append(
                [
                    room,
                    amenity["amenity_name"],
                    bool(amenity["is_present"]),
                    f"{confidence:.2f}" if confidence is not None else "—",
                ]
            )
    return rows


def _format_single_image_table(image: dict[str, Any]) -> list[list[Any]]:
    """
    Convert one image response into amenity table rows.

    Args:
        image: The ``image`` object from the per-image detection response.

    Returns:
        List of rows matching the editable amenity table.
    """
    return _format_amenities_table([image])


def _open_upload_file(file_obj: Any) -> tuple[str, bytes, str]:
    """
    Read a Gradio file object into the tuple shape requests expects.

    Returns:
        (filename, content_bytes, mime_type)
    """
    file_path = (
        file_obj
        if isinstance(file_obj, str)
        else (file_obj.name if hasattr(file_obj, "name") else str(file_obj))
    )
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as fh:
        content = fh.read()
    ext = filename.rsplit(".", 1)[-1].lower()
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")
    return filename, content, mime


def _hint_to_bool(value: str | None) -> bool | None:
    """
    Translate UI tri-state radio strings to bool | None for the API.
    """
    if value == "Yes":
        return True
    if value == "No":
        return False
    return None


def _sidebar_payload(
    num_rooms: int | float | None,
    kitchen_hint: str | None,
    balcony_hint: str | None,
    living_room_hint: str | None,
    expanded_hints: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """
    Build the sidebar hint payload used by reconciliation and /describe.
    """
    parsed_rooms = int(num_rooms or 0)
    hints = expanded_hints or {}
    return {
        "num_rooms": parsed_rooms if parsed_rooms > 0 else None,
        "has_kitchen": hints.get("kitchen", _hint_to_bool(kitchen_hint)),
        "has_balcony": hints.get("balcony", _hint_to_bool(balcony_hint)),
        "has_living_room": hints.get("living_room", _hint_to_bool(living_room_hint)),
        "hints": hints,
    }


def _expanded_hints_from_sequence(values: tuple[str | None, ...]) -> dict[str, bool]:
    """Build the sparse expanded hints dict from ordered hidden textbox values."""
    return hints_payload(dict(zip(HINT_KEYS, values, strict=False)))


def _listing_metadata_payload(
    listing_type: str | None,
    price: float | int | None,
    currency: str | None,
    price_period: str | None,
    property_type: str | None,
    furnishing: str | None,
    num_bedrooms: int | float | None,
    num_bathrooms: int | float | None,
    area_sqm: float | int | None,
    available_from: str | None,
    locality: str | None,
    postal_code: str | None,
    country_code: str | None,
    owner_email: str | None,
) -> dict[str, Any]:
    """
    Build the optional Phase 9 listing-metadata block for ``POST /api/v1/properties/``.

    Drops any field the user did not actually fill in, so the API layer's
    ``model_dump(exclude_unset=True)`` keeps the corresponding columns NULL.
    Numeric ``0`` is treated as "unset" because Gradio's ``gr.Number`` returns
    ``0`` when the field is left blank — passing literal ``0`` would force a
    misleading row value on every upload.
    """

    out: dict[str, Any] = {}

    def _put_str(key: str, value: str | None, *, upper: bool = False) -> None:
        if value is None:
            return
        cleaned = value.strip()
        if not cleaned:
            return
        out[key] = cleaned.upper() if upper else cleaned

    def _put_choice(key: str, value: str | None) -> None:
        # Gradio Dropdowns may emit None or the empty string for "no selection".
        if value is None or value == "" or value == "—":
            return
        out[key] = value

    def _put_number(key: str, value: int | float | None) -> None:
        if value is None or value == 0:
            return
        out[key] = value

    _put_choice("listing_type", listing_type)
    _put_number("price", price)
    _put_choice("currency", currency)
    _put_choice("price_period", price_period)
    _put_choice("property_type", property_type)
    _put_choice("furnishing", furnishing)
    _put_number("num_bedrooms", int(num_bedrooms) if num_bedrooms else None)
    _put_number("num_bathrooms", int(num_bathrooms) if num_bathrooms else None)
    _put_number("area_sqm", area_sqm)
    _put_str("available_from", available_from)
    _put_str("locality", locality)
    _put_str("postal_code", postal_code)
    _put_str("country_code", country_code, upper=True)
    _put_str("owner_email", owner_email)

    return out


def _strip_warning_prefix(value: Any) -> str:
    """Remove UI-only warning markers before sending data back to the API."""
    return str(value).removeprefix(WARNING_PREFIX)


def _to_rows(table: Any) -> list[list[Any]]:
    """
    Normalise whatever Gradio hands us into a plain list of row lists.

    Gradio's Dataframe can deliver a pandas DataFrame even when type="array" is
    set on the component (older handler paths, internal reconciliation calls,
    etc.). Evaluating ``df or []`` raises ``ValueError: The truth value of a
    DataFrame is ambiguous``, so we convert explicitly and treat None/empty as
    an empty list.
    """
    if table is None:
        return []
    to_values = getattr(table, "values", None)
    if to_values is not None and hasattr(to_values, "tolist"):
        return [list(row) for row in to_values.tolist()]
    if isinstance(table, list):
        return [list(row) for row in table]
    return list(table)


# ── CSS ────────────────────────────────────────────────────────────────────────
# Two palettes: warm cream/amber for light mode, dark stone/amber for dark mode.
# CSS custom properties are set per-media-query so the UI feels native in both.
# The amber accent (#b45309 light / #f59e0b dark) is the only colour shared.

_CSS = """
/* ─── SHARED: font + amber accent ─────────────────────────────── */
.gradio-container, .gradio-container * {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}
button.primary, .btn-primary {
    font-weight: 700 !important;
    border-radius: 9px !important;
}
button.primary:hover, .btn-primary:hover {
    transform: translateY(-1px) !important;
}
button.secondary {
    font-weight: 700 !important;
    border-radius: 9px !important;
    border-width: 2px !important;
    border-style: solid !important;
}

/* ─── LIGHT MODE ───────────────────────────────────────────────── */
:root {
    --ad-bg:          #fafaf7;
    --ad-bg2:         #f0ede4;
    --ad-border:      #e5e0d8;
    --ad-text:        #1c1917;
    --ad-text-mid:    #57534e;
    --ad-text-light:  #78716c;
    --ad-amber:       #b45309;
    --ad-amber-hover: #92400e;
    --ad-amber-bg:    rgba(180,83,9,0.10);
    --ad-yellow:      #fef3c7;
    --ad-chip-bg:     #ffffff;
    --ad-sec-btn-bg:  #ffffff;
}

/* Override Gradio CSS variables for light mode */
:root {
    --body-background-fill:              #fafaf7 !important;
    --background-fill-primary:           #fafaf7 !important;
    --background-fill-secondary:         #f0ede4 !important;
    --border-color-primary:              #e5e0d8 !important;
    --body-text-color:                   #1c1917 !important;
    --block-label-text-color:            #57534e !important;
    --block-title-text-color:            #1c1917 !important;
    --input-background-fill:             #fafaf7 !important;
    --block-background-fill:             #fafaf7 !important;
    --button-primary-background-fill:    #b45309 !important;
    --button-primary-text-color:         #ffffff !important;
    --button-secondary-background-fill:  #ffffff !important;
    --button-secondary-text-color:       #b45309 !important;
    --button-secondary-border-color:     #b45309 !important;
    --checkbox-background-color-selected:#b45309 !important;
    --table-row-focus:                   #fef3c7 !important;
}

/* ─── DARK MODE ────────────────────────────────────────────────── */
@media (prefers-color-scheme: dark) {
    :root {
        --ad-bg:          #1c1917;
        --ad-bg2:         #292524;
        --ad-border:      #44403c;
        --ad-text:        #fafaf7;
        --ad-text-mid:    #d6d3d1;
        --ad-text-light:  #a8a29e;
        --ad-amber:       #f59e0b;
        --ad-amber-hover: #d97706;
        --ad-amber-bg:    rgba(245,158,11,0.15);
        --ad-yellow:      rgba(245,158,11,0.12);
        --ad-chip-bg:     #292524;
        --ad-sec-btn-bg:  #292524;
    }

    /* Override Gradio CSS variables for dark mode */
    :root {
        --body-background-fill:              #1c1917 !important;
        --background-fill-primary:           #1c1917 !important;
        --background-fill-secondary:         #292524 !important;
        --border-color-primary:              #44403c !important;
        --body-text-color:                   #fafaf7 !important;
        --block-label-text-color:            #d6d3d1 !important;
        --block-title-text-color:            #fafaf7 !important;
        --input-background-fill:             #292524 !important;
        --block-background-fill:             #292524 !important;
        --button-primary-background-fill:    #f59e0b !important;
        --button-primary-text-color:         #1c1917 !important;
        --button-secondary-background-fill:  #292524 !important;
        --button-secondary-text-color:       #f59e0b !important;
        --button-secondary-border-color:     #f59e0b !important;
        --checkbox-background-color-selected:#f59e0b !important;
        --table-row-focus:                   rgba(245,158,11,0.12) !important;
    }
}

/* Gradio also applies a .dark class on <body> via JS (matchMedia), which can
   fire independently of the CSS @media query above. Mirror BOTH the hero's
   --ad-* custom properties AND the Gradio variables here so the hero stays in
   sync with the rest of the UI whenever dark mode is active.                 */
.dark {
    /* Hero palette — must match the @media (prefers-color-scheme: dark) block */
    --ad-bg:          #1c1917;
    --ad-bg2:         #292524;
    --ad-border:      #44403c;
    --ad-text:        #fafaf7;
    --ad-text-mid:    #d6d3d1;
    --ad-text-light:  #a8a29e;
    --ad-amber:       #f59e0b;
    --ad-amber-hover: #d97706;
    --ad-amber-bg:    rgba(245,158,11,0.15);
    --ad-yellow:      rgba(245,158,11,0.12);
    --ad-chip-bg:     #292524;
    --ad-sec-btn-bg:  #292524;

    --body-background-fill:              #1c1917 !important;
    --background-fill-primary:           #1c1917 !important;
    --background-fill-secondary:         #292524 !important;
    --border-color-primary:              #44403c !important;
    --body-text-color:                   #fafaf7 !important;
    --block-label-text-color:            #d6d3d1 !important;
    --block-title-text-color:            #fafaf7 !important;
    --input-background-fill:             #292524 !important;
    --block-background-fill:             #292524 !important;
    --button-primary-background-fill:    #f59e0b !important;
    --button-primary-text-color:         #1c1917 !important;
    --button-secondary-background-fill:  #292524 !important;
    --button-secondary-text-color:       #f59e0b !important;
    --button-secondary-border-color:     #f59e0b !important;
    --checkbox-background-color-selected:#f59e0b !important;
    --table-row-focus:                   rgba(245,158,11,0.12) !important;
}

/* ─── COMPONENT STYLES (use CSS vars, work in both modes) ────── */
.gradio-container {
    background: var(--ad-bg) !important;
}
input, textarea, select {
    background: var(--ad-bg) !important;
    color: var(--ad-text) !important;
    border: 1.5px solid var(--ad-border) !important;
    border-radius: 8px !important;
}
input:focus, textarea:focus { border-color: var(--ad-amber) !important; outline: none !important; }

label span, .label-wrap span, .block label span {
    color: var(--ad-text-mid) !important;
    font-weight: 600 !important; font-size: 12px !important;
    text-transform: uppercase !important; letter-spacing: 0.04em !important;
}

.tab-nav button { font-weight: 600 !important; color: var(--ad-text-light) !important; background: transparent !important; }
.tab-nav button.selected { color: var(--ad-amber) !important; border-bottom: 2px solid var(--ad-amber) !important; }

table, .table-wrap { background: var(--ad-bg) !important; color: var(--ad-text) !important; }
th { background: var(--ad-bg2) !important; color: var(--ad-text-mid) !important; font-weight: 700 !important; }
tr:hover td { background: var(--ad-yellow) !important; }

.dropdown-arrow, ul.options, ul.options li {
    background: var(--ad-bg) !important; color: var(--ad-text) !important; border-color: var(--ad-border) !important;
}

button.primary  { background: var(--ad-amber) !important; color: var(--button-primary-text-color, #fff) !important; border: none !important; box-shadow: 0 3px 12px rgba(0,0,0,0.2) !important; }
button.primary:hover  { background: var(--ad-amber-hover) !important; box-shadow: 0 5px 18px rgba(0,0,0,0.3) !important; }
button.secondary { background: var(--ad-sec-btn-bg) !important; color: var(--ad-amber) !important; border-color: var(--ad-amber) !important; }
button.secondary:hover { background: var(--ad-yellow) !important; transform: translateY(-1px) !important; }
"""

# ── Landing page HTML ──────────────────────────────────────────────────────────
# Injected as gr.HTML at the top of the app. Styles use CSS custom properties
# (--ad-*) that are defined differently for light and dark mode in _CSS above,
# so the hero always matches the surrounding Gradio palette.
#
# Gradio sanitises inline <script> blocks out of gr.HTML content, so the old
# typewriter animation never actually ran. We now render two static subtitle
# lines instead; revisit this when we move off Gradio (see SPEC.md Phase 7).
_HERO_HTML = """
<style>
/* Hero uses the same CSS custom properties defined in the Gradio _CSS block  */
.ad-hero {
    background: linear-gradient(160deg, var(--ad-bg, #fafaf7) 0%, var(--ad-bg2, #f0ede4) 100%);
    padding: 60px 24px 48px;
    text-align: center;
    border-bottom: 1px solid var(--ad-border, #e5e0d8);
    font-family: 'Segoe UI', system-ui, sans-serif;
}

/* Elements start VISIBLE (no opacity:0 default).
   The .ad-animated class is added by JS after a short delay so users with
   prefers-reduced-motion or slow JS still see the content immediately.     */
.ad-badge {
    display: inline-block;
    background: var(--ad-amber-bg, rgba(180,83,9,0.1));
    color: var(--ad-amber, #b45309);
    font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 5px 14px; border-radius: 100px;
    margin-bottom: 22px;
}
.ad-title {
    font-size: clamp(26px, 4vw, 46px); font-weight: 800;
    color: var(--ad-text, #1c1917);
    margin: 0 0 16px;
}
.ad-sub {
    font-size: 16px; color: var(--ad-text-light, #78716c);
    line-height: 1.6; margin: 0 auto 32px; max-width: 640px;
}
.ad-sub .ad-sub-line { display: block; }
.ad-sub .ad-sub-coming { color: var(--ad-amber, #b45309); font-style: italic; }

.ad-chips {
    display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-top: 36px;
}
.ad-chip {
    background: var(--ad-chip-bg, #fff);
    border: 1px solid var(--ad-border, #e5e0d8);
    color: var(--ad-text-mid, #57534e);
    font-size: 12px; font-weight: 500; padding: 5px 13px; border-radius: 100px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

</style>

<div class="ad-hero" id="ad-hero">
  <div class="ad-badge">✦ AI-Powered Property Analysis</div>
  <div class="ad-title">Smart Property Amenity Detection</div>
  <div class="ad-sub">
    <span class="ad-sub-line">AI-powered image analysis for real estate listings and property management.</span>
    <span class="ad-sub-line ad-sub-coming">Voice-based property search — coming soon.</span>
  </div>

  <div class="ad-chips">
    <span class="ad-chip">🏠 Room Classification</span>
    <span class="ad-chip">🛋️ Amenity Detection</span>
    <span class="ad-chip">✏️ Editable Results</span>
    <span class="ad-chip">📝 AI Descriptions</span>
    <span class="ad-chip">🎤 Voice Search <em style="color:var(--ad-amber,#b45309)">(coming soon)</em></span>
  </div>
</div>
"""

# ── Step indicator HTML ────────────────────────────────────────────────────────


def _step_html(active: int) -> str:
    """Build the 4-step progress indicator HTML for the Upload page."""
    return step_html(active)


# ── Upload tab handlers ────────────────────────────────────────────────────────


def _reconciled_review_state(
    images: list[dict[str, Any]],
    sidebar: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build a review state from API images and apply sidebar-hint warnings.

    Goes through the state → flatten → reconcile → apply-back round trip so
    the review panel shows ``[warning] `` prefixes on amenities that contradict
    user-provided hints, the same way the old table did.
    """
    review_state = from_detections(images)
    reconciled_rows = reconcile_amenities(sidebar, flatten_for_reconcile(review_state))
    return apply_reconciled_names(review_state, reconciled_rows)


def upload_and_detect(
    files: list[Any] | None,
    property_name: str,
    model_name: str,
    extra_info: str,
    num_rooms: int | float | None,
    kitchen_hint: str | None,
    balcony_hint: str | None,
    living_room_hint: str | None,
    expanded_hints: dict[str, bool] | None = None,
    listing_metadata: dict[str, Any] | None = None,
    progress: gr.Progress = gr.Progress(),
):
    """
    Handle 'Upload & Detect Amenities' button click.

    Creates a property shell, then calls the single-image endpoint once per
    image. This lets Gradio update the progress bar and review panel
    incrementally. Yields five values matching ``upload_btn.click`` outputs:

      - step_html:      Updated step indicator.
      - status_text:    Short status / error message.
      - review_state:   List of room blocks for the @gr.render panel.
      - description:    Empty at this stage.
      - upload_state:   Dict with {property_id, model_name} for the confirm step.
    """
    empty_upload_state: dict[str, str] = {}
    empty_review: list[dict[str, Any]] = []

    if not files:
        yield (
            _step_html(1),
            "Please upload at least one image.",
            empty_review,
            "",
            empty_upload_state,
        )
        return
    if not property_name.strip():
        yield _step_html(1), "Please enter a property name.", empty_review, "", empty_upload_state
        return
    if not model_name:
        yield _step_html(1), "Please select a model.", empty_review, "", empty_upload_state
        return

    sidebar = _sidebar_payload(
        num_rooms, kitchen_hint, balcony_hint, living_room_hint, expanded_hints
    )
    seen_images: list[dict[str, Any]] = []
    property_id = ""
    try:
        post_body: dict[str, Any] = {
            "name": property_name.strip(),
            "model_name": model_name,
            "extra_info": extra_info.strip() or None,
        }
        if listing_metadata:
            post_body.update(listing_metadata)
        response = requests.post(
            f"{_API_BASE_URL}/api/v1/properties/",
            json=post_body,
            timeout=30,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        property_id = data.get("property_id", "")
        upload_state = {"property_id": property_id, "model_name": model_name}

        total = len(files)
        for index, file_obj in enumerate(files, start=1):
            filename, content, mime = _open_upload_file(file_obj)
            progress((index - 1) / total, desc=f"Processing {filename} ({index}/{total})")

            image_response = requests.post(
                f"{_API_BASE_URL}/api/v1/properties/{property_id}/images",
                data={"model_name": model_name},
                files={"file": (filename, content, mime)},
                timeout=_TIMEOUT_SECONDS,
            )
            image_response.raise_for_status()

            image_data: dict[str, Any] = image_response.json().get("image", {})
            seen_images.append(image_data)
            review_state = _reconciled_review_state(seen_images, sidebar)
            status = (
                f"{index}/{total} images processed. You can keep editing the hints "
                "while detection continues."
            )
            yield _step_html(2), status, review_state, "", upload_state

        progress(1.0, desc="Detection complete")
        review_state = _reconciled_review_state(seen_images, sidebar)
        total_items = sum(len(block["items"]) for block in review_state)
        status = (
            f"Detected {total_items} amenities for '{property_name}'. "
            "Review per-room below, then click Confirm."
        )
        yield _step_html(3), status, review_state, "", upload_state

    except requests.exceptions.Timeout:
        logger.warning(
            "Upload timed out after %ds for property '%s'", _TIMEOUT_SECONDS, property_name
        )
        yield (
            _step_html(1),
            "Request timed out. Try a faster model or fewer images.",
            empty_review,
            "",
            empty_upload_state,
        )
    except requests.exceptions.ConnectionError:
        logger.warning("Cannot reach API at %s during upload", _API_BASE_URL)
        yield (
            _step_html(1),
            f"Cannot reach the API at {_API_BASE_URL}. Is the backend running?",
            empty_review,
            "",
            empty_upload_state,
        )
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            error_response = exc.response
            if error_response is not None:
                detail = error_response.json().get("detail", "")
        except Exception:
            pass
        logger.warning(
            "Upload failed (HTTP %s): %s", getattr(exc.response, "status_code", "?"), detail or exc
        )
        yield _step_html(1), f"Upload failed: {detail or exc}", empty_review, "", empty_upload_state
    except Exception as exc:
        # logger.exception includes the full traceback so that the real root
        # cause is visible in `docker compose logs ui` instead of just a
        # one-line "Unexpected error".
        logger.exception("Unexpected error in upload_and_detect for property '%s'", property_name)
        yield _step_html(1), f"Unexpected error: {exc}", empty_review, "", empty_upload_state


def confirm_and_describe(
    review_state: list[dict[str, Any]],
    upload_state: dict[str, str],
    num_rooms: int | float | None,
    kitchen_hint: str | None,
    balcony_hint: str | None,
    living_room_hint: str | None,
    expanded_hints: dict[str, bool] | None = None,
) -> tuple[str, str, str]:
    """
    Handle 'Confirm & Generate Description' button click.

    Takes the current review state (from the @gr.render per-room panel) and
    calls POST /api/v1/properties/{id}/describe to get a fresh VLM description
    based on confirmed amenities plus present pending ones.

    Returns:
        Tuple of 3 values:
          - step_html:    Step indicator at step 3 (Generate).
          - status_text:  Status message.
          - description:  The newly generated description.
    """
    property_id = upload_state.get("property_id", "")
    model_name = upload_state.get("model_name", "openai/gpt-4o-mini")

    if not property_id:
        return _step_html(3), "Upload images first before generating a description.", ""

    amenities = amenities_for_describe(review_state or [])
    sidebar = _sidebar_payload(
        num_rooms, kitchen_hint, balcony_hint, living_room_hint, expanded_hints
    )

    try:
        response = requests.post(
            f"{_API_BASE_URL}/api/v1/properties/{property_id}/describe",
            json={"amenities": amenities, "model_name": model_name, **sidebar},
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.warning("Describe call timed out for property %s", property_id)
        return _step_html(3), "Description generation timed out. Try again.", ""
    except requests.exceptions.ConnectionError:
        logger.warning("Cannot reach API at %s during describe", _API_BASE_URL)
        return _step_html(3), f"Cannot reach API at {_API_BASE_URL}.", ""
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = response.json().get("detail", "")
        except Exception:
            pass
        logger.warning(
            "Describe failed for property %s (HTTP %s): %s",
            property_id,
            getattr(exc.response, "status_code", "?"),
            detail or exc,
        )
        return _step_html(3), f"Description failed: {detail or exc}", ""
    except Exception as exc:
        logger.exception("Unexpected error in confirm_and_describe for property %s", property_id)
        return _step_html(3), f"Unexpected error: {exc}", ""

    description: str = response.json().get("description", "")
    return _step_html(3), "Description generated. You can edit it below before saving.", description


def reconcile_state_from_hints(
    num_rooms: int | float | None,
    kitchen_hint: str | None,
    balcony_hint: str | None,
    living_room_hint: str | None,
    review_state: list[dict[str, Any]] | None,
    expanded_hints: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """
    Re-run mismatch markers across the review state when sidebar hints change.

    Flattens non-editing items to the row shape reconcile expects, runs the
    warning-prefix logic, and writes the updated names back into state.
    """
    sidebar = _sidebar_payload(
        num_rooms, kitchen_hint, balcony_hint, living_room_hint, expanded_hints
    )
    state = review_state or []
    reconciled_rows = reconcile_amenities(sidebar, flatten_for_reconcile(state))
    return apply_reconciled_names(state, reconciled_rows)


def reconcile_table_from_hints(
    num_rooms: int | float | None,
    kitchen_hint: str | None,
    balcony_hint: str | None,
    living_room_hint: str | None,
    amenity_table: Any,
) -> list[list[Any]]:
    """
    Backward-compatible dataframe reconciliation helper.

    Phase 6 moved the user-facing review UI from a flat ``gr.Dataframe`` to a
    structured per-room state. Keep this small adapter because it locks in the
    original pandas truthiness regression: Gradio may hand a handler a pandas
    DataFrame, and callers must normalise it explicitly instead of doing
    ``amenity_table or []``.
    """
    sidebar = _sidebar_payload(num_rooms, kitchen_hint, balcony_hint, living_room_hint)
    return reconcile_amenities(sidebar, _to_rows(amenity_table))


# ── Browse tab handlers ────────────────────────────────────────────────────────


def search_properties(amenity_query: str) -> list[list[Any]]:
    """
    Search properties by amenity name.

    Args:
        amenity_query: Comma-separated amenity names.

    Returns:
        Table rows: [id, name, description preview, model, image count].
    """
    if not amenity_query.strip():
        return []
    try:
        response = requests.get(
            f"{_API_BASE_URL}/api/v1/properties/search",
            params={"amenities": amenity_query.strip()},
            timeout=30,
        )
        response.raise_for_status()
    except Exception:
        return []

    return [
        [
            p.get("id", ""),
            p.get("name", ""),
            (p.get("description") or "")[:120],
            p.get("model_used", ""),
            p.get("image_count", 0),
        ]
        for p in response.json()
    ]


def list_all_properties() -> list[list[Any]]:
    """Fetch all properties for the Browse tab default view."""
    try:
        response = requests.get(
            f"{_API_BASE_URL}/api/v1/properties/",
            params={"limit": 50},
            timeout=30,
        )
        response.raise_for_status()
    except Exception:
        return []

    return [
        [
            p.get("id", ""),
            p.get("name", ""),
            (p.get("description") or "")[:120],
            p.get("model_used", ""),
            p.get("image_count", 0),
        ]
        for p in response.json()
    ]


def _fetch_property_summaries(amenity_query: str | None = None) -> list[dict[str, Any]]:
    """Fetch browse summaries for card rendering."""
    try:
        if amenity_query and amenity_query.strip():
            response = requests.get(
                f"{_API_BASE_URL}/api/v1/properties/search",
                params={"amenities": amenity_query.strip()},
                timeout=30,
            )
        else:
            response = requests.get(
                f"{_API_BASE_URL}/api/v1/properties/",
                params={"limit": 50},
                timeout=30,
            )
        response.raise_for_status()
        return cast(list[dict[str, Any]], response.json())
    except Exception:
        return []


def list_property_cards() -> str:
    """Fetch all properties and render browse cards."""
    return cards_grid_html(_fetch_property_summaries(), _API_BASE_URL)


def search_property_cards(amenity_query: str) -> str:
    """Fetch matching properties and render browse cards."""
    return cards_grid_html(_fetch_property_summaries(amenity_query), _API_BASE_URL)


def get_property_detail(property_id: str) -> str:
    """
    Fetch and render full details for a single property as Markdown.

    Args:
        property_id: UUID of the property to look up.

    Returns:
        Markdown-formatted string with rooms, amenities, and description.
    """
    if not property_id.strip():
        return "Enter a property ID above to view details."
    try:
        response = requests.get(
            f"{_API_BASE_URL}/api/v1/properties/{property_id.strip()}",
            timeout=30,
        )
        if response.status_code == 404:
            return f"Property `{property_id}` not found."
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return f"Cannot reach the API at {_API_BASE_URL}."
    except Exception as exc:
        return f"Error: {exc}"

    prop: dict[str, Any] = response.json()
    lines: list[str] = [
        f"## {prop.get('name', 'Unknown')}",
        f"**ID:** `{prop.get('id', '')}` · **Model:** {prop.get('model_used', '—')} · **Created:** {prop.get('created_at', '—')}",
    ]
    if prop.get("extra_info"):
        lines.append(f"**Notes:** {prop['extra_info']}")
    if prop.get("description"):
        lines.append(f"\n**Description:**\n{prop['description']}")

    lines.append("\n---\n### Amenities by Room")
    for img in prop.get("images", []):
        filename = img.get("file_path", "").split("/")[-1]
        room = img.get("room_type") or "unknown"
        lines.append(f"\n**{filename}** — {room}")
        present = [a["amenity_name"] for a in img.get("amenities", []) if a.get("is_present")]
        absent = [a["amenity_name"] for a in img.get("amenities", []) if not a.get("is_present")]
        if present:
            lines.append("Present: " + ", ".join(present))
        if absent:
            lines.append("Not detected: " + ", ".join(absent))

    return "\n".join(lines)


# ── Review panel rendering ─────────────────────────────────────────────────────


def _humanise_room(room: str) -> str:
    """Turn 'living_room' into 'Living Room' for display."""
    return room.replace("_", " ").title() if room else "Unknown"


def _render_review_panel(state: list[dict[str, Any]], state_component: gr.State) -> None:
    """
    Render the per-room review cards inside an @gr.render decorator.

    Each room becomes a ``gr.Group`` card. Items render as one of three row
    shapes based on ``status``:

      - pending:   name + confidence + ✓ / ✎ / ✗ action buttons
      - editing:   textbox + Present checkbox + save / cancel buttons
      - confirmed: compact green chip (kept under its room so the mapping stays
                   visible — see the design spec for the reasoning)

    Handlers are declared inside this function; they call ``ui.review_state``
    helpers and write the new state back into ``state_component``. The
    ``item_id`` is bound as a default argument so each lambda closes over the
    right row (the usual loop-variable-capture trap).
    """
    # Keep the panel dynamic, but do not create dynamic Gradio event listeners
    # here. In Gradio 6.12 those listeners can become inert after rerendering.
    # Buttons below are plain HTML and delegate to one static hidden Gradio
    # button declared in build_app().
    gr.HTML(_review_panel_html(state or []))


def _review_panel_html(state: list[dict[str, Any]]) -> str:
    """Render review state as HTML controlled by one static JS event delegate."""
    if not state:
        return '<p class="ad-review-empty">No amenities yet. Upload images to start detection.</p>'

    cards: list[str] = []
    for block in state:
        room = str(block["room"])
        room_label = _humanise_room(room)
        rows = "".join(_review_item_html(item) for item in block.get("items", []))
        add_payload = html.escape(json.dumps({"action": "add", "room": room}))
        cards.append(
            '<section class="room-card">'
            f"<h4>🏠 {html.escape(room_label)}</h4>"
            f'<div class="amenity-list">{rows}</div>'
            f'<button type="button" class="review-add-btn" data-review-payload="{add_payload}">'
            "+ Add amenity"
            "</button>"
            "</section>"
        )
    return f'<div class="review-room-grid">{"".join(cards)}</div>'


def _review_item_html(item: dict[str, Any]) -> str:
    """Render one amenity row as inert HTML plus data attributes."""
    item_id = str(item["id"])
    status = str(item.get("status", "pending"))
    name = str(item.get("name", ""))
    present = bool(item.get("present", True))
    confidence = item.get("confidence")

    if status == "editing":
        editable_name = html.escape(name.removeprefix(WARNING_PREFIX), quote=True)
        checked = " checked" if present else ""
        return (
            '<div class="amenity-row amenity-edit-row">'
            f'<input class="review-edit-input" id="review-name-{html.escape(item_id)}" '
            f'value="{editable_name}" placeholder="Amenity name">'
            '<label class="review-present-check">'
            f'<input type="checkbox" id="review-present-{html.escape(item_id)}"{checked}> Present'
            "</label>"
            f"{_review_action_button('save', item_id, '✓ Save', 'primary')}"
            f"{_review_action_button('cancel', item_id, '✗ Cancel')}"
            "</div>"
        )

    if status == "confirmed":
        display_name = html.escape(name.removeprefix(WARNING_PREFIX) or "(empty)")
        icon = "✓" if present else "✗"
        tone = "present" if present else "absent"
        return (
            '<div class="amenity-row amenity-confirmed-row">'
            '<div class="amenity-copy">'
            f'<span class="amenity-chip amenity-chip-{tone}">{icon} {display_name} '
            f"<em>({tone})</em></span>"
            "</div>"
            f"{_review_action_button('edit', item_id, 'Undo')}"
            "</div>"
        )

    is_warning = name.startswith(WARNING_PREFIX)
    warning_class = " amenity-warning" if is_warning else ""
    badge = "⚠ " if is_warning else ""
    display_name = html.escape(name.removeprefix(WARNING_PREFIX))
    conf_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "—"
    present_str = "Present" if present else "Not present"
    return (
        '<div class="amenity-row amenity-pending-row">'
        '<div class="amenity-copy">'
        f'<div class="amenity-line{warning_class}">'
        f'<span class="amenity-name">{badge}{display_name}</span>'
        f'<span class="amenity-meta">{present_str} · confidence {html.escape(conf_str)}</span>'
        "</div>"
        "</div>"
        f"{_review_action_button('confirm', item_id, '✓', 'primary')}"
        f"{_review_action_button('edit', item_id, '✎')}"
        f"{_review_action_button('reject', item_id, '✗')}"
        "</div>"
    )


def _review_action_button(action: str, item_id: str, label: str, variant: str = "") -> str:
    payload = html.escape(json.dumps({"action": action, "id": item_id}))
    variant_class = " primary" if variant == "primary" else ""
    return (
        f'<button type="button" class="amenity-action review-action-btn{variant_class}" '
        f'data-review-payload="{payload}">{html.escape(label)}</button>'
    )


def apply_review_action(
    payload_json: str, state: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Apply one HTML review-button action to the Gradio review state."""
    state = state or []
    try:
        payload = json.loads(payload_json or "{}")
    except json.JSONDecodeError:
        logger.warning("Ignoring malformed review action payload: %r", payload_json)
        return state

    action = str(payload.get("action", ""))
    item_id = str(payload.get("id", ""))

    if action == "add":
        return add_item(state, str(payload.get("room", "unknown") or "unknown"))
    if not item_id:
        return state
    if action == "confirm":
        return confirm_item(state, item_id)
    if action == "edit":
        return start_edit(state, item_id)
    if action == "reject":
        return reject_item(state, item_id)
    if action == "save":
        return save_edit(
            state,
            item_id,
            str(payload.get("name", "")),
            bool(payload.get("present", True)),
        )
    if action == "cancel":
        return cancel_edit(state, item_id)

    logger.warning("Ignoring unknown review action: %r", action)
    return state


def _render_item_row(item: dict[str, Any], state_component: gr.State) -> None:
    """Render one amenity row based on its ``status``."""
    item_id = item["id"]
    status = item["status"]
    name = str(item.get("name", ""))
    present = bool(item.get("present", True))
    confidence = item.get("confidence")

    if status == "confirmed":
        _render_confirmed_chip(item_id, name, present, state_component)
        return

    if status == "editing":
        _render_editing_row(item_id, name, present, state_component)
        return

    _render_pending_row(item_id, name, present, confidence, state_component)


def _render_confirmed_chip(
    item_id: str,
    name: str,
    present: bool,
    state_component: gr.State,
) -> None:
    """Compact green chip for user-verified amenities, with an undo affordance."""
    display_name = name.removeprefix(WARNING_PREFIX) or "(empty)"
    icon = "✓" if present else "✗"
    tone = "present" if present else "absent"
    with gr.Row(elem_classes=["amenity-row", "amenity-confirmed-row"], key=f"{item_id}-confirmed"):
        gr.Markdown(
            f'<span class="amenity-chip amenity-chip-{tone}">'
            f"{icon} {display_name} <em>({tone})</em>"
            "</span>",
            elem_classes=["amenity-copy"],
        )
        undo_btn = gr.Button(
            "Undo",
            size="sm",
            scale=0,
            elem_classes=["amenity-action"],
            key=f"{item_id}-undo",
        )
        undo_btn.click(
            fn=lambda current, i=item_id: start_edit(current or [], i),
            inputs=[state_component],
            outputs=[state_component],
            queue=False,
        )


def _render_editing_row(
    item_id: str,
    name: str,
    present: bool,
    state_component: gr.State,
) -> None:
    """Inline textbox + save/cancel for a row being edited or freshly added."""
    # Strip any warning prefix when populating the edit textbox: the user is
    # about to change the name anyway, and the prefix should never end up
    # committed to the amenity's actual name.
    editable_name = name.removeprefix(WARNING_PREFIX)
    with gr.Row(elem_classes=["amenity-row", "amenity-edit-row"], key=f"{item_id}-editing"):
        name_tb = gr.Textbox(
            value=editable_name,
            placeholder="Amenity name (e.g. dishwasher)",
            show_label=False,
            scale=3,
            elem_classes=["amenity-edit-name"],
            key=f"{item_id}-name",
        )
        present_cb = gr.Checkbox(
            value=present,
            label="Present",
            scale=1,
            elem_classes=["amenity-present-check"],
            key=f"{item_id}-present",
        )
        save_btn = gr.Button(
            "✓ Save",
            variant="primary",
            size="sm",
            scale=0,
            elem_classes=["amenity-action"],
            key=f"{item_id}-save",
        )
        cancel_btn = gr.Button(
            "✗ Cancel",
            size="sm",
            scale=0,
            elem_classes=["amenity-action"],
            key=f"{item_id}-cancel",
        )

        save_btn.click(
            fn=lambda current, n, p, i=item_id: save_edit(current or [], i, n, bool(p)),
            inputs=[state_component, name_tb, present_cb],
            outputs=[state_component],
            queue=False,
        )
        cancel_btn.click(
            fn=lambda current, i=item_id: cancel_edit(current or [], i),
            inputs=[state_component],
            outputs=[state_component],
            queue=False,
        )


def _render_pending_row(
    item_id: str,
    name: str,
    present: bool,
    confidence: float | None,
    state_component: gr.State,
) -> None:
    """Default row: show name + confidence + ✓ / ✎ / ✗ buttons."""
    is_warning = name.startswith(WARNING_PREFIX)
    display_name = name.removeprefix(WARNING_PREFIX)
    warning_class = " amenity-warning" if is_warning else ""
    badge = "⚠ " if is_warning else ""
    conf_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "—"
    present_str = "Present" if present else "Not present"

    with gr.Row(elem_classes=["amenity-row", "amenity-pending-row"], key=f"{item_id}-pending"):
        gr.Markdown(
            f'<div class="amenity-line{warning_class}">'
            f'<span class="amenity-name">{badge}{display_name}</span>'
            f'<span class="amenity-meta">{present_str} · confidence {conf_str}</span>'
            "</div>",
            elem_classes=["amenity-copy"],
        )
        confirm_btn = gr.Button(
            "✓",
            size="sm",
            scale=0,
            variant="primary",
            elem_classes=["amenity-action"],
            key=f"{item_id}-confirm",
        )
        edit_btn = gr.Button(
            "✎", size="sm", scale=0, elem_classes=["amenity-action"], key=f"{item_id}-edit"
        )
        reject_btn = gr.Button(
            "✗", size="sm", scale=0, elem_classes=["amenity-action"], key=f"{item_id}-reject"
        )

        confirm_btn.click(
            fn=lambda current, i=item_id: confirm_item(current or [], i),
            inputs=[state_component],
            outputs=[state_component],
            queue=False,
        )
        edit_btn.click(
            fn=lambda current, i=item_id: start_edit(current or [], i),
            inputs=[state_component],
            outputs=[state_component],
            queue=False,
        )
        reject_btn.click(
            fn=lambda current, i=item_id: reject_item(current or [], i),
            inputs=[state_component],
            outputs=[state_component],
            queue=False,
        )


# ── Build the Gradio app ───────────────────────────────────────────────────────


def _build_theme() -> Any:
    """
    Build the amber/stone theme used when launching the Gradio app.
    """
    return gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.stone,
    ).set(
        # ── Light mode ──
        body_background_fill="#fafaf7",
        background_fill_primary="#fafaf7",
        background_fill_secondary="#f0ede4",
        block_background_fill="#fafaf7",
        block_border_color="#e5e0d8",
        input_background_fill="#fafaf7",
        border_color_primary="#e5e0d8",
        body_text_color="#1c1917",
        block_label_text_color="#57534e",
        button_primary_background_fill="#b45309",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_text_color="#b45309",
        button_secondary_border_color="#b45309",
        # ── Dark mode (stone background, brighter amber for contrast) ──
        body_background_fill_dark="#1c1917",
        background_fill_primary_dark="#1c1917",
        background_fill_secondary_dark="#292524",
        block_background_fill_dark="#292524",
        block_border_color_dark="#44403c",
        input_background_fill_dark="#292524",
        border_color_primary_dark="#44403c",
        body_text_color_dark="#fafaf7",
        block_label_text_color_dark="#d6d3d1",
        button_primary_background_fill_dark="#f59e0b",
        button_primary_text_color_dark="#1c1917",
        button_secondary_background_fill_dark="#292524",
        button_secondary_text_color_dark="#f59e0b",
        button_secondary_border_color_dark="#f59e0b",
    )


def build_app() -> gr.Blocks:
    """Construct and return the three-page Gradio Blocks application."""
    available_models = _get_available_models()
    default_model = available_models[0] if available_models else "openai/gpt-4o-mini"

    with gr.Blocks(title="Amenity Detector") as demo:
        page_state = gr.State(HOME)
        upload_state = gr.State({})
        review_state = gr.State([])
        gr.HTML(bg_orbs_html())

        with gr.Group(visible=True) as home_group:
            with gr.Column(elem_classes=["app-shell"]):
                gr.HTML(
                    """
                    <div class="top-bar">
                      <button class="login-btn" disabled>Sign in</button>
                      <button type="button" class="theme-toggle" aria-label="Toggle theme"></button>
                    </div>
                    <section class="ad-hero">
                      <div>
                        <div class="ad-badge">AI-powered property analysis</div>
                        <h1 class="ad-title">Amenity Detector</h1>
                        <p class="ad-sub"><span id="ad-typewriter"></span><span class="cursor"></span></p>
                        <div class="ad-chips">
                          <span class="ad-chip">Room classification</span>
                          <span class="ad-chip">Editable detections</span>
                          <span class="ad-chip">Browse saved properties</span>
                        </div>
                      </div>
                    </section>
                    """
                )
                with gr.Row(elem_classes=["cta-row"]):
                    home_to_upload_btn = gr.Button("Upload & Detect", variant="primary", scale=1)
                    home_to_browse_btn = gr.Button(
                        "Browse Properties", variant="secondary", scale=1
                    )

        with gr.Group(visible=False) as upload_group:
            with gr.Column(elem_classes=["app-shell"]):
                with gr.Row(elem_classes=["top-bar"]):
                    upload_back_btn = gr.Button(
                        "Back", variant="secondary", elem_classes=["back-chip"]
                    )
                    gr.HTML(
                        '<button type="button" class="theme-toggle" aria-label="Toggle theme"></button>'
                    )
                gr.HTML(
                    '<div class="section-title"><h2>Upload & Detect</h2><span class="dim">Step-by-step property analysis</span></div>'
                )
                step_indicator = gr.HTML(_step_html(0))

                # ── Step 1: Config ──────────────────────────────────────────
                with gr.Group(
                    visible=True, elem_classes=["step-panel"], elem_id="step-config"
                ) as step1_config:
                    gr.Markdown("### Step 1 — Configuration")
                    gr.Markdown("Tell us about the property and how you'd like it analysed.")
                    property_name_input = gr.Textbox(
                        label="Property Name", placeholder="e.g. Frankfurt Apartment, 2BR"
                    )
                    model_dropdown = gr.Dropdown(
                        choices=available_models, value=default_model, label="AI Model"
                    )
                    extra_info_input = gr.Textbox(
                        label="Additional Notes", lines=2, elem_id="additional-notes"
                    )

                    # ── Phase 9: Listing details (optional, collapsed by default) ─
                    # Every field is optional. Filled values are passed through to
                    # POST /api/v1/properties/ alongside the existing name +
                    # model_name + extra_info. Empty fields stay NULL on the row.
                    with gr.Accordion("Listing details (optional)", open=False):
                        with gr.Row():
                            listing_type_input = gr.Dropdown(
                                choices=["", "rent", "sale"],
                                value="",
                                label="Listing type",
                            )
                            property_type_input = gr.Dropdown(
                                choices=["", "apartment", "house", "villa", "studio", "other"],
                                value="",
                                label="Property type",
                            )
                            furnishing_input = gr.Dropdown(
                                choices=["", "furnished", "semi_furnished", "unfurnished"],
                                value="",
                                label="Furnishing",
                            )
                        with gr.Row():
                            price_input = gr.Number(value=None, label="Price", precision=2)
                            currency_input = gr.Dropdown(
                                choices=["", "EUR", "USD", "GBP", "INR", "JPY", "CHF"],
                                value="",
                                label="Currency",
                            )
                            price_period_input = gr.Dropdown(
                                choices=["", "monthly", "weekly", "nightly", "total"],
                                value="",
                                label="Price period",
                            )
                        with gr.Row():
                            num_bedrooms_input = gr.Number(
                                value=None, label="Bedrooms (BHK)", precision=0
                            )
                            num_bathrooms_input = gr.Number(
                                value=None, label="Bathrooms", precision=0
                            )
                            area_sqm_input = gr.Number(value=None, label="Area (m²)", precision=2)
                        with gr.Row():
                            available_from_input = gr.Textbox(
                                label="Available from (YYYY-MM-DD)", placeholder="2026-06-01"
                            )
                            owner_email_input = gr.Textbox(
                                label="Owner email", placeholder="owner@example.com"
                            )
                        with gr.Row():
                            locality_input = gr.Textbox(
                                label="Locality / neighbourhood",
                                placeholder="e.g. Sachsenhausen",
                            )
                            postal_code_input = gr.Textbox(
                                label="Postal / ZIP code", placeholder="60594"
                            )
                            country_code_input = gr.Textbox(
                                label="Country code (ISO 2)", placeholder="DE"
                            )

                    listing_metadata_inputs: list[Any] = [
                        listing_type_input,
                        price_input,
                        currency_input,
                        price_period_input,
                        property_type_input,
                        furnishing_input,
                        num_bedrooms_input,
                        num_bathrooms_input,
                        area_sqm_input,
                        available_from_input,
                        locality_input,
                        postal_code_input,
                        country_code_input,
                        owner_email_input,
                    ]

                    num_rooms_input = gr.Slider(
                        minimum=0, maximum=6, step=1, value=0, label="Number of rooms"
                    )
                    hint_textboxes: list[Any] = []
                    gr.HTML(hints_grid_html())
                    for _group, hints in AMENITY_GROUPS:
                        for key, _label in hints:
                            tb = gr.Textbox(
                                value="Not specified",
                                elem_id=f"{key}-val",
                                elem_classes=["hint-hidden"],
                                show_label=False,
                            )
                            hint_textboxes.append(tb)
                    with gr.Row(elem_classes=["step-actions", "right-only"]):
                        step1_next_btn = gr.Button("Next: Upload images →", variant="primary")

                # ── Step 2: Upload ──────────────────────────────────────────
                with gr.Group(
                    visible=False, elem_classes=["step-panel"], elem_id="step-upload"
                ) as step2_upload:
                    gr.Markdown("### Step 2 — Upload property images")
                    gr.Markdown("Drag and drop one or more images, or click to browse.")
                    image_files = gr.Files(
                        label="Property Images", file_types=["image"], file_count="multiple"
                    )
                    with gr.Row(elem_classes=["step-actions"]):
                        step2_back_btn = gr.Button("← Back", variant="secondary")
                        step2_detect_btn = gr.Button("Start detection →", variant="primary")

                # ── Step 3: Detect (processing animation) ───────────────────
                with gr.Group(
                    visible=False, elem_classes=["step-panel"], elem_id="step-detect"
                ) as step3_detect:
                    gr.HTML(
                        """
                        <div class="processing-wrap">
                          <div class="processing-icon">⚡</div>
                          <div class="processing-title">Analysing your property</div>
                          <div class="processing-sub">This may take a few seconds per image.</div>
                          <div class="progress-shimmer-outer"><div class="progress-shimmer-inner"></div></div>
                          <div class="proc-steps">
                            <div class="proc-step done"><span class="proc-dot"></span>Uploading images</div>
                            <div class="proc-step active"><span class="proc-dot"></span>Classifying rooms</div>
                            <div class="proc-step"><span class="proc-dot"></span>Detecting amenities</div>
                            <div class="proc-step"><span class="proc-dot"></span>Generating descriptions</div>
                          </div>
                        </div>
                        """
                    )

                # ── Step 4: Review ──────────────────────────────────────────
                with gr.Group(
                    visible=False,
                    elem_classes=["step-panel"],
                    elem_id="step-review",
                ) as step4_review:
                    gr.Markdown("### Step 4 — Review & describe")
                    upload_status = gr.Textbox(label="Status", interactive=False)
                    gr.Markdown("#### Detected Amenities")
                    with gr.Column(elem_id="review-cards-grid"):
                        review_html = gr.HTML(value=_review_panel_html([]))

                    review_action_payload = gr.Textbox(
                        value="",
                        show_label=False,
                        elem_id="review-action-payload",
                        elem_classes=["hint-hidden"],
                    )
                    review_action_apply = gr.Button(
                        "Apply Review Action",
                        elem_id="review-action-apply",
                        elem_classes=["hint-hidden"],
                    )
                    confirm_btn = gr.Button("Confirm & Generate Description", variant="primary")
                    description_output = gr.Textbox(
                        label="Generated Description",
                        lines=5,
                        interactive=True,
                        placeholder="Description will appear here after confirmation.",
                    )
                    with gr.Row(elem_classes=["step-actions"]):
                        step4_back_btn = gr.Button("← Back to upload", variant="secondary")

        with gr.Group(visible=False) as browse_group:
            with gr.Column(elem_classes=["app-shell"]):
                with gr.Row(elem_classes=["top-bar"]):
                    browse_back_btn = gr.Button(
                        "Back", variant="secondary", elem_classes=["back-chip"]
                    )
                    gr.HTML(
                        '<button type="button" class="theme-toggle" aria-label="Toggle theme"></button>'
                    )
                gr.HTML(
                    '<div class="section-title"><h2>Browse Properties</h2><span class="dim">Search saved detections</span></div>'
                )
                with gr.Row():
                    amenity_search_input = gr.Textbox(
                        label="Search by amenity", placeholder="e.g. refrigerator, sofa", scale=4
                    )
                    search_btn = gr.Button("Search", variant="primary", scale=1)
                    list_all_btn = gr.Button("List All", scale=1)
                property_cards = gr.HTML(list_property_cards())
                gr.Markdown("### Property Details")
                property_id_input = gr.Textbox(label="Property ID")
                view_btn = gr.Button("View Details", variant="secondary")
                property_detail_output = gr.Markdown("Enter a property ID above to view details.")

        def _nav(page: str):
            home, upload, browse, state = go_to(page)
            return (
                gr.update(**cast(Any, home)),
                gr.update(**cast(Any, upload)),
                gr.update(**cast(Any, browse)),
                state,
            )

        # Layout for ``_upload_with_hints``:
        #   values[0..7]   image_files, name, model, extra_info,
        #                  num_rooms, kitchen_hint, balcony_hint, living_room_hint
        #   values[8..21]  14 listing-metadata inputs (Phase 9)
        #   values[22..]   expanded amenity hint textboxes
        _LISTING_INPUT_OFFSET = 8
        _LISTING_INPUT_COUNT = 14

        def _upload_with_hints(*values: Any):
            listing_values = values[
                _LISTING_INPUT_OFFSET : _LISTING_INPUT_OFFSET + _LISTING_INPUT_COUNT
            ]
            listing_metadata = _listing_metadata_payload(*listing_values)
            hints = _expanded_hints_from_sequence(
                cast(
                    tuple[str | None, ...],
                    tuple(values[_LISTING_INPUT_OFFSET + _LISTING_INPUT_COUNT :]),
                )
            )
            for step_html, status, state, description, next_upload_state in upload_and_detect(
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                values[6],
                values[7],
                hints,
                listing_metadata,
            ):
                if state and str(status).startswith("Detected "):
                    active = 3
                elif str(status).startswith(
                    (
                        "Please ",
                        "Upload failed",
                        "Request timed out",
                        "Cannot reach",
                        "Unexpected error",
                    )
                ):
                    active = 3
                else:
                    active = 2
                yield (
                    *[gr.update(visible=(i == active)) for i in range(4)],
                    step_html,
                    status,
                    state,
                    description,
                    next_upload_state,
                )

        def _confirm_with_hints(*values: Any) -> tuple[str, str, str]:
            hints = _expanded_hints_from_sequence(cast(tuple[str | None, ...], tuple(values[6:])))
            return confirm_and_describe(
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                hints,
            )

        def _reconcile_with_hints(*values: Any) -> list[dict[str, Any]]:
            hints = _expanded_hints_from_sequence(cast(tuple[str | None, ...], tuple(values[5:])))
            return reconcile_state_from_hints(
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                hints,
            )

        home_to_upload_btn.click(
            lambda: _nav(UPLOAD), outputs=[home_group, upload_group, browse_group, page_state]
        )
        home_to_browse_btn.click(
            lambda: _nav(BROWSE), outputs=[home_group, upload_group, browse_group, page_state]
        )
        upload_back_btn.click(
            lambda: _nav(HOME), outputs=[home_group, upload_group, browse_group, page_state]
        )
        browse_back_btn.click(
            lambda: _nav(HOME), outputs=[home_group, upload_group, browse_group, page_state]
        )

        legacy_hint_inputs: list[Any] = [
            num_rooms_input,
            hint_textboxes[HINT_KEYS.index("kitchen")],
            hint_textboxes[HINT_KEYS.index("balcony")],
            hint_textboxes[HINT_KEYS.index("living_room")],
        ]

        # ── Step navigation: visibility + step indicator ──
        step_panels = [step1_config, step2_upload, step3_detect, step4_review]

        def _show_step(active: int) -> list[Any]:
            """Return Gradio updates that show only the requested step group."""
            return [gr.update(visible=(i == active)) for i in range(4)] + [_step_html(active)]

        step1_next_btn.click(
            fn=lambda: _show_step(1),
            outputs=[*step_panels, step_indicator],
            queue=False,
        )
        step2_back_btn.click(
            fn=lambda: _show_step(0),
            outputs=[*step_panels, step_indicator],
            queue=False,
        )
        step4_back_btn.click(
            fn=lambda: _show_step(1),
            outputs=[*step_panels, step_indicator],
            queue=False,
        )

        # Detect: jump to step 3 (animation), run detection, then jump to step 4.
        step2_detect_btn.click(
            fn=lambda: _show_step(2),
            outputs=[*step_panels, step_indicator],
            queue=False,
        ).then(
            fn=_upload_with_hints,
            inputs=[
                image_files,
                property_name_input,
                model_dropdown,
                extra_info_input,
                *legacy_hint_inputs,
                *listing_metadata_inputs,
                *hint_textboxes,
            ],
            outputs=[
                *step_panels,
                step_indicator,
                upload_status,
                review_state,
                description_output,
                upload_state,
            ],
        ).then(
            fn=_review_panel_html,
            inputs=[review_state],
            outputs=[review_html],
            queue=False,
        )

        confirm_btn.click(
            fn=_confirm_with_hints,
            inputs=[review_state, upload_state, *legacy_hint_inputs, *hint_textboxes],
            outputs=[step_indicator, upload_status, description_output],
        )
        review_action_apply.click(
            fn=apply_review_action,
            inputs=[review_action_payload, review_state],
            outputs=[review_state],
            queue=False,
        ).then(
            fn=_review_panel_html,
            inputs=[review_state],
            outputs=[review_html],
            queue=False,
        )
        for hint in [num_rooms_input, *hint_textboxes]:
            hint.change(
                fn=_reconcile_with_hints,
                inputs=[*legacy_hint_inputs, review_state, *hint_textboxes],
                outputs=[review_state],
            ).then(
                fn=_review_panel_html,
                inputs=[review_state],
                outputs=[review_html],
                queue=False,
            )
        search_btn.click(
            fn=search_property_cards, inputs=[amenity_search_input], outputs=[property_cards]
        )
        list_all_btn.click(fn=list_property_cards, inputs=[], outputs=[property_cards])
        view_btn.click(
            fn=get_property_detail, inputs=[property_id_input], outputs=[property_detail_output]
        )

    return cast(gr.Blocks, demo)


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        theme=_build_theme(),
        css=CSS,
        head=HEAD,
    )
