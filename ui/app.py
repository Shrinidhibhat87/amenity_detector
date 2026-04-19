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

import logging
import os
from typing import Any, cast

# Keep ``build_app()`` smoke tests deterministic and avoid Gradio background
# telemetry threads in CI/local test runs.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import requests

from core.logging_config import setup_logging
from ui.helpers import WARNING_PREFIX, reconcile_amenities
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
        List of model name strings, e.g. ["gemini-2.0-flash", "qwen2.5vl:7b"].
    """
    try:
        response = requests.get(f"{_API_BASE_URL}/api/v1/models/", timeout=10)
        response.raise_for_status()
        models_data: list[dict[str, Any]] = response.json()
        return [m["name"] for m in models_data if m.get("available")]
    except Exception:
        return ["gemini-2.0-flash", "qwen2.5vl:7b", "llama3.2-vision:11b"]


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
        image: The ``image`` object from SingleImageUploadResponse.

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
) -> dict[str, Any]:
    """
    Build the sidebar hint payload used by reconciliation and /describe.
    """
    parsed_rooms = int(num_rooms or 0)
    return {
        "num_rooms": parsed_rooms if parsed_rooms > 0 else None,
        "has_kitchen": _hint_to_bool(kitchen_hint),
        "has_balcony": _hint_to_bool(balcony_hint),
        "has_living_room": _hint_to_bool(living_room_hint),
    }


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
    """
    Build the 4-step progress indicator HTML for the Upload tab.

    Args:
        active: Index of the currently active step (0-based).
                0=Upload, 1=Detect, 2=Review, 3=Generate.

    Returns:
        HTML string to inject into a gr.HTML component.
    """
    # All colours come from the --ad-* CSS variables defined in _CSS, so the
    # indicator switches automatically with the rest of the theme. Hardcoded
    # hex would silently fail in dark mode (dark text on dark background).
    labels = ["1. Upload", "2. Detect", "3. Review & Edit", "4. Generate Description"]
    parts: list[str] = [
        "<div style='display:flex;align-items:center;gap:0;margin:0 0 20px;flex-wrap:wrap;gap:4px;'>"
    ]
    for i, label in enumerate(labels):
        if i < active:
            # Completed — solid amber dot with checkmark
            dot_style = "background:var(--ad-amber);color:var(--button-primary-text-color,#fff);"
            text_style = "color:var(--ad-amber);"
            symbol = "✓"
        elif i == active:
            # Current — amber outline on a faint amber wash
            dot_style = (
                "background:var(--ad-yellow);color:var(--ad-amber);"
                "border:2px solid var(--ad-amber);"
            )
            text_style = "color:var(--ad-text);font-weight:700;"
            symbol = str(i + 1)
        else:
            # Future — muted, sits quietly in the background
            dot_style = "background:var(--ad-bg2);color:var(--ad-text-light);"
            text_style = "color:var(--ad-text-light);"
            symbol = str(i + 1)

        parts.append(
            f"<div style='display:flex;align-items:center;gap:6px;'>"
            f"<div style='width:26px;height:26px;border-radius:50%;display:flex;align-items:center;"
            f"justify-content:center;font-size:11px;font-weight:800;flex-shrink:0;{dot_style}'>{symbol}</div>"
            f"<span style='font-size:12px;font-weight:600;{text_style}'>{label}</span>"
            f"</div>"
        )
        if i < len(labels) - 1:
            line_color = "var(--ad-amber)" if i < active else "var(--ad-border)"
            parts.append(
                f"<div style='flex:1;min-width:20px;max-width:40px;height:2px;background:{line_color};'></div>"
            )
    parts.append("</div>")
    return "".join(parts)


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
            _step_html(0),
            "Please upload at least one image.",
            empty_review,
            "",
            empty_upload_state,
        )
        return
    if not property_name.strip():
        yield _step_html(0), "Please enter a property name.", empty_review, "", empty_upload_state
        return
    if not model_name:
        yield _step_html(0), "Please select a model.", empty_review, "", empty_upload_state
        return

    sidebar = _sidebar_payload(num_rooms, kitchen_hint, balcony_hint, living_room_hint)
    seen_images: list[dict[str, Any]] = []
    property_id = ""
    try:
        response = requests.post(
            f"{_API_BASE_URL}/api/v1/properties/",
            json={
                "name": property_name.strip(),
                "model_name": model_name,
                "extra_info": extra_info.strip() or None,
            },
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
            yield _step_html(1), status, review_state, "", upload_state

        progress(1.0, desc="Detection complete")
        review_state = _reconciled_review_state(seen_images, sidebar)
        total_items = sum(len(block["items"]) for block in review_state)
        status = (
            f"Detected {total_items} amenities for '{property_name}'. "
            "Review per-room below, then click Confirm."
        )
        yield _step_html(2), status, review_state, "", upload_state

    except requests.exceptions.Timeout:
        logger.warning(
            "Upload timed out after %ds for property '%s'", _TIMEOUT_SECONDS, property_name
        )
        yield (
            _step_html(0),
            "Request timed out. Try a faster model or fewer images.",
            empty_review,
            "",
            empty_upload_state,
        )
    except requests.exceptions.ConnectionError:
        logger.warning("Cannot reach API at %s during upload", _API_BASE_URL)
        yield (
            _step_html(0),
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
        yield _step_html(0), f"Upload failed: {detail or exc}", empty_review, "", empty_upload_state
    except Exception as exc:
        # logger.exception includes the full traceback so that the real root
        # cause is visible in `docker compose logs ui` instead of just a
        # one-line "Unexpected error".
        logger.exception("Unexpected error in upload_and_detect for property '%s'", property_name)
        yield _step_html(0), f"Unexpected error: {exc}", empty_review, "", empty_upload_state


def confirm_and_describe(
    review_state: list[dict[str, Any]],
    upload_state: dict[str, str],
    num_rooms: int | float | None,
    kitchen_hint: str | None,
    balcony_hint: str | None,
    living_room_hint: str | None,
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
    model_name = upload_state.get("model_name", "gemini-2.0-flash")

    if not property_id:
        return _step_html(2), "Upload images first before generating a description.", ""

    amenities = amenities_for_describe(review_state or [])
    sidebar = _sidebar_payload(num_rooms, kitchen_hint, balcony_hint, living_room_hint)

    try:
        response = requests.post(
            f"{_API_BASE_URL}/api/v1/properties/{property_id}/describe",
            json={"amenities": amenities, "model_name": model_name, **sidebar},
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.warning("Describe call timed out for property %s", property_id)
        return _step_html(2), "Description generation timed out. Try again.", ""
    except requests.exceptions.ConnectionError:
        logger.warning("Cannot reach API at %s during describe", _API_BASE_URL)
        return _step_html(2), f"Cannot reach API at {_API_BASE_URL}.", ""
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
        return _step_html(2), f"Description failed: {detail or exc}", ""
    except Exception as exc:
        logger.exception("Unexpected error in confirm_and_describe for property %s", property_id)
        return _step_html(2), f"Unexpected error: {exc}", ""

    description: str = response.json().get("description", "")
    return _step_html(3), "Description generated. You can edit it below before saving.", description


def reconcile_state_from_hints(
    num_rooms: int | float | None,
    kitchen_hint: str | None,
    balcony_hint: str | None,
    living_room_hint: str | None,
    review_state: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """
    Re-run mismatch markers across the review state when sidebar hints change.

    Flattens non-editing items to the row shape reconcile expects, runs the
    warning-prefix logic, and writes the updated names back into state.
    """
    sidebar = _sidebar_payload(num_rooms, kitchen_hint, balcony_hint, living_room_hint)
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
    if not state:
        gr.Markdown(
            "_No amenities yet. Upload images on the left to start detection._",
            elem_classes="ad-review-empty",
        )
        return

    for block in state:
        room = block["room"]
        items = block["items"]
        with gr.Group():
            gr.Markdown(f"#### 🏠 {_humanise_room(room)}")
            for item in items:
                _render_item_row(item, state_component)

            add_btn = gr.Button(
                "+ Add amenity",
                variant="secondary",
                size="sm",
            )
            add_btn.click(
                fn=lambda current, r=room: add_item(current or [], r),
                inputs=[state_component],
                outputs=[state_component],
            )


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
    with gr.Row():
        gr.Markdown(
            f"<span style='background:var(--ad-amber-bg);color:var(--ad-amber);"
            f"padding:4px 10px;border-radius:100px;font-size:13px;font-weight:600;'>"
            f"{icon} {display_name} <em style='font-weight:400;opacity:0.7'>({tone})</em>"
            f"</span>"
        )
        undo_btn = gr.Button("Undo", size="sm", scale=0)
        undo_btn.click(
            fn=lambda current, i=item_id: start_edit(current or [], i),
            inputs=[state_component],
            outputs=[state_component],
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
    with gr.Row():
        name_tb = gr.Textbox(
            value=editable_name,
            placeholder="Amenity name (e.g. dishwasher)",
            show_label=False,
            scale=3,
        )
        present_cb = gr.Checkbox(value=present, label="Present", scale=1)
        save_btn = gr.Button("✓ Save", variant="primary", size="sm", scale=0)
        cancel_btn = gr.Button("✗ Cancel", size="sm", scale=0)

        save_btn.click(
            fn=lambda current, n, p, i=item_id: save_edit(current or [], i, n, bool(p)),
            inputs=[state_component, name_tb, present_cb],
            outputs=[state_component],
        )
        cancel_btn.click(
            fn=lambda current, i=item_id: cancel_edit(current or [], i),
            inputs=[state_component],
            outputs=[state_component],
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
    badge = "⚠ " if is_warning else ""
    colour = "var(--ad-amber)" if is_warning else "var(--ad-text)"
    conf_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "—"
    present_str = "Present" if present else "Not present"

    with gr.Row():
        gr.Markdown(
            f"<span style='color:{colour};font-weight:600'>{badge}{display_name}</span> "
            f"<span style='color:var(--ad-text-light);font-size:12px;margin-left:8px'>"
            f"{present_str} · confidence {conf_str}</span>"
        )
        confirm_btn = gr.Button("✓", size="sm", scale=0, variant="primary")
        edit_btn = gr.Button("✎", size="sm", scale=0)
        reject_btn = gr.Button("✗", size="sm", scale=0)

        confirm_btn.click(
            fn=lambda current, i=item_id: confirm_item(current or [], i),
            inputs=[state_component],
            outputs=[state_component],
        )
        edit_btn.click(
            fn=lambda current, i=item_id: start_edit(current or [], i),
            inputs=[state_component],
            outputs=[state_component],
        )
        reject_btn.click(
            fn=lambda current, i=item_id: reject_item(current or [], i),
            inputs=[state_component],
            outputs=[state_component],
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
    """
    Construct and return the full Gradio Blocks application.

    Structure:
      1. gr.HTML — hero landing page (animations, CTA buttons)
      2. gr.Tabs
         ├── Tab "Upload & Detect"
         └── Tab "Browse Properties"

    Returns:
        Configured gr.Blocks instance ready to launch.
    """
    available_models = _get_available_models()
    default_model = available_models[0] if available_models else "gemini-2.0-flash"

    # In Gradio 6 the `theme=`, `css=` and `js=` parameters belong on launch(),
    # not on gr.Blocks() — passing them here raises a UserWarning. The hero
    # typewriter script is embedded directly inside the gr.HTML block below
    # (see _HERO_HTML) so it runs as soon as the hero markup is inserted into
    # the DOM, regardless of how Gradio handles `launch(js=...)`.
    with gr.Blocks(title="Amenity Detector") as demo:
        # ── Hero landing page ──────────────────────────────────────────────────
        gr.HTML(_HERO_HTML)

        # ── Tabs ──────────────────────────────────────────────────────────────
        with gr.Tabs():
            # ── Upload & Detect tab ────────────────────────────────────────────
            with gr.Tab("↑ Upload & Detect"):
                # gr.State stores {property_id, model_name} between steps.
                # review_state holds the per-room amenity data consumed by the
                # @gr.render panel below (see ui/review_state.py for shape).
                upload_state = gr.State({})
                review_state = gr.State([])

                step_indicator = gr.HTML(_step_html(0))

                with gr.Row():
                    # Left column: form
                    with gr.Column(scale=1, min_width=280):
                        image_files = gr.Files(
                            label="Property Images",
                            file_types=["image"],
                            file_count="multiple",
                        )
                        property_name_input = gr.Textbox(
                            label="Property Name",
                            placeholder="e.g. Frankfurt Apartment · 2BR",
                        )
                        model_dropdown = gr.Dropdown(
                            choices=available_models,
                            value=default_model,
                            label="AI Model",
                            info="Ollama models need a running local Ollama server. Gemini needs GEMINI_API_KEY.",
                        )
                        extra_info_input = gr.Textbox(
                            label="Additional Notes (optional)",
                            placeholder="e.g. 2-bed flat in Frankfurt, central heating",
                            lines=2,
                        )
                        gr.Markdown("### Property Hints")
                        num_rooms_input = gr.Slider(
                            minimum=0,
                            maximum=6,
                            step=1,
                            value=0,
                            label="Number of rooms",
                            info="Leave at 0 if you do not want to specify.",
                        )
                        kitchen_hint_input = gr.Radio(
                            choices=["Not specified", "Yes", "No"],
                            value="Not specified",
                            label="Kitchen",
                        )
                        balcony_hint_input = gr.Radio(
                            choices=["Not specified", "Yes", "No"],
                            value="Not specified",
                            label="Balcony",
                        )
                        living_room_hint_input = gr.Radio(
                            choices=["Not specified", "Yes", "No"],
                            value="Not specified",
                            label="Living room",
                        )
                        upload_btn = gr.Button("↑ Upload & Detect Amenities", variant="primary")

                    # Right column: results
                    with gr.Column(scale=2):
                        upload_status = gr.Textbox(label="Status", interactive=False)

                        gr.Markdown("### Detected Amenities — Review per-room")

                        # The @gr.render-decorated function below redraws the
                        # review panel on every change to ``review_state``.
                        # Buttons declared inside re-register their click
                        # handlers each re-render — Gradio handles the diffing.
                        @gr.render(inputs=[review_state])
                        def render_review_panel(state: list[dict[str, Any]]) -> None:
                            _render_review_panel(state, review_state)

                        confirm_btn = gr.Button(
                            "✔ Confirm & Generate Description", variant="primary"
                        )

                        description_output = gr.Textbox(
                            label="Generated Description (editable)",
                            lines=5,
                            interactive=True,  # user can manually edit the text
                            placeholder="Description will appear here after you click Confirm…",
                        )

                # Wire up events
                upload_btn.click(
                    fn=upload_and_detect,
                    inputs=[
                        image_files,
                        property_name_input,
                        model_dropdown,
                        extra_info_input,
                        num_rooms_input,
                        kitchen_hint_input,
                        balcony_hint_input,
                        living_room_hint_input,
                    ],
                    outputs=[
                        step_indicator,
                        upload_status,
                        review_state,
                        description_output,
                        upload_state,
                    ],
                )

                confirm_btn.click(
                    fn=confirm_and_describe,
                    inputs=[
                        review_state,
                        upload_state,
                        num_rooms_input,
                        kitchen_hint_input,
                        balcony_hint_input,
                        living_room_hint_input,
                    ],
                    outputs=[step_indicator, upload_status, description_output],
                )

                hint_inputs: list[Any] = [
                    num_rooms_input,
                    kitchen_hint_input,
                    balcony_hint_input,
                    living_room_hint_input,
                ]
                for hint in hint_inputs:
                    hint.change(
                        fn=reconcile_state_from_hints,
                        inputs=[*hint_inputs, review_state],
                        outputs=[review_state],
                    )

            # ── Browse Properties tab ──────────────────────────────────────────
            with gr.Tab("🔍 Browse Properties"):
                gr.Markdown(
                    "Search stored properties by amenity name, or list all. "
                    "Paste a property ID below to view full details."
                )

                with gr.Row():
                    amenity_search_input = gr.Textbox(
                        label="Search by amenity (comma-separated)",
                        placeholder="e.g. refrigerator, sofa",
                        scale=4,
                    )
                    search_btn = gr.Button("Search", variant="primary", scale=1)
                    list_all_btn = gr.Button("List All", scale=1)

                results_table = gr.Dataframe(
                    headers=["ID", "Name", "Description (preview)", "Model", "Images"],
                    label="Properties",
                    interactive=False,
                    wrap=True,
                )

                gr.Markdown("### Property Details")
                property_id_input = gr.Textbox(
                    label="Property ID",
                    placeholder="Paste an ID from the table above",
                )
                view_btn = gr.Button("View Details", variant="secondary")
                property_detail_output = gr.Markdown("Enter a property ID above to view details.")

                search_btn.click(
                    fn=search_properties, inputs=[amenity_search_input], outputs=[results_table]
                )
                list_all_btn.click(fn=list_all_properties, inputs=[], outputs=[results_table])
                view_btn.click(
                    fn=get_property_detail,
                    inputs=[property_id_input],
                    outputs=[property_detail_output],
                )

    return cast(gr.Blocks, demo)


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        theme=_build_theme(),
        css=_CSS,
    )
