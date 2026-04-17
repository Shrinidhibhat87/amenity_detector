"""
Gradio frontend for the Amenity Detector — Phase 3.

Architecture overview:
  This file contains the entire Gradio UI. It talks to the FastAPI backend over HTTP
  so the UI and the backend are completely decoupled:
    - The UI does NOT import any FastAPI, SQLAlchemy, or VLM code directly.
    - It communicates via the REST API (api/routers/properties.py, api/routers/models.py).
    - The API base URL is configured via the API_BASE_URL environment variable so it
      works both locally (http://localhost:8000) and inside Docker Compose (http://api:8000).

Why Gradio?
  Gradio is purpose-built for ML demos. It gives us multi-file image upload, a clean
  component model, and a Python-only stack — no JavaScript needed.
  The upload and browse tabs cover the two core user journeys.

How to run locally (without Docker):
    uv run python -m ui.app
    # or:
    API_BASE_URL=http://localhost:8000 uv run gradio ui/app.py

How to run in Docker Compose (Phase 3):
    docker compose up ui
    # UI is then reachable at http://localhost:7860

Tabs:
  1. Upload Property  — Upload images, choose a VLM, run detection, see results.
  2. Browse Properties — Search by amenity name, view a property details.
"""

import os
from typing import Any

import gradio as gr
import requests

# ── Configuration ─────────────────────────────────────────────────────────────
# The API base URL comes from an environment variable so it works in both local
# development and Docker Compose without code changes.
# Docker Compose sets API_BASE_URL=http://api:8000 in the ui service environment.
_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

# Request timeout for API calls in seconds.
# VLM inference (especially local Ollama models) can be slow on low-end hardware,
# so we use a generous timeout. Adjust if your model is particularly slow.
_TIMEOUT_SECONDS = 300


# ── Helper functions ─────────────────────────────────────────────────────────


def _get_available_models() -> list[str]:
    """
    Fetch available model names from the FastAPI backend.

    Called at UI startup to populate the model dropdown. Falls back to a
    hardcoded list if the backend is unreachable (e.g., starting order issues
    in Docker Compose).

    Returns:
        List of model name strings (e.g. ["gemini-2.0-flash", "qwen2.5vl:7b"]).
    """
    try:
        response = requests.get(f"{_API_BASE_URL}/api/v1/models/", timeout=10)
        response.raise_for_status()
        models_data: list[dict[str, Any]] = response.json()
        # Return only models that are actually available (registered at startup)
        return [m["name"] for m in models_data if m.get("available")]
    except Exception:
        # Fallback — matches SUPPORTED_MODELS in models/registry.py
        return ["gemini-2.0-flash", "qwen2.5vl:7b", "llama3.2-vision:11b"]


def _format_amenities_table(
    images: list[dict[str, Any]],
) -> list[list[Any]]:
    """
    Flatten detected amenities from the upload response into a table format.

    Each row represents one amenity detection result for one image.

    Args:
        images: The ``images`` list from the PropertyDetailResponse JSON.

    Returns:
        List of rows, where each row is:
            [filename, room_type, amenity_name, present (yes/no), confidence]
    """
    rows: list[list[Any]] = []
    for img in images:
        filename = img.get("file_path", "").split("/")[-1]
        room = img.get("room_type") or "unknown"
        for amenity in img.get("amenities", []):
            rows.append(
                [
                    filename,
                    room,
                    amenity["amenity_name"],
                    "yes" if amenity["is_present"] else "no",
                    f"{amenity['confidence']:.2f}"
                    if amenity.get("confidence") is not None
                    else "—",
                ]
            )
    return rows


# ── Upload tab handler ────────────────────────────────────────────────────────


def upload_and_detect(
    files: list[Any] | None,
    property_name: str,
    model_name: str,
    extra_info: str,
) -> tuple[str, list[list[Any]], str]:
    """
    Handle the Upload tab form submission.

    Sends the uploaded images and metadata to POST /api/v1/properties/upload,
    then formats the response for display.

    Args:
        files:          List of file objects from gr.Files (or None if none uploaded).
        property_name:  User-provided property label.
        model_name:     Selected VLM from the dropdown.
        extra_info:     Optional free-text notes about the property.

    Returns:
        Tuple of three values to update three Gradio output components:
          - status_text:   Short success or error message.
          - amenity_rows:  Table rows for the amenity results dataframe.
          - description:   VLM-generated property description.
    """
    # ── Input validation ──
    if not files:
        return "Please upload at least one image.", [], ""
    if not property_name.strip():
        return "Please enter a property name.", [], ""
    if not model_name:
        return "Please select a model from the dropdown.", [], ""

    # ── Build the multipart request ──
    # Gradio returns file objects (or file paths as strings in some modes).
    # We open each one and send it as a multipart file upload.
    try:
        opened_files: list[tuple[str, Any, str]] = []
        for f in files:
            # Gradio's gr.Files widget can return either a file path (str) or a
            # NamedTemporaryFile-like object depending on the Gradio version.
            if isinstance(f, str):
                file_path = f
                filename = os.path.basename(f)
            else:
                file_path = f.name if hasattr(f, "name") else str(f)
                filename = os.path.basename(file_path)
            with open(file_path, "rb") as fh:
                content = fh.read()
            # Infer MIME type from extension
            ext = filename.rsplit(".", 1)[-1].lower()
            mime = {
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "png": "image/png",
                "webp": "image/webp",
            }.get(ext, "image/jpeg")
            opened_files.append((filename, content, mime))

        # Construct the multipart files list for requests
        multipart_files = [("files", (name, content, mime)) for name, content, mime in opened_files]

        form_data: dict[str, str] = {
            "name": property_name.strip(),
            "model_name": model_name,
        }
        if extra_info.strip():
            form_data["extra_info"] = extra_info.strip()

        response = requests.post(
            f"{_API_BASE_URL}/api/v1/properties/upload",
            data=form_data,
            files=multipart_files,
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

    except requests.exceptions.Timeout:
        return (
            "Request timed out. The VLM is taking too long — try a faster model or fewer images.",
            [],
            "",
        )
    except requests.exceptions.ConnectionError:
        return (
            f"Cannot reach the API at {_API_BASE_URL}. Is the backend running?",
            [],
            "",
        )
    except requests.exceptions.HTTPError as e:
        # Extract the FastAPI error detail message if available
        detail = ""
        try:
            detail = response.json().get("detail", "")
        except Exception:
            pass
        return f"Upload failed ({e}): {detail}", [], ""
    except Exception as e:
        return f"Unexpected error: {e}", [], ""

    # ── Format response ──
    data: dict[str, Any] = response.json()
    prop: dict[str, Any] = data.get("property", {})

    status = f"Uploaded successfully. Property ID: {data.get('property_id', '?')}"
    description: str = prop.get("description") or "No description generated."
    amenity_rows = _format_amenities_table(prop.get("images", []))

    return status, amenity_rows, description


# ── Browse tab handlers ───────────────────────────────────────────────────────


def search_properties(amenity_query: str) -> list[list[Any]]:
    """
    Handle the Browse tab search form submission.

    Sends the amenity query to GET /api/v1/properties/search and formats the
    results for display in the results dataframe.

    Args:
        amenity_query: Comma-separated amenity names entered by the user.

    Returns:
        List of table rows: [property_id, name, description, model_used, image_count]
        Empty list on error or no results.
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

    rows: list[list[Any]] = []
    for prop in response.json():
        rows.append(
            [
                prop.get("id", ""),
                prop.get("name", ""),
                (prop.get("description") or "")[:120],  # Truncate for table display
                prop.get("model_used", ""),
                prop.get("image_count", 0),
            ]
        )
    return rows


def list_all_properties() -> list[list[Any]]:
    """
    Fetch all properties for the Browse tab default view.

    Returns:
        List of table rows: [property_id, name, description, model_used, image_count]
    """
    try:
        response = requests.get(
            f"{_API_BASE_URL}/api/v1/properties/",
            params={"limit": 50},
            timeout=30,
        )
        response.raise_for_status()
    except Exception:
        return []

    rows: list[list[Any]] = []
    for prop in response.json():
        rows.append(
            [
                prop.get("id", ""),
                prop.get("name", ""),
                (prop.get("description") or "")[:120],
                prop.get("model_used", ""),
                prop.get("image_count", 0),
            ]
        )
    return rows


def get_property_detail(property_id: str) -> str:
    """
    Fetch and format full details for a single property.

    Called when the user enters a property ID in the detail viewer.

    Args:
        property_id: UUID of the property to look up.

    Returns:
        Markdown-formatted string with property details, images, and amenities.
    """
    if not property_id.strip():
        return "Enter a property ID above to view details."

    try:
        response = requests.get(
            f"{_API_BASE_URL}/api/v1/properties/{property_id.strip()}",
            timeout=30,
        )
        if response.status_code == 404:
            return f"Property '{property_id}' not found."
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return f"Cannot reach the API at {_API_BASE_URL}."
    except Exception as e:
        return f"Error fetching property: {e}"

    prop: dict[str, Any] = response.json()

    # Build Markdown output
    lines: list[str] = [
        f"## {prop.get('name', 'Unknown')}",
        f"**ID:** `{prop.get('id', '')}`",
        f"**Model:** {prop.get('model_used', '—')}",
        f"**Created:** {prop.get('created_at', '—')}",
    ]

    if prop.get("extra_info"):
        lines.append(f"**Notes:** {prop['extra_info']}")

    description = prop.get("description") or ""
    if description:
        lines.append(f"\n**Description:**\n{description}")

    lines.append("\n---\n### Detected Amenities by Image")
    for img in prop.get("images", []):
        filename = img.get("file_path", "").split("/")[-1]
        room = img.get("room_type") or "unknown"
        lines.append(f"\n**{filename}** (room: {room})")

        present = [
            f"- {a['amenity_name']} (confidence: {a['confidence']:.2f})"
            for a in img.get("amenities", [])
            if a.get("is_present")
        ]
        absent = [a["amenity_name"] for a in img.get("amenities", []) if not a.get("is_present")]

        if present:
            lines.append(
                "Present: "
                + ", ".join(
                    a["amenity_name"] for a in img.get("amenities", []) if a.get("is_present")
                )
            )
        if absent:
            lines.append("Not detected: " + ", ".join(absent))

    return "\n".join(lines)


# ── Build the Gradio app ──────────────────────────────────────────────────────


def build_app() -> "gr.Blocks":  # type: ignore[type-arg,misc]
    """
    Construct and return the Gradio Blocks application.

    Using gr.Blocks instead of gr.Interface gives us full layout control:
    we can have two tabs, custom column widths, and shared state.

    Returns:
        A configured gr.Blocks instance ready to be launched.
    """
    available_models = _get_available_models()
    default_model = available_models[0] if available_models else "gemini-2.0-flash"

    with gr.Blocks(title="Amenity Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Amenity Detector
            Upload property images to automatically detect amenities, or browse
            previously uploaded properties.

            **Backend:** `{}` | **Available models:** {}
            """.format(
                _API_BASE_URL,
                ", ".join(available_models) if available_models else "none (backend offline?)",
            )
        )

        with gr.Tab("Upload Property"):
            gr.Markdown(
                "Upload one or more room images. The selected VLM will identify visible amenities "
                "and generate a natural-language description."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    # File upload — accepts multiple images
                    image_files = gr.Files(
                        label="Property Images",
                        file_types=["image"],
                        file_count="multiple",
                    )
                    property_name_input = gr.Textbox(
                        label="Property Name",
                        placeholder="e.g. Frankfurt House 1",
                    )
                    model_dropdown = gr.Dropdown(
                        choices=available_models,
                        value=default_model,
                        label="Select VLM",
                        info="Ollama models need a running local Ollama server. "
                        "Gemini requires GEMINI_API_KEY.",
                    )
                    extra_info_input = gr.Textbox(
                        label="Additional Info (optional)",
                        placeholder="e.g. 2-bed flat in Frankfurt, has central heating",
                        lines=2,
                    )
                    upload_btn = gr.Button("Upload & Detect Amenities", variant="primary")

                with gr.Column(scale=2):
                    upload_status = gr.Textbox(label="Status", interactive=False)
                    description_output = gr.Textbox(
                        label="Generated Description",
                        lines=4,
                        interactive=False,
                    )
                    amenities_table = gr.Dataframe(
                        headers=["Image", "Room", "Amenity", "Present", "Confidence"],
                        label="Detected Amenities",
                        interactive=False,
                        wrap=True,
                    )

            upload_btn.click(
                fn=upload_and_detect,
                inputs=[image_files, property_name_input, model_dropdown, extra_info_input],
                outputs=[upload_status, amenities_table, description_output],
            )

        with gr.Tab("Browse Properties"):
            gr.Markdown(
                "Search previously uploaded properties by amenity, or view all. "
                "Click a row to see full details."
            )

            with gr.Row():
                amenity_search_input = gr.Textbox(
                    label="Search by amenity (comma-separated)",
                    placeholder="e.g. refrigerator, oven",
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
                label="Property ID (paste from table above)",
                placeholder="00000000-0000-0000-0000-000000000000",
            )
            view_btn = gr.Button("View Details")
            property_detail_output = gr.Markdown(value="Enter a property ID above to view details.")

            search_btn.click(
                fn=search_properties,
                inputs=[amenity_search_input],
                outputs=[results_table],
            )
            list_all_btn.click(
                fn=list_all_properties,
                inputs=[],
                outputs=[results_table],
            )
            view_btn.click(
                fn=get_property_detail,
                inputs=[property_id_input],
                outputs=[property_detail_output],
            )

    return demo  # type: ignore[return-value, no-any-return]


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Launch settings:
    #   server_name="0.0.0.0"  — bind to all interfaces so Docker port mapping works
    #   server_port=7860       — matches docker-compose.yml and Dockerfile.ui
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
    )
