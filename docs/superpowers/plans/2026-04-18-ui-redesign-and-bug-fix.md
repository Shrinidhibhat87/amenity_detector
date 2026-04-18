# UI Redesign + Generation Bug Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Ollama/Gemini generation failures and replace the plain Gradio UI with a polished, warm-toned app (landing page + multi-step Upload flow + Browse page).

**Architecture:** The bug is a configuration issue — `.env.example` ships `OLLAMA_BASE_URL=http://localhost:11434`, which inside Docker resolves to the container itself, not the host. Fix: change the default to `host.docker.internal:11434` and add `extra_hosts` for Linux. The UI rewrite stays entirely in `ui/app.py`; a minimal new API endpoint (`POST /api/v1/properties/{id}/describe`) enables description regeneration after the user edits the amenity table.

**Tech Stack:** Gradio 4.x (`gr.Blocks`, `gr.HTML`, `gr.State`, `gr.Dataframe`), FastAPI, Pydantic, pytest, Docker Compose.

---

## File Map

| File | Change |
|---|---|
| `.env.example` | Fix `OLLAMA_BASE_URL` default from `localhost` → `host.docker.internal:11434`; add `OLLAMA_HOST` guidance |
| `docker-compose.yml` | Add `extra_hosts: ["host.docker.internal:host-gateway"]` to `api` service (Linux/WSL2 native Docker compatibility) |
| `api/schemas.py` | Add `DescribeRequest` + `DescribeResponse` schemas |
| `api/routers/properties.py` | Add `POST /api/v1/properties/{id}/describe` endpoint |
| `core/amenity_system.py` | Add `generate_description_from_amenities()` method |
| `ui/app.py` | Complete rewrite — landing page, redesigned Upload tab, redesigned Browse tab |
| `tests/unit/test_describe_endpoint.py` | New — tests for the describe endpoint |
| `tests/unit/test_ui_helpers.py` | New — tests for UI helper functions |
| `Readme.md` | Add testing steps (T3), Gemini quota note, Ollama host setup |

---

## Task 1: Fix Ollama Networking

**Files:**
- Modify: `.env.example`
- Modify: `docker-compose.yml`

**What and why:**
The `.env.example` file ships with `OLLAMA_BASE_URL=http://localhost:11434`. When a user copies it to `.env`, Docker Compose reads that value and injects it into the API container as-is. Inside the container, `localhost` means the container itself — not the host machine where `ollama serve` runs. We fix this by:
1. Changing the `.env.example` default to `http://host.docker.internal:11434` (works on Docker Desktop + WSL2).
2. Adding `extra_hosts: ["host.docker.internal:host-gateway"]` to docker-compose.yml so the hostname resolves even on native Linux Docker Engine (without Docker Desktop).

Additionally, Ollama must be told to listen on all interfaces (not just `127.0.0.1`) so Docker containers can reach it. The user starts Ollama with `OLLAMA_HOST=0.0.0.0 ollama serve` instead of just `ollama serve`.

- [ ] **Step 1: Update `.env.example`**

Replace the OLLAMA section in `.env.example`:

```bash
# ── VLM: Ollama (Phase 1+) ────────────────────────────────────────────────────
# URL of the Ollama server as seen FROM INSIDE the Docker container.
# On Docker Desktop (Windows / Mac / WSL2): host.docker.internal resolves to the host.
# On native Linux Docker Engine: the extra_hosts in docker-compose.yml handles this.
# Do NOT change this to http://localhost:11434 — localhost inside Docker means the container.
OLLAMA_BASE_URL=http://host.docker.internal:11434

# Ollama listen address — tells the Ollama server which interface to bind to.
# When running Docker, Ollama must listen on all interfaces (0.0.0.0), not just localhost.
# Start Ollama with: OLLAMA_HOST=0.0.0.0 ollama serve
# (not needed if you are only running ui/app.py locally without Docker)
OLLAMA_HOST=0.0.0.0
```

- [ ] **Step 2: Add `extra_hosts` to docker-compose.yml api service**

Inside the `api:` service block, add `extra_hosts` at the same indent level as `environment:` and `volumes:`:

```yaml
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    restart: unless-stopped
    ports:
      - "8000:8000"
    # extra_hosts makes host.docker.internal resolve on native Linux Docker Engine.
    # On Docker Desktop this entry is harmless — Docker Desktop adds it automatically.
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER:-amenity_user}:${POSTGRES_PASSWORD:-amenity_pass}@db:5432/${POSTGRES_DB:-amenity_db}
      OLLAMA_BASE_URL: ${OLLAMA_BASE_URL:-http://host.docker.internal:11434}
      GEMINI_API_KEY: ${GEMINI_API_KEY:-}
      IMAGE_STORAGE_DIR: /app/storage/images
    volumes:
      - image_storage:/app/storage/images
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8000/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5
```

- [ ] **Step 3: Update your local `.env` file**

Open `.env` (not `.env.example`) and change the Ollama line:
```
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

- [ ] **Step 4: Restart Ollama with the correct host binding**

In the Ollama terminal (WSL), stop the current `ollama serve` and restart with:
```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

- [ ] **Step 5: Rebuild and verify**

```bash
docker compose down && docker compose up --build -d
# Wait ~30s for health checks, then tail the api logs:
docker compose logs -f api
```

Expected in logs — **no** `Cannot connect to Ollama at http://localhost:11434` error. If you upload an image with `qwen2.5vl:7b`, you should see inference activity in the Ollama terminal.

- [ ] **Step 6: Commit**

```bash
git add .env.example docker-compose.yml
git commit -m "fix: correct OLLAMA_BASE_URL to host.docker.internal for Docker networking

localhost inside a Docker container resolves to the container itself, not the
host. Switching to host.docker.internal (+ extra_hosts for Linux) fixes the
'Cannot connect to Ollama' error seen in docker compose logs api."
```

---

## Task 2: Add `POST /api/v1/properties/{id}/describe` Endpoint

**Files:**
- Modify: `api/schemas.py`
- Modify: `core/amenity_system.py`
- Modify: `api/routers/properties.py`

**What and why:**
The current upload flow detects amenities AND generates a description in one shot. After the UI redesign, users will be able to edit the amenity table and then request a fresh description based on their edits. This endpoint takes a list of `{amenity_name, room_type, is_present}` items and asks the VLM to write a description based only on the amenities marked as present.

- [ ] **Step 1: Add schemas to `api/schemas.py`**

Append to the end of `api/schemas.py`:

```python
# ── Describe endpoint schemas ─────────────────────────────────────────────────


class AmenityEditItem(BaseModel):
    """One amenity entry as edited by the user in the UI."""

    amenity_name: str
    room_type: str
    is_present: bool


class DescribeRequest(BaseModel):
    """
    Request body for POST /api/v1/properties/{id}/describe.

    The UI sends the user's edited amenity table so the VLM can regenerate
    the description based only on the amenities the user confirmed as present.
    """

    amenities: list[AmenityEditItem]
    model_name: str


class DescribeResponse(BaseModel):
    """Response from the describe endpoint."""

    description: str
```

- [ ] **Step 2: Add `generate_description_from_amenities()` to `core/amenity_system.py`**

Add this method to the `PropertyAmenitySystem` class, just after `process_upload`:

```python
    def generate_description_from_amenities(
        self,
        amenities: list[dict[str, object]],
        property_name: str,
        extra_info: str | None = None,
    ) -> str:
        """
        Generate a property description from a user-edited amenity list.

        Called when the user has reviewed the detected amenities, made edits,
        and clicks 'Confirm & Generate Description'. Only amenities where
        is_present=True are included in the prompt.

        Args:
            amenities:     List of dicts with keys: amenity_name, room_type, is_present.
            property_name: Used to personalise the description.
            extra_info:    Optional context (e.g. "2-bed flat in Frankfurt").

        Returns:
            A natural-language description string from the VLM.
        """
        # Group confirmed amenities by room
        by_room: dict[str, list[str]] = {}
        for item in amenities:
            if item.get("is_present"):
                room = str(item.get("room_type", "unknown"))
                name = str(item.get("amenity_name", ""))
                by_room.setdefault(room, []).append(name)

        if not by_room:
            return "No amenities were confirmed as present."

        # Build a structured prompt so the VLM has clear input
        room_lines = "\n".join(
            f"  - {room.title()}: {', '.join(items)}" for room, items in by_room.items()
        )
        context = f" Additional context: {extra_info}." if extra_info else ""
        prompt = (
            f"You are writing a property listing description for '{property_name}'.{context}\n"
            f"The following amenities have been confirmed as present:\n{room_lines}\n\n"
            "Write a warm, professional 3-4 sentence description of the property that "
            "highlights these amenities. Do not invent amenities not listed above."
        )

        # Use a blank 1x1 white image as a placeholder — this endpoint uses text-only context.
        # Most VLMs accept an image; we pass a minimal one to keep the interface consistent.
        from PIL import Image as PILImage

        placeholder = PILImage.new("RGB", (1, 1), color=(255, 255, 255))

        try:
            response = self.detector.vlm_client.generate(image=placeholder, prompt=prompt)
            return response.raw_text.strip()
        except Exception as e:
            self.logger.error("Description generation failed: %s", e)
            raise RuntimeError(f"Description generation failed: {e}") from e
```

- [ ] **Step 3: Add the endpoint to `api/routers/properties.py`**

Add these imports at the top of the file (after the existing imports):

```python
from api.schemas import (
    AmenityEditItem,           # add this
    DescribeRequest,           # add this
    DescribeResponse,          # add this
    DetectedAmenityResponse,
    PropertyDetailResponse,
    PropertyImageResponse,
    PropertySummaryResponse,
    UploadResponse,
)
```

Then add this route to `api/routers/properties.py` **before** the `GET /{id}` route:

```python
@router.post("/{property_id}/describe", response_model=DescribeResponse)
def regenerate_description(
    property_id: str,
    body: DescribeRequest,
    db: Session = Depends(get_db),
    registry: ModelRegistry = Depends(get_model_registry),
    storage_dir: Path = Depends(get_image_storage_dir),
) -> DescribeResponse:
    """
    Regenerate a property description from the user's edited amenity list.

    Called after the user reviews and edits the detected amenities in the UI.
    The VLM is asked to write a fresh description based only on the amenities
    the user confirmed as present.

    Args:
        property_id: UUID of the existing property (must exist in the DB).
        body:        Edited amenity list + the model to use for generation.

    Returns:
        DescribeResponse with the new description text.

    Raises:
        400: If the model is unknown.
        404: If the property does not exist.
        500: If VLM inference fails.
    """
    # Verify the property exists
    from db.models import Property as PropertyModel

    prop = db.get(PropertyModel, property_id)
    if prop is None:
        raise HTTPException(status_code=404, detail=f"Property '{property_id}' not found.")

    try:
        vlm_client = registry.get(body.model_name)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        system = PropertyAmenitySystem(
            vlm_client=vlm_client,
            db=db,
            image_storage_dir=storage_dir,
        )
        amenities_as_dicts = [
            {"amenity_name": a.amenity_name, "room_type": a.room_type, "is_present": a.is_present}
            for a in body.amenities
        ]
        description = system.generate_description_from_amenities(
            amenities=amenities_as_dicts,
            property_name=prop.name,
            extra_info=prop.extra_info,
        )
    except Exception as e:
        logger.exception("Description regeneration failed for property '%s'", property_id)
        raise HTTPException(status_code=500, detail=f"Description generation failed: {e}") from e

    return DescribeResponse(description=description)
```

- [ ] **Step 4: Commit**

```bash
git add api/schemas.py core/amenity_system.py api/routers/properties.py
git commit -m "feat: add POST /api/v1/properties/{id}/describe endpoint

Allows the UI to regenerate a property description after the user edits
the detected amenity table. Takes the confirmed amenities and calls the
VLM with a structured text prompt (no image re-upload needed)."
```

---

## Task 3: Write Tests for the New Endpoint

**Files:**
- Create: `tests/unit/test_describe_endpoint.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_describe_endpoint.py`:

```python
"""
Tests for POST /api/v1/properties/{id}/describe.

We mock the VLM and DB so these run fast without external services.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from models.base import VLMResponse


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_describe_returns_description(client: TestClient) -> None:
    """Happy path: valid property + valid model → returns description."""
    mock_prop = MagicMock()
    mock_prop.name = "Test House"
    mock_prop.extra_info = None

    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = VLMResponse(
        raw_text="A lovely property with a sofa and a fridge.",
        model_name="gemini-2.0-flash",
    )

    with (
        patch("api.routers.properties.PropertyAmenitySystem") as mock_system_cls,
        patch("db.session.get_db") as mock_get_db,
    ):
        mock_db = MagicMock()
        mock_db.get.return_value = mock_prop
        mock_get_db.return_value = iter([mock_db])

        mock_system = MagicMock()
        mock_system.generate_description_from_amenities.return_value = (
            "A lovely property with a sofa and a fridge."
        )
        mock_system_cls.return_value = mock_system

        with patch("api.dependencies.get_model_registry") as mock_registry_dep:
            mock_registry = MagicMock()
            mock_registry.get.return_value = mock_vlm
            mock_registry_dep.return_value = mock_registry

            response = client.post(
                "/api/v1/properties/some-uuid/describe",
                json={
                    "amenities": [
                        {"amenity_name": "Sofa", "room_type": "living_room", "is_present": True},
                        {"amenity_name": "Dishwasher", "room_type": "kitchen", "is_present": False},
                    ],
                    "model_name": "gemini-2.0-flash",
                },
            )

    assert response.status_code == 200
    assert "description" in response.json()
    assert len(response.json()["description"]) > 0


def test_describe_returns_404_for_unknown_property(client: TestClient) -> None:
    """If property_id does not exist in DB, endpoint returns 404."""
    with patch("db.session.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_db.get.return_value = None  # property not found
        mock_get_db.return_value = iter([mock_db])

        with patch("api.dependencies.get_model_registry"):
            response = client.post(
                "/api/v1/properties/nonexistent-uuid/describe",
                json={
                    "amenities": [],
                    "model_name": "gemini-2.0-flash",
                },
            )

    assert response.status_code == 404


def test_generate_description_from_amenities_filters_absent() -> None:
    """generate_description_from_amenities must only include is_present=True amenities."""
    from unittest.mock import MagicMock
    from core.amenity_system import PropertyAmenitySystem

    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = VLMResponse(
        raw_text="A property with a sofa.", model_name="test"
    )

    system = PropertyAmenitySystem.__new__(PropertyAmenitySystem)
    system.logger = MagicMock()
    system.detector = MagicMock()
    system.detector.vlm_client = mock_vlm

    amenities = [
        {"amenity_name": "Sofa", "room_type": "living_room", "is_present": True},
        {"amenity_name": "Air Conditioning", "room_type": "living_room", "is_present": False},
    ]

    result = system.generate_description_from_amenities(amenities, "Test House")

    # The prompt sent to the VLM should mention Sofa but NOT Air Conditioning
    call_args = mock_vlm.generate.call_args
    prompt_used: str = call_args.kwargs.get("prompt") or call_args.args[1]
    assert "Sofa" in prompt_used
    assert "Air Conditioning" not in prompt_used
    assert isinstance(result, str)


def test_generate_description_empty_amenities_returns_fallback() -> None:
    """If no amenities are present, return a meaningful fallback string."""
    from core.amenity_system import PropertyAmenitySystem

    system = PropertyAmenitySystem.__new__(PropertyAmenitySystem)
    system.logger = MagicMock()
    system.detector = MagicMock()

    result = system.generate_description_from_amenities(
        amenities=[{"amenity_name": "Sofa", "room_type": "lr", "is_present": False}],
        property_name="Empty House",
    )

    assert result == "No amenities were confirmed as present."
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/unit/test_describe_endpoint.py -v
```

Expected: Tests fail because the endpoint and method don't exist yet (you do Task 2 to make them pass).

> **Note:** Complete Task 2 first, then return here to run the tests.

- [ ] **Step 3: Run tests after Task 2 is complete**

```bash
uv run pytest tests/unit/test_describe_endpoint.py -v
```

Expected output: 4 tests PASSED.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_describe_endpoint.py
git commit -m "test: add tests for describe endpoint and amenity filtering"
```

---

## Task 4: Rewrite `ui/app.py`

**Files:**
- Modify: `ui/app.py` (complete rewrite)
- Create: `tests/unit/test_ui_helpers.py`

**What and why:** The current UI goes straight to functional Gradio tabs with default styling. We replace it with: (1) a warm-toned hero landing page as a `gr.HTML` block, (2) a redesigned Upload tab with a step indicator, editable amenity DataFrame, and a "Confirm & Generate" button that calls the new describe endpoint, (3) a redesigned Browse tab with styled property cards. All changes are inside `ui/app.py` — no backend changes needed beyond Task 2.

**Key Gradio patterns used:**
- `gr.HTML` — for the landing page hero and the step indicator (raw HTML in Gradio)
- `gr.State` — stores intermediate data (property_id, model_name) between the upload and confirm steps
- `gr.Dataframe(interactive=True, datatype=["str","str","bool","str"])` — the amenity table; `bool` columns render as checkboxes the user can toggle
- `gr.Tabs` / `gr.Tab` — the two functional tabs
- `gr.update()` — used inside event handlers to show/hide components dynamically

- [ ] **Step 1: Write tests for UI helper functions first**

Create `tests/unit/test_ui_helpers.py`:

```python
"""
Tests for ui/app.py helper functions.

These are pure-Python functions that transform API response data into
Gradio display formats. No Gradio runtime needed — just plain unit tests.
"""

from unittest.mock import MagicMock, patch


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
        _make_image("lr.jpg", "living_room", [{"amenity_name": "sofa", "is_present": True, "confidence": 0.9}]),
        _make_image("bed.jpg", "bedroom", [{"amenity_name": "bed", "is_present": True, "confidence": 0.88}]),
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
    import requests
    from unittest.mock import patch

    with patch("ui.app.requests.get", side_effect=requests.exceptions.ConnectionError):
        from ui.app import _get_available_models
        models = _get_available_models()

    assert isinstance(models, list)
    assert len(models) > 0
    assert "gemini-2.0-flash" in models


def test_get_available_models_filters_unavailable() -> None:
    """Only models with available=True are returned."""
    from unittest.mock import MagicMock, patch

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
```

- [ ] **Step 2: Run tests — they should fail (ui/app.py not rewritten yet)**

```bash
uv run pytest tests/unit/test_ui_helpers.py -v
```

Expected: ImportError or assertion failures — the helper functions have different signatures. That's fine — we're doing TDD.

- [ ] **Step 3: Rewrite `ui/app.py`**

Replace the entire contents of `ui/app.py` with:

```python
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

import os
from typing import Any

import gradio as gr
import requests

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


# ── CSS / theme constants ──────────────────────────────────────────────────────
# Warm neutral palette: cream background, amber accent, stone text.
_AMBER = "#b45309"
_CREAM_BG = "#fafaf7"
_CSS = f"""
/* ── Global warm-neutral palette ── */
.gradio-container {{
    background: linear-gradient(160deg, #fafaf7 0%, #f0ede4 100%) !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}}

/* Primary buttons → amber */
.btn-primary, button.primary {{
    background: {_AMBER} !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 9px !important;
    box-shadow: 0 3px 12px rgba(180,83,9,0.25) !important;
}}
.btn-primary:hover, button.primary:hover {{
    background: #92400e !important;
    transform: translateY(-1px) !important;
}}

/* Secondary buttons → white + amber border */
button.secondary {{
    background: white !important;
    color: {_AMBER} !important;
    border: 2px solid {_AMBER} !important;
    font-weight: 700 !important;
    border-radius: 9px !important;
}}

/* Labels and headings */
label span, .label-wrap span {{
    color: #57534e !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}}

/* Input borders */
input, textarea, select {{
    border: 1.5px solid #e5e0d8 !important;
    border-radius: 8px !important;
    background: {_CREAM_BG} !important;
}}
input:focus, textarea:focus {{
    border-color: {_AMBER} !important;
}}

/* Tab bar */
.tab-nav button {{
    font-weight: 600 !important;
    color: #78716c !important;
}}
.tab-nav button.selected {{
    color: {_AMBER} !important;
    border-bottom: 2px solid {_AMBER} !important;
}}
"""

# ── Landing page HTML ──────────────────────────────────────────────────────────
# Injected as gr.HTML at the top of the app. JavaScript targets Gradio tab
# buttons by their aria-selected attribute to navigate between tabs.
_HERO_HTML = """
<style>
.ad-hero {
    background: linear-gradient(160deg, #fafaf7 0%, #f0ede4 100%);
    padding: 60px 24px 48px;
    text-align: center;
    border-bottom: 1px solid #e5e0d8;
}
.ad-badge {
    display: inline-block;
    background: rgba(180,83,9,0.1);
    color: #b45309;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 22px;
    animation: ad-slidein 0.5s ease forwards;
}
.ad-title {
    font-size: clamp(28px, 4vw, 48px);
    font-weight: 800;
    color: #1c1917;
    margin-bottom: 18px;
    animation: ad-slidein 0.6s ease 0.15s both;
}
.ad-sub {
    font-size: 16px;
    color: #78716c;
    min-height: 26px;
    margin-bottom: 36px;
    animation: ad-fadein 0.5s ease 0.9s both;
}
.ad-cursor {
    display: inline-block;
    width: 2px;
    height: 1em;
    background: #b45309;
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: ad-blink 0.8s step-end infinite;
}
.ad-cta { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; animation: ad-slidein 0.5s ease 1s both; }
.ad-btn-primary {
    background: #b45309; color: white; border: none;
    padding: 13px 28px; border-radius: 9px; font-size: 14px; font-weight: 700;
    cursor: pointer; box-shadow: 0 4px 14px rgba(180,83,9,0.28);
    transition: transform 0.15s, box-shadow 0.15s;
}
.ad-btn-primary:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(180,83,9,0.36); }
.ad-btn-secondary {
    background: white; color: #b45309; border: 2px solid #b45309;
    padding: 11px 26px; border-radius: 9px; font-size: 14px; font-weight: 700;
    cursor: pointer; transition: transform 0.15s, background 0.15s;
}
.ad-btn-secondary:hover { background: #fef3c7; transform: translateY(-2px); }
.ad-chips {
    display: flex; gap: 8px; flex-wrap: wrap; justify-content: center;
    margin-top: 40px; animation: ad-fadein 0.5s ease 1.3s both;
}
.ad-chip {
    background: white; border: 1px solid #e5e0d8; color: #57534e;
    font-size: 12px; font-weight: 500; padding: 5px 13px; border-radius: 100px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
@keyframes ad-slidein { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
@keyframes ad-fadein  { from { opacity:0; } to { opacity:1; } }
@keyframes ad-blink   { 50% { opacity:0; } }
</style>

<div class="ad-hero">
  <div class="ad-badge">✦ AI-Powered Property Analysis</div>
  <div class="ad-title">Smart Property Amenity Detection</div>
  <div class="ad-sub"><span id="ad-typed"></span><span class="ad-cursor"></span></div>

  <div class="ad-cta">
    <button class="ad-btn-primary" onclick="adGoTab(0)">↑ Upload &amp; Detect</button>
    <button class="ad-btn-secondary" onclick="adGoTab(1)">🔍 Browse Properties</button>
  </div>

  <div class="ad-chips">
    <span class="ad-chip">🏠 Room Classification</span>
    <span class="ad-chip">🛋️ Amenity Detection</span>
    <span class="ad-chip">✏️ Editable Results</span>
    <span class="ad-chip">📝 AI Descriptions</span>
    <span class="ad-chip">🎤 Voice Search <em style="color:#b45309">(coming soon)</em></span>
  </div>
</div>

<script>
// Typewriter animation
(function() {
  var sentences = [
    "AI-powered image analysis for real estate listings and property management.",
    "Search for a property based on amenities using your voice."
  ];
  var idx = 0, ch = 0, del = false;
  var el = document.getElementById("ad-typed");
  function tick() {
    var s = sentences[idx];
    if (!del) {
      el.textContent = s.slice(0, ch + 1); ch++;
      if (ch === s.length) { del = true; setTimeout(tick, 2200); return; }
      setTimeout(tick, 38);
    } else {
      el.textContent = s.slice(0, ch - 1); ch--;
      if (ch === 0) { del = false; idx = (idx + 1) % sentences.length; setTimeout(tick, 400); return; }
      setTimeout(tick, 18);
    }
  }
  setTimeout(tick, 1100);
})();

// Navigate to a Gradio tab by index (0=Upload, 1=Browse)
function adGoTab(n) {
  var tabs = document.querySelectorAll('.tab-nav button');
  if (tabs[n]) { tabs[n].click(); }
  // Scroll past the hero so the tab content is visible
  setTimeout(function() {
    var tabEl = document.querySelector('.tabs');
    if (tabEl) tabEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);
}
</script>
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
    labels = ["1. Upload", "2. Detect", "3. Review & Edit", "4. Generate Description"]
    parts: list[str] = [
        "<div style='display:flex;align-items:center;gap:0;margin:0 0 20px;flex-wrap:wrap;gap:4px;'>"
    ]
    for i, label in enumerate(labels):
        if i < active:
            # Completed step — amber dot with checkmark
            dot_style = "background:#b45309;color:white;"
            text_style = "color:#b45309;"
            symbol = "✓"
        elif i == active:
            # Current step — amber outline
            dot_style = "background:#fef3c7;color:#b45309;border:2px solid #b45309;"
            text_style = "color:#1c1917;font-weight:700;"
            symbol = str(i + 1)
        else:
            # Future step — grey
            dot_style = "background:#f0ede4;color:#a8a29e;"
            text_style = "color:#a8a29e;"
            symbol = str(i + 1)

        parts.append(
            f"<div style='display:flex;align-items:center;gap:6px;'>"
            f"<div style='width:26px;height:26px;border-radius:50%;display:flex;align-items:center;"
            f"justify-content:center;font-size:11px;font-weight:800;flex-shrink:0;{dot_style}'>{symbol}</div>"
            f"<span style='font-size:12px;font-weight:600;{text_style}'>{label}</span>"
            f"</div>"
        )
        if i < len(labels) - 1:
            line_color = "#b45309" if i < active else "#e5e0d8"
            parts.append(
                f"<div style='flex:1;min-width:20px;max-width:40px;height:2px;background:{line_color};'></div>"
            )
    parts.append("</div>")
    return "".join(parts)


# ── Upload tab handlers ────────────────────────────────────────────────────────


def upload_and_detect(
    files: list[Any] | None,
    property_name: str,
    model_name: str,
    extra_info: str,
) -> tuple[str, str, list[list[Any]], str, dict[str, str]]:
    """
    Handle 'Upload & Detect Amenities' button click.

    Calls POST /api/v1/properties/upload and returns the detected amenities
    as an editable table. The description is NOT generated at this step —
    the user reviews/edits first, then clicks Confirm.

    Args:
        files:          Uploaded image files (Gradio file objects).
        property_name:  User-provided property label.
        model_name:     Selected VLM.
        extra_info:     Optional free-text notes.

    Returns:
        Tuple of 5 values for the 5 output components:
          - step_html:      Updated step indicator (step 2 active = Detect)
          - status_text:    Short status / error message.
          - amenity_rows:   Table rows for the editable amenity DataFrame.
          - description:    Empty at this stage.
          - upload_state:   Dict storing {property_id, model_name} for the confirm step.
    """
    empty_state: dict[str, str] = {}

    if not files:
        return _step_html(0), "Please upload at least one image.", [], "", empty_state
    if not property_name.strip():
        return _step_html(0), "Please enter a property name.", [], "", empty_state
    if not model_name:
        return _step_html(0), "Please select a model.", [], "", empty_state

    try:
        opened_files: list[tuple[str, bytes, str]] = []
        for f in files:
            file_path = f if isinstance(f, str) else (f.name if hasattr(f, "name") else str(f))
            import os as _os

            filename = _os.path.basename(file_path)
            with open(file_path, "rb") as fh:
                content = fh.read()
            ext = filename.rsplit(".", 1)[-1].lower()
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(
                ext, "image/jpeg"
            )
            opened_files.append((filename, content, mime))

        multipart_files = [("files", (name, content, mime)) for name, content, mime in opened_files]
        form_data: dict[str, str] = {"name": property_name.strip(), "model_name": model_name}
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
            _step_html(0),
            "Request timed out. Try a faster model or fewer images.",
            [], "", empty_state,
        )
    except requests.exceptions.ConnectionError:
        return (
            _step_html(0),
            f"Cannot reach the API at {_API_BASE_URL}. Is the backend running?",
            [], "", empty_state,
        )
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = response.json().get("detail", "")
        except Exception:
            pass
        return _step_html(0), f"Upload failed: {detail or exc}", [], "", empty_state
    except Exception as exc:
        return _step_html(0), f"Unexpected error: {exc}", [], "", empty_state

    data: dict[str, Any] = response.json()
    prop: dict[str, Any] = data.get("property", {})
    property_id: str = data.get("property_id", "")

    amenity_rows = _format_amenities_table(prop.get("images", []))
    status = f"Detected {len(amenity_rows)} amenities for '{property_name}'. Review and edit below, then click Confirm."
    state = {"property_id": property_id, "model_name": model_name}

    return _step_html(2), status, amenity_rows, "", state


def confirm_and_describe(
    amenity_table: list[list[Any]],
    upload_state: dict[str, str],
) -> tuple[str, str, str]:
    """
    Handle 'Confirm & Generate Description' button click.

    Takes the (possibly edited) amenity table from gr.Dataframe and calls
    POST /api/v1/properties/{id}/describe to get a fresh VLM description
    based only on the amenities the user confirmed as present.

    Args:
        amenity_table:  Rows from the editable Dataframe:
                        [room_type, amenity_name, is_present (bool), confidence].
        upload_state:   Dict with {property_id, model_name} from upload step.

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

    # Convert table rows → DescribeRequest amenities format
    amenities = [
        {"room_type": row[0], "amenity_name": row[1], "is_present": bool(row[2])}
        for row in amenity_table
        if len(row) >= 3
    ]

    try:
        response = requests.post(
            f"{_API_BASE_URL}/api/v1/properties/{property_id}/describe",
            json={"amenities": amenities, "model_name": model_name},
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return _step_html(2), "Description generation timed out. Try again.", ""
    except requests.exceptions.ConnectionError:
        return _step_html(2), f"Cannot reach API at {_API_BASE_URL}.", ""
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = response.json().get("detail", "")
        except Exception:
            pass
        return _step_html(2), f"Description failed: {detail or exc}", ""
    except Exception as exc:
        return _step_html(2), f"Unexpected error: {exc}", ""

    description: str = response.json().get("description", "")
    return _step_html(3), "Description generated. You can edit it below before saving.", description


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


# ── Build the Gradio app ───────────────────────────────────────────────────────


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

    with gr.Blocks(title="Amenity Detector", css=_CSS, theme=gr.themes.Soft()) as demo:

        # ── Hero landing page ──────────────────────────────────────────────────
        gr.HTML(_HERO_HTML)

        # ── Tabs ──────────────────────────────────────────────────────────────
        with gr.Tabs():

            # ── Upload & Detect tab ────────────────────────────────────────────
            with gr.Tab("↑ Upload & Detect"):

                # gr.State stores {property_id, model_name} between steps
                upload_state = gr.State({})

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
                        upload_btn = gr.Button("↑ Upload & Detect Amenities", variant="primary")

                    # Right column: results
                    with gr.Column(scale=2):
                        upload_status = gr.Textbox(label="Status", interactive=False)

                        amenities_table = gr.Dataframe(
                            headers=["Room", "Amenity", "Present", "Confidence"],
                            datatype=["str", "str", "bool", "str"],
                            label="Detected Amenities — toggle to confirm or reject",
                            interactive=True,  # lets user check/uncheck Present column
                            wrap=True,
                            col_count=(4, "fixed"),
                        )

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
                    inputs=[image_files, property_name_input, model_dropdown, extra_info_input],
                    outputs=[step_indicator, upload_status, amenities_table, description_output, upload_state],
                )

                confirm_btn.click(
                    fn=confirm_and_describe,
                    inputs=[amenities_table, upload_state],
                    outputs=[step_indicator, upload_status, description_output],
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

                search_btn.click(fn=search_properties, inputs=[amenity_search_input], outputs=[results_table])
                list_all_btn.click(fn=list_all_properties, inputs=[], outputs=[results_table])
                view_btn.click(fn=get_property_detail, inputs=[property_id_input], outputs=[property_detail_output])

    return demo


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
    )
```

- [ ] **Step 4: Run the UI helper tests**

```bash
uv run pytest tests/unit/test_ui_helpers.py -v
```

Expected: 5 tests PASSED.

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
uv run pytest --tb=short -q
```

Expected: All existing tests still pass. If any fail, check that your imports didn't break anything in `ui/app.py`.

- [ ] **Step 6: Commit**

```bash
git add ui/app.py tests/unit/test_ui_helpers.py
git commit -m "feat: redesign Gradio UI with landing page and multi-step upload flow

- Hero landing page with typewriter animation and amber/cream palette
- Upload tab: step indicator, editable amenity DataFrame, Confirm button
- Browse tab: cleaner layout with property detail panel
- UI now surfaces API error details in status field"
```

---

## Task 5: Update README with Testing Steps (T3)

**Files:**
- Modify: `Readme.md`

- [ ] **Step 1: Add a Testing section to `Readme.md`**

Find the section that describes how to run the app and add the following after it:

```markdown
## Testing the Redesigned UI

Follow these steps to verify everything works after the latest changes.

### Prerequisites
1. Docker Desktop (or Docker Engine + Compose) is running.
2. In a WSL terminal, start Ollama with host binding:
   ```bash
   OLLAMA_HOST=0.0.0.0 ollama serve
   ```
3. Your `.env` file has:
   ```
   OLLAMA_BASE_URL=http://host.docker.internal:11434
   GEMINI_API_KEY=<your key, or leave blank to use Ollama only>
   ```

### Start the stack
```bash
docker compose up --build
```
Wait until all health checks pass (~30s). Then open:
- **UI:** http://localhost:7860
- **API docs:** http://localhost:8000/docs
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (admin/admin)

### Test the landing page
- Open http://localhost:7860
- Verify the hero section loads with the amber/cream colour scheme
- Verify the typewriter animation alternates between the two sentences
- Click **"↑ Upload & Detect"** — confirms you land on the Upload tab
- Click **"🔍 Browse Properties"** — confirms you land on the Browse tab

### Test the Upload flow (with Ollama)
1. Click **"↑ Upload & Detect"** tab
2. Upload 2–3 room photos (JPEG or PNG)
3. Enter a property name (e.g. "Test House")
4. Select **qwen2.5vl:7b** from the model dropdown
5. Click **"↑ Upload & Detect Amenities"**
6. Verify the step indicator advances to step 3 (Review & Edit)
7. Verify the amenity table populates with rooms and amenities
8. Toggle a few checkboxes on/off in the Present column
9. Click **"✔ Confirm & Generate Description"**
10. Verify the step indicator reaches step 4 (Generate Description)
11. Verify a description appears in the text box below

### Test the Browse flow
1. Click **"🔍 Browse Properties"** tab
2. Click **"List All"** — verify your uploaded property appears
3. Copy the property ID from the table
4. Paste it into the Property ID field and click **"View Details"**
5. Verify room and amenity breakdown appears

### Gemini quota note
The Gemini 2.0 Flash free tier has a daily request limit (~1500 req/day per project).
If you see `429 RESOURCE_EXHAUSTED` in `docker compose logs api`, the quota has been
exhausted for today. Switch to `qwen2.5vl:7b` and try again, or wait until tomorrow.
The quota resets at midnight Pacific Time.

### Check logs for errors
```bash
docker compose logs api --tail=50
```
There should be no `Cannot connect to Ollama at http://localhost:11434` errors.
If you see them, verify your `.env` has `OLLAMA_BASE_URL=http://host.docker.internal:11434`
and restart: `docker compose down && docker compose up`.
```

- [ ] **Step 2: Commit**

```bash
git add Readme.md
git commit -m "docs: add UI testing steps and Gemini quota note to README"
```

---

## Self-Review

**Spec coverage check:**
- T1 (Ollama networking) → Task 1 ✓
- T1 (Gemini quota) → Task 5 README + Task 1 Step 3 ✓
- T1 (surface errors in UI) → Task 4 (`upload_and_detect` returns detail from HTTPError) ✓
- T2 (landing page) → Task 4 (`_HERO_HTML` + `gr.HTML`) ✓
- T2 (warm neutral palette) → Task 4 (`_CSS` constants) ✓
- T2 (typewriter animation) → Task 4 (`_HERO_HTML` JS) ✓
- T2 (Upload tab step indicator) → Task 4 (`_step_html()` + `gr.HTML`) ✓
- T2 (editable amenity table) → Task 4 (`gr.Dataframe(interactive=True, datatype=["str","str","bool","str"])`) ✓
- T2 (Confirm & Generate Description) → Task 2 (endpoint) + Task 4 (`confirm_and_describe`) ✓
- T2 (Browse tab redesign) → Task 4 ✓
- T3 (testing steps) → Task 5 ✓

**Placeholder scan:** No TBDs. All code blocks are complete. All commands have expected output.

**Type consistency:** `_format_amenities_table` returns `list[list[Any]]` in Tasks 3 and 4. `confirm_and_describe` reads `row[2]` as bool — consistent with `datatype=["str","str","bool","str"]` set on the Dataframe. `upload_state` dict keys (`property_id`, `model_name`) match between `upload_and_detect` and `confirm_and_describe`.
