# Amenity Detector — Project Specification

## Overview

This document defines the rework of the Amenity Detector project from a quick prototype into a well-structured, end-to-end AI Engineering project. The goal is **dual**: build a useful product _and_ use it as a learning vehicle for modern software and AI engineering practices.

**What it does**: Given property images (e.g., AirBnB listings), automatically identify amenities present in each room, score their presence, store the results, and generate a natural language description of the property. Users can later browse/search properties by amenities.

---

## Decisions Made

| Concern | Decision | Rationale |
|---|---|---|
| Package manager | `uv` | Faster than pip, modern Python packaging, reproducible lockfile |
| Frontend | Gradio + FastAPI | Python-only, purpose-built for ML demos, clean API separation |
| Database | PostgreSQL (Docker) | Mirrors production patterns; swap connection string to go to cloud |
| VLM provider | OpenRouter (OpenAI-compatible) | Single API key, four models behind one client, no local GPU constraint |
| Deployment target | Local (Docker Compose) → Cloud later | Design for portability from day one |
| CI/CD | GitHub Actions | Free, integrates with existing GitHub repo |
| Config | `.env` for secrets only | No Hydra; runtime config is environment variables |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Docker Compose (local)                        │
│                                                                      │
│  ┌────────────────────┐     ┌─────────────────────┐                 │
│  │   Gradio UI         │────▶│   FastAPI Backend    │                │
│  │   (port 7860)       │     │   (port 8000)        │                │
│  └────────────────────┘     └──────────┬──────────┘                 │
│                                         │                            │
│                          ┌──────────────┼──────────────┐            │
│                          ▼              ▼              ▼            │
│               ┌───────────────┐  ┌──────────┐  ┌──────────────┐   │
│               │ VLM Abstraction│  │PostgreSQL│  │  File Store  │   │
│               │    Layer       │  │(port 5432│  │(images/local)│   │
│               └───────┬───────┘  └──────────┘  └──────────────┘   │
│                        │                                             │
│                        ▼                                             │
│             ┌──────────────────────────┐                            │
│             │ OpenRouter API (external) │                           │
│             │  openai/gpt-4o-mini       │                           │
│             │  google/gemini-pro-1.5    │                           │
│             │  meta-llama/llama-3.2-11b │                           │
│             │  qwen/qwen2-vl-72b        │                           │
│             └──────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
```

> **Why one provider**: OpenRouter exposes every supported model through a
> single OpenAI-compatible endpoint. Switching models is a string change in
> the registry; no GPU, no local server, no per-provider SDK.

---

## Repository Structure (target)

```
amenity_detector/
├── api/                        # FastAPI application
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entrypoint
│   ├── routers/
│   │   ├── properties.py       # Upload, retrieve, search endpoints
│   │   └── models.py           # List/select available VLMs
│   └── dependencies.py         # DB session, model registry injection
│
├── core/                       # Business logic (model-agnostic)
│   ├── __init__.py
│   ├── amenity_schema.py       # Amenity definitions (keep existing)
│   ├── amenity_detector.py     # Orchestrates VLM calls + parsing
│   ├── amenity_data_manager.py # DB read/write logic
│   └── amenity_system.py       # High-level pipeline wrapper
│
├── models/                     # VLM abstraction layer
│   ├── __init__.py
│   ├── base.py                 # Abstract VLMClient interface
│   └── openrouter_client.py    # OpenAI-compatible OpenRouter client
│
├── db/                         # Database layer
│   ├── __init__.py
│   ├── models.py               # SQLAlchemy ORM models
│   ├── session.py              # DB connection + session factory
│   └── migrations/             # Alembic migration scripts
│
├── ui/                         # Gradio frontend
│   ├── __init__.py
│   └── app.py                  # Gradio app (tabs: Upload, Browse)
│
├── config/
│   ├── config.yaml             # Hydra config
│   └── models.yaml             # VLM model definitions
│
├── tests/
│   ├── unit/                   # Unit tests (no external deps)
│   └── integration/            # Integration tests (real DB, mock VLM)
│
├── docker/
│   ├── Dockerfile.api          # FastAPI service
│   └── Dockerfile.ui           # Gradio service
│
├── .github/
│   └── workflows/
│       └── ci.yml              # Lint, type-check, test
│
├── docker-compose.yml          # Orchestrates all services
├── pyproject.toml              # uv / project metadata
├── uv.lock                     # Lockfile (committed to repo)
├── .env.example                # Template for secrets (never commit .env)
├── CLAUDE.md
├── SPEC.md
└── Readme.md
```

---

## Data Model

### `properties` table
| Column | Type | Notes |
|---|---|---|
| `id` | UUID (PK) | Unique property ID, auto-generated |
| `name` | VARCHAR | Human-readable label (e.g., "Frankfurt House 1") |
| `description` | TEXT | VLM-generated natural language description |
| `model_used` | VARCHAR | Which VLM produced the result |
| `extra_info` | TEXT | Free-text metadata provided by the user at upload |
| `created_at` | TIMESTAMP | Auto-set on insert |

### `images` table
| Column | Type | Notes |
|---|---|---|
| `id` | UUID (PK) | |
| `property_id` | UUID (FK → properties) | |
| `file_path` | VARCHAR | Local/cloud path to the stored image |
| `room_type` | VARCHAR | Detected room type (kitchen, bedroom, etc.) |

### `detected_amenities` table
| Column | Type | Notes |
|---|---|---|
| `id` | UUID (PK) | |
| `property_id` | UUID (FK → properties) | |
| `image_id` | UUID (FK → images) | |
| `amenity_name` | VARCHAR | e.g., "refrigerator" |
| `room_type` | VARCHAR | e.g., "kitchen" |
| `confidence` | FLOAT | 0.0–1.0 score (parsed from VLM output or heuristic) |
| `is_present` | BOOLEAN | True if detected |

---

## VLM Abstraction Layer

A clean interface so the rest of the codebase doesn't care which model is in use:

```python
# models/base.py (conceptual sketch — not final code)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import Image

@dataclass
class VLMResponse:
    raw_text: str           # Raw text from the model
    model_name: str         # Which model was used

class VLMClient(ABC):
    """Abstract base class all VLM clients must implement."""

    @abstractmethod
    def generate(self, image: Image.Image, prompt: str) -> VLMResponse:
        """Send image + prompt, return model response."""
        ...
```

One concrete implementation:
1. `OpenRouterVLMClient` — uses the `openai` SDK against
   `https://openrouter.ai/api/v1`. The constructor takes the model id
   (e.g. `openai/gpt-4o-mini`) and a `json_mode` flag.

`ModelRegistry` holds a list of supported model ids and instantiates the
client lazily per id. Adding a new model means appending one entry to
`AVAILABLE_MODELS`.

---

## API Endpoints (FastAPI)

### Properties
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/properties/` | Create an empty property shell (name + model) |
| `POST` | `/api/v1/properties/{id}/images` | Upload and process one image |
| `POST` | `/api/v1/properties/{id}/describe` | Regenerate description from reviewed amenities |
| `GET` | `/api/v1/properties/` | List all properties (paginated) |
| `GET` | `/api/v1/properties/{id}` | Get full property details (amenities, description, images) |
| `GET` | `/api/v1/properties/search` | Filter by amenities (e.g., `?amenities=pool,wifi`) |
| `DELETE` | `/api/v1/properties/{id}` | Remove a property |

### Models
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/models` | List available VLMs and their status (loaded / available) |

### Health
| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check (used by Docker Compose health checks) |

---

## Gradio UI

Two tabs:

### Tab 1 — Upload Property
- Image upload (multi-file supported)
- Text field: "Property name"
- Text area: "Additional info / notes" (e.g., "This is a 2-bed flat in Frankfurt, has central heating")
- Dropdown: "Select VLM" (GPT-4o Mini | Gemini Pro 1.5 | Llama 3.2 11B Vision | Qwen2-VL 72B)
- Submit button → shows detected amenities per room + generated description

### Tab 2 — Browse Properties
- Search bar: type amenity requirements (e.g., "pool, wifi, parking")
- Results table: property ID, name, matched amenities, description snippet
- Click a row → expand full property details + images

---

## Implementation Phases

### Phase 0 — Cleanup & Migration (start here)
**Goal**: Get the existing codebase into a clean, working state before adding anything new.

- [x] Migrate from `pip` + `requirements.txt` to `uv` + `pyproject.toml`
- [x] Add `.gitignore` entries for `uv.lock` artifacts, `__pycache__`, `.env`
- [x] Rename/remove the `amenity_detector/` virtual environment folder that is currently tracked in git
- [x] Set up `pre-commit` hooks: `ruff` (linting), `mypy` (type checking)
- [x] Set up GitHub Actions CI: runs linting + type checking + unit tests on every push/PR

**Deliverable**: Clean repo, `uv run python main.py` works, CI passes.

---

### Phase 1 — VLM Abstraction + Ollama Setup
**Goal**: Replace the hard-coded LLaVA model with a pluggable VLM layer.

- [x] Start to ignore SPEC.md file and also resources folder using .gitignore
- [x] Cleanup the .gitignore file and make sure only the required files/folders are mentioned
- [x] Install and configure Ollama locally (outside Docker for now) — manual step, see README
- [x] Pull quantized models: `ollama pull qwen2.5vl:7b` and `ollama pull llama3.2-vision:11b` — manual step
- [x] Implement `models/base.py` — `VLMClient` ABC + `VLMResponse` dataclass
- [x] Implement `models/ollama_client.py` — wraps Ollama REST API
- [x] Implement `models/gemini_client.py` — wraps `google-generativeai` SDK
- [x] Implement `ModelRegistry` — maps model name string → client instance
- [x] Update `core/amenity_detector.py` to use `VLMClient` instead of direct HuggingFace calls
- [x] Write unit tests for the abstraction layer (mock HTTP calls)
- [ ] Manual test: run detection with all 3 models, compare output quality — deferred to Phase 3

**Deliverable**: `python main.py` works with any of the 3 models via a config change.

---

### Phase 2 — Database + FastAPI Backend
**Goal**: Replace SQLite with PostgreSQL and expose a proper REST API.

- [x] Set up `docker-compose.yml` with PostgreSQL service
- [x] Set up SQLAlchemy + Alembic for ORM and migrations
- [x] Define ORM models (`db/models.py`): `Property`, `Image`, `DetectedAmenity`
- [x] Implement `db/session.py` — connection factory using env-var `DATABASE_URL`
- [x] Rewrite `core/amenity_data_manager.py` to use SQLAlchemy instead of SQLite
- [x] Implement FastAPI app (`api/main.py`) with routers for properties and models
- [x] Add image file storage (local volume, Docker-mounted)
- [x] Write integration tests for API endpoints (using a test SQLite DB; override `TEST_DATABASE_URL` for PostgreSQL)

**Deliverable**: `docker compose up` starts the API + DB; `/api/v1/properties/upload` works end-to-end.

---

### Phase 3 — Gradio Frontend
**Goal**: Replace Streamlit with a Gradio UI that talks to the FastAPI backend.

- [x] Implement `ui/app.py` — Upload tab and Browse tab
- [x] Upload tab calls `POST /api/v1/properties/upload`, displays results
- [x] Browse tab calls `GET /api/v1/properties/search`, renders table
- [x] Model selector dropdown reads from `GET /api/v1/models`
- [x] Add Gradio service to `docker-compose.yml`
- [ ] End-to-end manual test: upload images → see amenities → browse — deferred (requires running services)

**Deliverable**: `docker compose up` starts everything; full upload-to-browse flow works.

---

### Phase 4 — Observability & Hardening
**Goal**: Make it feel like a real production service.

- [x] Structured JSON logging across all services (`api/logging_config.py`, `LOG_FORMAT=json`)
- [x] Add request/response logging middleware to FastAPI (`api/middleware.py`)
- [x] Add Prometheus metrics endpoint (`/metrics`) to FastAPI via `prometheus-fastapi-instrumentator`
- [x] Add Grafana + Prometheus to `docker-compose.yml` (ports 9090 and 3000)
- [x] Improve confidence scoring — VLM now returns `{"present": bool, "confidence": float}` per amenity
- [x] Handle edge cases: VLM timeouts/errors return empty dicts (no pipeline crash); bad images rejected at upload boundary
- [x] Expand test coverage to ~70%+ (119 tests: unit tests for AmenityDetector, PropertyAmenitySystem, RequestLoggingMiddleware + existing 78 tests)

**Deliverable**: `docker compose up` with monitoring stack; resilient to bad inputs.

---

### Phase 5 — UX Rework + Speed
**Goal**: Fix the two things currently hurting the product — slow per-image inference and opaque feedback — and let the user contribute structured property hints.

**Design doc**: `docs/superpowers/specs/2026-04-19-phase-5-ux-rework-and-speed-design.md` (committed locally, `docs/` is gitignored).

**Problem recap**:
- Detection is slow because the VLM is asked "is this a kitchen? is this a bedroom?" per room type per image (O(rooms × amenities) prompt volume).
- No image preprocessing — full-resolution phone photos are sent straight to the VLM.
- No per-image feedback — the UI sits silent until everything is done.
- No channel for the user to contribute what they already know about the property.

**What we're building**:

1. **Image preprocessing** — resize the longest edge to 768 px before inference. New module `core/preprocessing.py`, called from `AmenityDetector.detect_from_image()`. No filtering or colour changes (see Open Questions for EXIF + RGB conversion deferral).
2. **Single structured-JSON VLM call per image** — replaces the per-room loop. One prompt asks the VLM to return both the `room_type` label and per-amenity `{"present": bool, "confidence": float}` in a single JSON response. Public signature of `AmenityDetector.detect_from_image()` stays the same.
3. **Per-image API endpoints** —
   - `POST /api/v1/properties` creates an empty shell (name + model only).
   - `POST /api/v1/properties/{id}/images` handles one image at a time (preprocess → detect → persist → return).
   - Old batch `POST /api/v1/properties/upload` stays, marked `deprecated=True` for Phase 2/3 backward-compat.
4. **Sidebar hints in the Upload tab** — sliders/toggles for number of rooms, kitchen yes/no, balcony yes/no, living-room yes/no. Detection is **not blocked** by these; they can be filled while images process.
5. **Live progress bar** — `upload_and_detect` becomes a `gr.Progress()` generator. UI loops through files and calls the per-image endpoint once per file, yielding partial results so the amenity table fills in live and the bar reaches 100% when all images are done.
6. **Client-side reconciliation** — a pure helper compares the current sidebar hints against the detected amenity table and flags contradictions with a ⚠️ row marker (e.g. user says *balcony: no* but balcony amenities are present).
7. **`/describe` extension** — accepts the sidebar hints so the final description reflects the user's stated facts alongside the confirmed amenity list.

**Testing discipline** — every step has a test first; `uv run pytest` stays green between steps:
- Unit: `test_preprocessing.py`, `test_amenity_detector.py` (JSON parsing), `test_ui_helpers.py` (reconciliation), `test_amenity_system.py` (hints-in-prompt).
- Integration: `test_properties_api.py` for new shell + per-image endpoints, plus deprecation check on the old batch route.

**Implementation order**:
1. `core/preprocessing.py` + tests
2. `AmenityDetector` rewrite (structured JSON) + tests
3. New API endpoints + tests
4. `/describe` hints extension + tests
5. UI sidebar + progress generator + reconciliation + tests
6. Mark old batch `/upload` deprecated
7. Manual end-to-end test (Gradio → API → DB, with both Gemini and Ollama backends)

**Deliverable**: `docker compose up` runs the full new flow. A user can upload N images, watch a live progress bar, fill sidebar hints in parallel, see mismatch warnings in the review table, edit, confirm, and get a description that reflects both the confirmed amenities and the sidebar hints.

---

### Phase 6 — Amenity Review Redesign
**Goal**: Replace the flat editable `gr.Dataframe` with a per-room, per-row action UI that the user actually wants to use — and fix the pandas-truthiness class of bugs for good.

**Design doc**: `docs/superpowers/specs/2026-04-19-phase-6-amenity-review-redesign-design.md` (committed locally, `docs/` is gitignored).

**Problem recap** (from `docs/error_logs.md` in the same branch):
- `gr.Dataframe` is handed to handlers as a pandas DataFrame in Gradio 6; `amenity_table or []` trips `DataFrame.__bool__`. Hot-fixed via `type="array"` + a defensive `_to_rows` helper, but the real cost is that the Dataframe component simply can't express the per-row UX the user spec'd (tick ✓, edit ✎, reject ✗, add +, greyed columns while editing).
- Per-room grouping isn't possible inside a single Dataframe.
- Inline `<script>` blocks inside `gr.HTML` are sanitised, so the hero typewriter was never actually running.

**What we're building**:

1. **`ui/review_state.py`** — pure state helpers: `from_detections`, `confirm_item`, `reject_item`, `start_edit`, `save_edit`, `cancel_edit`, `add_item`, `flatten_for_reconcile`, `amenities_for_describe`. State is one JSON-serialisable structure grouped by room. Every helper is a pure function (new list out, no mutation).
2. **`@gr.render`-driven review panel** — replaces the `gr.Dataframe`. Each room is a card; each amenity inside a card is a row with `[✓] [✎] [✗]` for pending, an inline textbox + `[✓ save][✗ cancel]` for editing, and a green chip for confirmed. A `+ Add amenity` button appends an editing row to that room.
3. **Confirmed chips stay grouped under the originating room** — the room → amenity mapping is preserved in the UI (user clarified this is important).
4. **Sidebar reconcile integration** — unchanged behaviour: hints flag contradictions via the `[warning] ` prefix on the name. We run `reconcile_amenities(flatten_for_reconcile(state))` and merge the prefixed names back into state.
5. **`/describe` read path** — `confirm_and_describe` reads state through `amenities_for_describe(state)`; that function returns every `"confirmed"` item plus every `"pending"` item where `present=True` (so the global Confirm button still works without per-row ticks). `"editing"` rows are excluded.
6. **Hero subtitle fallback** — drop the typewriter entirely (Gradio sanitises the `<script>`; the JS never runs). Replace with a static two-line subtitle that mentions both image analysis and voice-driven search.

**Not in scope** (explicit):
- Restoring the typewriter. Revisit when the frontend leaves Gradio (the user's framing: this is a Node/TS-style concern).
- Gemini free-tier workarounds. The free tier is **20 RPD per day** for `gemini-2.5-flash` — effectively unusable for multi-image uploads. Tracked as a separate decision in Phase 7.

**Testing discipline** — every helper has a test first; `uv run pytest` stays green:
- Unit: `test_review_state.py` covering each pure helper + edge cases (reject last item in room → room removed; cancel_edit on never-detected item → item removed).
- Smoke: `ui.app.build_app()` returns a `gr.Blocks` instance without raising — catches `@gr.render` wiring mistakes.
- Existing `test_ui_helpers.py` stays green (the `reconcile_amenities` contract is unchanged).

**Implementation order**:
1. `ui/review_state.py` + `tests/unit/test_review_state.py`.
2. Remove the `gr.Dataframe`, wire in the `@gr.render` panel, migrate `upload_and_detect` + `confirm_and_describe` to state.
3. Drop `_HERO_JS` and the embedded `<script>`; swap hero subtitle to static copy.
4. Run full suite, manual smoke via `docker compose up`, commit.

**Deliverable**: `docker compose up` runs the new review flow. Per-row confirm/edit/reject/add all work; confirmed rows render as green chips under their room; the global Confirm button still generates a description; no pandas-truthiness errors anywhere.

---

### Phase 8 — OpenRouter migration and project cleanup (active)

**Goal**: Collapse the multi-provider VLM layer (Ollama + Gemini) into one
OpenRouter-backed client, and delete every legacy file that is no longer on
the live Docker Compose path.

**Design doc**: `docs/superpowers/specs/2026-05-02-openrouter-migration-and-cleanup-design.md`.

**What we're building**:

1. New `models/openrouter_client.py` — `OpenRouterVLMClient` implementing
   `VLMClient`. Uses the `openai` SDK pointed at
   `https://openrouter.ai/api/v1`. Sends images as base64 data URLs in
   `image_url` blocks. Honours `json_mode=True` for GPT-4o-mini and Gemini.
2. `models/registry.py` rewrite. `AVAILABLE_MODELS` lists four entries:
   `openai/gpt-4o-mini`, `google/gemini-pro-1.5`,
   `meta-llama/llama-3.2-11b-vision-instruct`, `qwen/qwen2-vl-72b-instruct`.
   `from_env()` instantiates one `OpenRouterVLMClient` per id.
3. Delete legacy code: `streamlit/`, `model/`, `dataloader/`, `utilities/`,
   `main.py`, `config/`, `amenity_detector.db`, `models/ollama_client.py`,
   `models/gemini_client.py`, related tests.
4. Delete the deprecated batch endpoint
   `POST /api/v1/properties/upload`, the `UploadResponse` schema,
   `PropertyAmenitySystem.process_upload()`, and integration tests for the
   removed route.
5. Dependency cleanup: add `openai>=1.30.0`; drop `google-genai`,
   `hydra-core`, `omegaconf`, `matplotlib`, `scikit-learn`, the `local-gpu`
   group, and matching mypy/ruff configuration.
6. Environment cleanup: remove `GEMINI_API_KEY` and `OLLAMA_BASE_URL` from
   `.env`, `.env.example`, `docker-compose.yml`, and `Dockerfile.api`. Add
   `OPENROUTER_API_KEY`. Drop `extra_hosts` from the `api` service.
7. Documentation refresh: this `SPEC.md` (you're reading it),
   `Readme.md`, and `ARCHITECTURE.md` reflect the OpenRouter-only stack and
   the new test workflow.

**Testing discipline**:

- Unit: new `test_openrouter_client.py` (constructor, `json_mode` flag, SDK
  errors → `RuntimeError`); updated `test_model_registry.py` (four ids,
  unknown id raises `KeyError`); deleted `test_ollama_client.py` and
  `test_gemini_client.py`.
- Integration: keep per-image and `/describe` coverage; delete the
  `/upload` cases.
- Manual: `docker compose up --build`, hit the UI, run an upload with
  `openai/gpt-4o-mini`, confirm description and Browse tab.

**Implementation order**:

1. `OpenRouterVLMClient` + unit tests.
2. Registry rewrite + tests.
3. Update `api/routers/models.py` descriptions and `ui/app.py` fallback.
4. Delete dead code + tests; full suite green.
5. Delete `/upload` route, schema, system method, and tests.
6. Update `pyproject.toml`, `.env.example`, `docker-compose.yml`,
   `Dockerfile.api`.
7. Refresh `Readme.md`, `ARCHITECTURE.md`.
8. `docker compose up --build` smoke run.

**Deliverable**: `docker compose up --build` runs the full flow against
OpenRouter. The repository contains only files used by the live pipeline.
Tests, lint, and type checks pass. README documents how to test the new
flow end-to-end.

---

### Phase 9 — Cloud Migration & Model Serving (future)
**Goal**: Deploy to a cloud provider without major code changes, and fix the model-serving bottleneck that makes local demos frustrating.

**Context** (after Phase 8):
- Phase 8 removed the local-GPU bottleneck by routing every model through
  OpenRouter. The remaining cloud work is hosting the FastAPI/UI containers
  and the database/storage.

**Cloud deployment tasks**:
- [ ] Choose cloud provider (AWS/GCP free tier or Fly.io for the app containers).
- [ ] Replace local PostgreSQL with managed DB (RDS / Cloud SQL / Neon).
- [ ] Replace local image store with object storage (S3 / GCS / R2).
- [ ] Deploy FastAPI + Gradio containers to a cloud container service.
- [ ] Add GitHub Actions CD pipeline that deploys on merge to `main`.
- [ ] Optionally add a self-hosted endpoint as a second `VLMClient` for cost
      comparison.

---

## Tech Stack Summary

| Layer | Tool | Version target |
|---|---|---|
| Python | CPython | 3.11+ |
| Package manager | `uv` | latest |
| Web framework | FastAPI | 0.110+ |
| Frontend | Gradio | 4.x |
| ORM | SQLAlchemy | 2.x |
| DB migrations | Alembic | latest |
| Database | PostgreSQL | 16 |
| Config | Hydra-core | 1.3+ |
| VLM provider | OpenRouter (OpenAI-compatible) | — |
| VLM SDK | `openai` | 1.30+ |
| Linting | `ruff` | latest |
| Type checking | `mypy` | latest |
| Testing | `pytest` + `httpx` | latest |
| Containerization | Docker + Docker Compose | v2 |
| CI/CD | GitHub Actions | — |

---

## What We Are NOT Doing (yet)

- **RAG pipeline**: Mentioned in the README for non-visual amenities (e.g., WiFi, heating). Deliberately deferred to avoid scope creep. Placeholder comments will mark where it hooks in.
- **Fine-tuned YOLO/DETR models**: A valid optimization path but requires labeled data. Out of scope for the rework phases.
- **User authentication**: Not needed for a learning project. Can be added later via FastAPI middleware.
- **Agentic / conversational flows**: Interesting idea from the README. Deferred until the base pipeline is solid.

---

## Open Questions (revisit later)

1. **Image storage in cloud**: S3 vs GCS — depends on which cloud provider you pick in Phase 6. No decision needed now.
2. **Confidence scoring**: The VLMs return free-text, not structured scores. Resolved in Phase 4 (VLM returns `{"present": bool, "confidence": float}` per amenity).
3. **LLaMA 3.2 Vision 11B on 6GB VRAM**: Closed in Phase 8. All inference now runs through OpenRouter; the local-GPU path is gone.
4. **Zero-shot room classifier** *(deferred from Phase 5 Q1 option C)*: Use a small CLIP-style model to pick the room type before the VLM runs, so the VLM can focus only on amenity checks. Likely a meaningful speed win, but introduces a new model dependency. Revisit after measuring Phase 5 numbers.
5. **EXIF auto-orient + RGB conversion** *(deferred from Phase 5 Q4b)*: Not part of Phase 5 preprocessing (resize only). Worth adding when rotated phone photos are observed in the wild, or when a VLM rejects images with alpha channels. Cheap, correctness-focused win — just no evidence we need it yet.
6. **Image processing parallelism** *(deferred from Phase 5 Q5)*: `concurrent.futures.ThreadPoolExecutor` across per-image calls. Helps most with paid Gemini or cloud GPUs; Ollama on the GTX 1660 Ti is VRAM-bound and won't benefit. Revisit once Phase 5 gives us baseline numbers.

---

## Choosing An OpenRouter Model

| Model id | Best for | JSON mode |
|---|---|---|
| `openai/gpt-4o-mini` | Cheapest reliable JSON output; the default. | Yes |
| `google/gemini-pro-1.5` | Strong vision reasoning, JSON-safe. | Yes |
| `meta-llama/llama-3.2-11b-vision-instruct` | Open-weights baseline. | No (parser handles plain JSON) |
| `qwen/qwen2-vl-72b-instruct` | Highest-capacity open-weights option. | No (parser handles plain JSON) |

OpenRouter pricing changes frequently. Check
https://openrouter.ai/models for the current per-1M-token rates before
running a long evaluation.

---

## Important pointers when developing code

1. **Testing each phase**: Testing of each phase is necessary and required as we try to make a robust application.
2. **Decision documentation and explanation**: When writing code, also explain the decision behind this and present an ARCHITECTURE.md file, that keeps updating. Please ignore this file as we do not want this to be shown on github.
