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
| Local VLM serving | Ollama | Handles model downloads, quantization, and unified API automatically |
| Cloud VLM | Gemini 2.0 Flash API | Free up to 1500 req/day, no GPU required, fastest option |
| Deployment target | Local (Docker Compose) → Cloud later | Design for portability from day one |
| CI/CD | GitHub Actions | Free, integrates with existing GitHub repo |
| Config | Hydra + `.env` for secrets | Flexible config, secrets kept out of code |

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
│              ┌─────────┴──────────┐                                 │
│              ▼                    ▼                                  │
│   ┌──────────────────┐   ┌──────────────────┐                      │
│   │   Ollama Server   │   │  Gemini API      │                      │
│   │   (port 11434)    │   │  (external)      │                      │
│   │  - Qwen2.5-VL-7B │   │  - Flash 2.0     │                      │
│   │  - LLaMA 3.2 11B  │   └──────────────────┘                     │
│   └──────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────┘
```

> **Note on GPU constraints**: The GTX 1660 Ti has 6GB VRAM. With Ollama's default 4-bit quantization (Q4_K_M):
> - Qwen2.5-VL-7B: ~5GB VRAM — fits comfortably.
> - LLaMA 3.2 Vision 11B: ~6.5GB — slightly over; Ollama will offload some layers to CPU (works, but slower). This is a valuable learning experience: you'll directly see the VRAM/speed trade-off.
> - Gemini 2.0 Flash: No GPU needed — useful as a fast baseline.

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
│   ├── ollama_client.py        # Ollama API client (Qwen, LLaMA)
│   └── gemini_client.py        # Google Generative AI client
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
│   ├── Dockerfile.ui           # Gradio service
│   └── Dockerfile.ollama       # Ollama + model pre-pull
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

Three concrete implementations:
1. `OllamaClient` — calls `http://localhost:11434` (Qwen2.5-VL-7B or LLaMA 3.2 Vision)
2. `GeminiClient` — calls Google Generative AI API (Gemini 2.0 Flash)

A `ModelRegistry` will hold the mapping from model name (string from config/UI) to the right client instance. This makes model switching a config change, not a code change.

---

## API Endpoints (FastAPI)

### Properties
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/properties/upload` | Upload images + optional metadata; triggers amenity detection |
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
- Dropdown: "Select VLM" (Qwen2.5-VL-7B | LLaMA 3.2 Vision | Gemini 2.0 Flash)
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
- [ ] Install and configure Ollama locally (outside Docker for now) — manual step, see README
- [ ] Pull quantized models: `ollama pull qwen2.5vl:7b` and `ollama pull llama3.2-vision:11b` — manual step
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

- [ ] Implement `ui/app.py` — Upload tab and Browse tab
- [ ] Upload tab calls `POST /api/v1/properties/upload`, displays results
- [ ] Browse tab calls `GET /api/v1/properties/search`, renders table
- [ ] Model selector dropdown reads from `GET /api/v1/models`
- [ ] Add Gradio service to `docker-compose.yml`
- [ ] End-to-end manual test: upload images → see amenities → browse

**Deliverable**: `docker compose up` starts everything; full upload-to-browse flow works.

---

### Phase 4 — Observability & Hardening
**Goal**: Make it feel like a real production service.

- [ ] Structured JSON logging across all services
- [ ] Add request/response logging middleware to FastAPI
- [ ] Add Prometheus metrics endpoint (`/metrics`) to FastAPI
- [ ] Optionally: add Grafana + Prometheus to `docker-compose.yml`
- [ ] Improve confidence scoring (parse model output more robustly)
- [ ] Handle edge cases: no amenities found, model timeout, bad image format
- [ ] Expand test coverage to ~70%+

**Deliverable**: `docker compose up` with monitoring stack; resilient to bad inputs.

---

### Phase 5 — Cloud Migration (future)
**Goal**: Deploy to a cloud provider without major code changes.

- [ ] Choose cloud provider (AWS/GCP recommended for free tier)
- [ ] Replace local PostgreSQL with managed DB (RDS / Cloud SQL)
- [ ] Replace local image store with object storage (S3 / GCS)
- [ ] Deploy FastAPI + Gradio containers to a cloud container service
- [ ] Update GitHub Actions CD pipeline to deploy on merge to `main`
- [ ] For VLMs: evaluate GPU cloud instance vs. Ollama-on-cloud vs. fully managed API (Gemini)

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
| Local VLM serving | Ollama | latest |
| VLM SDK (Gemini) | `google-genai` | latest (switched from deprecated google-generativeai) |
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

## Open Questions (to revisit before Phase 2)

1. **Image storage in cloud**: S3 vs GCS — depends on which cloud provider you pick in Phase 5. No decision needed now.
2. **Confidence scoring**: The VLMs return free-text, not structured scores. We'll need to parse or prompt-engineer for this. Exact strategy TBD during Phase 1 experimentation.
3. **LLaMA 3.2 Vision 11B on 6GB VRAM**: May require CPU offloading (slow). We'll measure latency during Phase 1 and document findings. If too slow, we drop to Qwen2.5-VL-7B as the sole local option.

-- 

## Important pointers when developing code

1. **Testing each phase**: Testing of each phase is necessary and required as we try to make a robust application.
2. **Decision documentation and explanation**: When writing code, also explain the decision behind this and present an ARCHITECTURE.md file, that keeps updating. Please ignore this file as we do not want this to be shown on github.
