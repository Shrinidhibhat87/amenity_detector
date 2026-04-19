# Amenity Detector

Amenity Detector is an end-to-end AI engineering project for analysing property
photos. It detects visible amenities room by room, lets a user review and correct
the model output, stores the result in PostgreSQL, and generates a listing-style
property description.

The project is intentionally built like a small production system: FastAPI
backend, Gradio frontend, PostgreSQL persistence, pluggable vision-language
models, Docker Compose orchestration, structured logs, Prometheus metrics,
Grafana dashboards, and CI checks.

## Current State

The local Docker Compose stack runs:

| Component | Status | Purpose |
|---|---:|---|
| Gradio UI | Done | Upload, per-room amenity review, property hints, browse/search |
| FastAPI API | Done | Property, image, model, search, and description endpoints |
| PostgreSQL | Done | Stores properties, images, and detected amenities |
| VLM clients | Done | Ollama and Gemini implementations behind a common interface |
| Image preprocessing | Done | Resizes images before inference to reduce VLM latency |
| Observability | Done | Structured logging, request logs, Prometheus, Grafana dashboard |
| CI | Done | Ruff lint/format, mypy, and unit tests |

Recent work added the Phase 5/6 flow:

1. Create a property shell.
2. Upload and process one image at a time.
3. Detect room type and amenities from one structured JSON VLM response.
4. Show incremental progress in the UI.
5. Review amenities grouped by room.
6. Confirm, edit, reject, or add amenities.
7. Reconcile property hints against detections.
8. Generate the final description from the reviewed state.

## Features

- Multi-image property upload through Gradio or REST.
- Per-image processing endpoint for live progress updates.
- Pluggable VLM layer:
  - `gemini-2.0-flash`
  - `qwen2.5vl:7b`
  - `llama3.2-vision:11b`
- Structured JSON amenity detection with confidence scores.
- Room classification from the same VLM call as amenity detection.
- Image preprocessing with longest-edge resize to 768 px.
- Per-room review UI powered by `gr.render`.
- Row-level actions: confirm, edit, reject, add amenity.
- User hints for room count, kitchen, balcony, and living room.
- Description regeneration from reviewed amenities and user hints.
- Browse/search UI for stored properties.
- FastAPI OpenAPI docs at `/docs`.
- Prometheus metrics endpoint at `/metrics`.
- Grafana dashboard provisioning through Docker Compose.

## Architecture

```text
Browser
  |
  | http://localhost:7860
  v
Gradio UI
  |
  | REST calls
  v
FastAPI API
  |             \
  | SQLAlchemy   \ VLMClient
  v               v
PostgreSQL       Ollama or Gemini
  |
  v
Stored property, image, and amenity records
```

The UI never imports API internals. It talks to the backend through HTTP only.
The backend owns model selection, image storage, detection orchestration, and
database writes.

## Quick Start

### Prerequisites

- Docker Desktop or Docker Engine with Docker Compose v2
- `uv` for local tests and checks
- At least one model backend:
  - Gemini API key for `gemini-2.0-flash`
  - Ollama running locally for `qwen2.5vl:7b` or `llama3.2-vision:11b`

### Configure Environment

```bash
cp .env.example .env
```

For Gemini:

```bash
GEMINI_API_KEY=your_key_here
```

For Ollama from Docker Compose on WSL/Linux, start Ollama with host binding:

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

Then set:

```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

The default PostgreSQL values in `.env.example` match `docker-compose.yml`.

### Start the Stack

```bash
docker compose up --build
```

Open:

| Service | URL |
|---|---|
| Gradio UI | http://localhost:7860 |
| FastAPI docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Grafana uses the default local credentials unless changed in your environment:

```text
admin / admin
```

### Health Check

```bash
curl http://localhost:8000/health
```

Expected shape:

```json
{"status":"ok","database":"ok","version":"1.0.0"}
```

## Using The App

1. Open http://localhost:7860.
2. Go to `Upload & Detect`.
3. Upload 2-3 representative property images.
4. Enter a property name.
5. Select a model.
6. Optionally fill property hints.
7. Click `Upload & Detect Amenities`.
8. Review the detected amenities grouped by room.
9. Confirm, edit, reject, or add amenities as needed.
10. Click `Confirm & Generate Description`.
11. Use `Browse Properties` to list, search, and inspect stored properties.

For faster demos, prefer Gemini. For local-model learning, use Ollama, but expect
slower inference on low-VRAM GPUs.

## API Overview

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/properties/` | Create an empty property shell |
| `POST` | `/api/v1/properties/{id}/images` | Upload and process one image |
| `POST` | `/api/v1/properties/{id}/describe` | Regenerate description from reviewed amenities |
| `POST` | `/api/v1/properties/upload` | Deprecated batch upload endpoint |
| `GET` | `/api/v1/properties/` | List properties |
| `GET` | `/api/v1/properties/{id}` | Get full property details |
| `GET` | `/api/v1/properties/search` | Search by amenity names |
| `DELETE` | `/api/v1/properties/{id}` | Delete a property |
| `GET` | `/api/v1/models/` | List available models |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |

### Example: Per-Image Flow

Create a property:

```bash
curl -X POST http://localhost:8000/api/v1/properties/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Frankfurt Apartment",
    "model_name": "gemini-2.0-flash",
    "extra_info": "Two-bed flat near public transport"
  }'
```

Upload one image to the returned `property_id`:

```bash
curl -X POST http://localhost:8000/api/v1/properties/<property_id>/images \
  -F "model_name=gemini-2.0-flash" \
  -F "file=@/path/to/kitchen.jpg"
```

Generate a description from reviewed amenities:

```bash
curl -X POST http://localhost:8000/api/v1/properties/<property_id>/describe \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gemini-2.0-flash",
    "num_rooms": 2,
    "has_kitchen": true,
    "has_balcony": false,
    "has_living_room": true,
    "amenities": [
      {"room_type": "kitchen", "amenity_name": "refrigerator", "is_present": true},
      {"room_type": "living_room", "amenity_name": "sofa", "is_present": true}
    ]
  }'
```

## Local Development

Install dependencies:

```bash
uv sync --group dev
```

Run the API locally:

```bash
uv run uvicorn api.main:app --reload
```

Run the UI locally:

```bash
API_BASE_URL=http://localhost:8000 uv run python -m ui.app
```

When changing UI code in Docker, rebuild the UI service:

```bash
docker compose up -d --build ui
```

Rebuilding only the API service does not update Gradio code because the UI runs
in a separate container.

## Quality Checks

The GitHub Actions CI currently runs these commands:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy . --ignore-missing-imports
uv run pytest tests/unit/ -v
```

For broader local verification:

```bash
uv run pytest tests/ -v
```

Integration tests use an in-memory SQLite database by default. To run them
against local PostgreSQL:

```bash
TEST_DATABASE_URL=postgresql://amenity_user:amenity_pass@localhost:5432/amenity_db \
  uv run pytest tests/integration/ -v
```

## Project Structure

```text
amenity_detector/
├── api/                         # FastAPI app, schemas, routers, middleware
├── core/                        # Detection, preprocessing, orchestration, CRUD
├── db/                          # SQLAlchemy models, sessions, migrations
├── models/                      # VLMClient interface and model clients
├── ui/                          # Gradio frontend and review-state helpers
├── monitoring/                  # Prometheus and Grafana provisioning
├── docker/                      # API and UI Dockerfiles
├── tests/                       # Unit and integration tests
├── docker-compose.yml           # Local full-stack orchestration
├── pyproject.toml               # Dependencies and tool config
└── .github/workflows/ci.yml     # CI checks
```

## Important Notes

- `qwen2.5vl:7b` can be slow on 6 GB VRAM machines because Ollama may offload
  layers to CPU.
- The old batch upload endpoint remains available for compatibility but the UI
  now uses the property-shell plus per-image upload flow.
- `ARCHITECTURE.md`, `SPEC.md`, and internal design docs are intentionally
  ignored by git.
- Runtime uploads are stored under `storage/` and should not be committed.

## Roadmap

Completed:

- Repository cleanup with `uv`, ruff, mypy, pytest, and CI.
- VLM abstraction for Ollama and Gemini.
- PostgreSQL-backed FastAPI service.
- Docker Compose local stack.
- Gradio upload and browse UI.
- Per-image detection flow with incremental progress.
- Per-room amenity review UI.
- Structured logging, Prometheus metrics, and Grafana dashboards.

Next likely work:

- Cloud deployment target and managed database/storage.
- Hosted VLM strategy for faster demos.
- Voice-assisted property search.
- Richer property search and ranking.
