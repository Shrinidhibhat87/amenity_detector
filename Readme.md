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
| VLM provider | Done | OpenRouter (OpenAI-compatible) backs every supported model |
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
- One VLM client (OpenRouter) with four registered models:
  - `openai/gpt-4o-mini`
  - `google/gemini-pro-1.5`
  - `meta-llama/llama-3.2-11b-vision-instruct`
  - `qwen/qwen2-vl-72b-instruct`
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
PostgreSQL       OpenRouter API
  |
  v
Stored property, image, and amenity records
```

The UI never imports API internals. It talks to the backend through HTTP only.
The backend owns model selection, image storage, detection orchestration, and
database writes.

## Quick Start

### Prerequisites

- Docker Desktop or Docker Engine with Docker Compose v2.
- `uv` for local tests and checks.
- An OpenRouter API key with credits. Sign up at
  https://openrouter.ai/ and create a key under "Keys".

### Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
OPENROUTER_API_KEY=sk-or-v1-...your-key...
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

Default model is `openai/gpt-4o-mini` — cheapest reliable option. Switch to
`google/gemini-pro-1.5`, `meta-llama/llama-3.2-11b-vision-instruct`, or
`qwen/qwen2-vl-72b-instruct` from the dropdown. Live pricing:
https://openrouter.ai/models.

## API Overview

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/properties/` | Create an empty property shell |
| `POST` | `/api/v1/properties/{id}/images` | Upload and process one image |
| `POST` | `/api/v1/properties/{id}/describe` | Regenerate description from reviewed amenities |
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
    "model_name": "openai/gpt-4o-mini",
    "extra_info": "Two-bed flat near public transport"
  }'
```

Upload one image to the returned `property_id`:

```bash
curl -X POST http://localhost:8000/api/v1/properties/<property_id>/images \
  -F "model_name=openai/gpt-4o-mini" \
  -F "file=@/path/to/kitchen.jpg"
```

Generate a description from reviewed amenities:

```bash
curl -X POST http://localhost:8000/api/v1/properties/<property_id>/describe \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "openai/gpt-4o-mini",
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

## Testing The OpenRouter Flow

Use this checklist after a fresh `git pull` or after Phase 8 implementation
work. It covers static checks, container health, and the user-facing flow.

1. Static checks (host machine):

   ```bash
   uv sync --group dev
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy . --ignore-missing-imports
   uv run pytest tests/unit/ -v
   ```

2. Set the API key in `.env`:

   ```bash
   OPENROUTER_API_KEY=sk-or-v1-...your-key...
   ```

3. Boot the stack:

   ```bash
   docker compose up --build
   ```

4. Backend smoke checks (new terminal):

   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/api/v1/models/
   ```

   Expect four registered models in the second response: `openai/gpt-4o-mini`,
   `google/gemini-pro-1.5`, `meta-llama/llama-3.2-11b-vision-instruct`,
   `qwen/qwen2-vl-72b-instruct`.

5. End-to-end via the UI at http://localhost:7860:

   - Open `Upload & Detect`.
   - Pick `openai/gpt-4o-mini`.
   - Upload 2–3 property images (kitchen + living room is a good baseline).
   - Watch per-image progress fill in.
   - Confirm the per-room review panel renders confirm/edit/reject/add
     actions.
   - Click `Confirm & Generate Description` and read the generated text.
   - Open `Browse Properties` and confirm the new property appears.

6. Switch the dropdown to `meta-llama/llama-3.2-11b-vision-instruct` and
   repeat with one image. This exercises the non-JSON-mode parser path.

7. Tail logs while testing:

   ```bash
   docker compose logs -f api
   docker compose logs -f ui
   ```

If any step fails, file the symptom (HTTP status, log line, screenshot)
before reverting.

## Project Structure

```text
amenity_detector/
├── api/                         # FastAPI app, schemas, routers, middleware
├── core/                        # Detection, preprocessing, orchestration, CRUD
├── db/                          # SQLAlchemy models, sessions, migrations
├── models/                      # VLMClient interface + OpenRouter client
├── ui/                          # Gradio frontend and review-state helpers
├── monitoring/                  # Prometheus and Grafana provisioning
├── docker/                      # API and UI Dockerfiles
├── tests/                       # Unit and integration tests
├── docker-compose.yml           # Local full-stack orchestration
├── pyproject.toml               # Dependencies and tool config
└── .github/workflows/ci.yml     # CI checks
```

## Important Notes

- OpenRouter is paid. Set a spending cap in the OpenRouter dashboard before
  running long evaluations.
- The old batch upload endpoint was removed in Phase 8. The UI uses the
  property-shell plus per-image upload flow.
- `ARCHITECTURE.md`, `SPEC.md`, and internal design docs are intentionally
  ignored by git.
- Runtime uploads are stored under `storage/` and should not be committed.

## Roadmap

Completed:

- Repository cleanup with `uv`, ruff, mypy, pytest, and CI.
- VLM abstraction (now backed by a single OpenRouter client).
- PostgreSQL-backed FastAPI service.
- Docker Compose local stack.
- Gradio upload and browse UI.
- Per-image detection flow with incremental progress.
- Per-room amenity review UI.
- Structured logging, Prometheus metrics, and Grafana dashboards.
- Phase 8: OpenRouter migration and project cleanup (in progress).

Next likely work:

- Cloud deployment target and managed database/storage.
- Voice-assisted property search.
- Richer property search and ranking.
