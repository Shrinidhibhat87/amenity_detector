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
| Gradio UI | Done (legacy) | Upload, listing-details accordion, per-room amenity review, browse/search. Kept running until the TypeScript frontend reaches feature parity. |
| Next.js web frontend | In progress (Phase A) | TypeScript revamp of the UI. Foundation shipped: design tokens, primitives, Zod schemas, Vitest. |
| FastAPI API | Done | Property/image CRUD, search, description, and listing-metadata PATCH endpoints |
| PostgreSQL | Done | Stores properties, images, detected amenities, and Phase 9 listing metadata |
| Alembic migrations | Done | Schema versioning (`0001_phase8_baseline` → `0002_phase9_listing_metadata`) |
| VLM provider | Done | OpenRouter (OpenAI-compatible) backs every supported model |
| Image preprocessing | Done | Resizes images before inference to reduce VLM latency |
| VLM-generated alt text | Done | Same prompt that detects amenities also returns `alt_text` + `room_caption` |
| Observability | Done | Structured logging, request logs, Prometheus, Grafana dashboard |
| CI | Done | Ruff lint/format, mypy, and unit tests |

Recent work added the frontend revamp flow:

1. Start on Home and choose `Upload & Detect` or `Browse Properties`.
2. Create a property shell and upload/process one image at a time.
3. Detect room type and amenities from one structured JSON VLM response.
4. Set any of the 18 grouped amenity hints with segmented pills.
5. Review amenities grouped by room.
6. Confirm, edit, reject, or add amenities.
7. Reconcile expanded property hints against detections.
8. Generate the final description from the reviewed state.
9. Browse saved properties as thumbnail cards powered by the image endpoint.

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
- VLM-generated `alt_text` and `room_caption` per image (Phase 9), reused by
  Phase 12 listing pages and JSON-LD; fully editable via the image PATCH
  endpoint.
- Listing metadata on every new property: type (rent/sale), price + currency
  + period, bedrooms / bathrooms / area, property type, furnishing, locality,
  postal code, country code, available-from date, owner email.
- URL-safe `slug` auto-generated from `name + UUID prefix`, immutable for
  the lifetime of the property — used by the upcoming Phase 12 public
  routes.
- Image preprocessing with longest-edge resize to 768 px.
- Per-room review UI powered by `gr.render`.
- Row-level actions: confirm, edit, reject, add amenity.
- Explicit light/dark theme toggle persisted in the browser.
- Expanded property hints with 18 items across rooms, outdoor, access/storage,
  and features.
- Description regeneration from reviewed amenities and user hints.
- Browse/search UI for stored properties with lazy image thumbnails.
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
| Gradio UI (legacy) | http://localhost:7860 |
| Next.js web (Phase A) | http://localhost:3001 |
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
2. Click `Upload & Detect`.
3. Enter a property name.
4. Select a model.
5. Optionally open the **Listing details** accordion and fill in any subset
   of: listing type, price + currency + period, property type, furnishing,
   bedrooms, bathrooms, area, available-from date, locality, postal code,
   country code, owner email. Empty fields stay NULL on the row and can be
   patched later via `PATCH /api/v1/properties/{id}`.
6. Optionally fill property hints with the segmented pill controls.
7. Upload 2-3 representative property images.
8. Click `Upload & Detect Amenities`.
9. Review the detected amenities grouped by room.
10. Confirm, edit, reject, or add amenities as needed.
11. Click `Confirm & Generate Description`.
12. Use `Back`, then `Browse Properties`, to list, search, and inspect stored properties.

Default model is `openai/gpt-4o-mini` — cheapest reliable option. Switch to
`google/gemini-pro-1.5`, `meta-llama/llama-3.2-11b-vision-instruct`, or
`qwen/qwen2-vl-72b-instruct` from the dropdown. Live pricing:
https://openrouter.ai/models.

## API Overview

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/properties/` | Create an empty property shell with optional listing metadata |
| `PATCH` | `/api/v1/properties/{id}` | Update listing metadata after creation (slug is immutable) |
| `POST` | `/api/v1/properties/{id}/images` | Upload and process one image |
| `POST` | `/api/v1/properties/{id}/describe` | Regenerate description from reviewed amenities |
| `GET` | `/api/v1/properties/` | List properties |
| `GET` | `/api/v1/properties/{id}` | Get full property details |
| `GET` | `/api/v1/properties/search` | Search by amenity names |
| `GET` | `/api/v1/images/{id}` | Serve stored image bytes for thumbnails |
| `PATCH` | `/api/v1/images/{id}` | Update `alt_text`, `caption`, `is_primary`, or `display_order` on an image |
| `DELETE` | `/api/v1/properties/{id}` | Delete a property |
| `GET` | `/api/v1/models/` | List available models |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |

### Example: Per-Image Flow

Create a property (minimal — only `name` and `model_name` are required):

```bash
curl -X POST http://localhost:8000/api/v1/properties/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Frankfurt Apartment",
    "model_name": "openai/gpt-4o-mini",
    "extra_info": "Two-bed flat near public transport"
  }'
```

Or create one with full Phase 9 listing metadata:

```bash
curl -X POST http://localhost:8000/api/v1/properties/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Frankfurt 3BHK",
    "model_name": "openai/gpt-4o-mini",
    "listing_type": "rent",
    "price": 1500,
    "currency": "EUR",
    "price_period": "monthly",
    "num_bedrooms": 3,
    "num_bathrooms": 2,
    "area_sqm": 82.5,
    "property_type": "apartment",
    "furnishing": "semi_furnished",
    "available_from": "2026-06-01",
    "locality": "Sachsenhausen",
    "postal_code": "60594",
    "country_code": "DE",
    "owner_email": "owner@example.com"
  }'
```

The response includes an auto-generated `slug` derived from the name plus the
first six hex characters of the property UUID (e.g. `frankfurt-3bhk-d4e1f2`).
The slug is immutable for the lifetime of the property so public URLs stay
stable.

Update listing metadata after creation:

```bash
curl -X PATCH http://localhost:8000/api/v1/properties/<property_id> \
  -H "Content-Type: application/json" \
  -d '{"price": 1450, "furnishing": "furnished"}'
```

Only fields present in the body are written; sending `null` clears a field;
omitting a field leaves it untouched. The `slug` field is intentionally not
patchable (sending it returns HTTP 422).

Upload one image to the returned `property_id`:

```bash
curl -X POST http://localhost:8000/api/v1/properties/<property_id>/images \
  -F "model_name=openai/gpt-4o-mini" \
  -F "file=@/path/to/kitchen.jpg"
```

The image response now also carries `alt_text` (8–20 word description for the
HTML `alt` attribute, generated by the VLM), `caption` (short room caption,
also from the VLM), `is_primary` (hero flag, default `false`), and
`display_order` (gallery ordering, default `0`).

Edit any of those image fields:

```bash
curl -X PATCH http://localhost:8000/api/v1/images/<image_id> \
  -H "Content-Type: application/json" \
  -d '{"alt_text": "Custom alt text", "is_primary": true}'
```

Setting `is_primary: true` automatically clears the flag on every other image
of the same property — exactly one hero per property.

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

Run the Gradio UI locally (legacy frontend, kept until the TypeScript revamp reaches parity):

```bash
API_BASE_URL=http://localhost:8000 uv run python -m ui.app
```

### Web frontend (TypeScript, in progress)

The new frontend lives in `/web` and is a Next.js 15 App Router app written in
strict TypeScript. Phase A (Foundation) shipped:

- Next.js scaffold with Turbopack + Tailwind v4
- `tsconfig` strict + `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes`
- Design tokens lifted from `ui/docs/design-system.jsx` into Tailwind `@theme`
- Instrument Serif + Geist + JetBrains Mono via `next/font/google`
- Zod schemas mirroring `api/schemas.py` (single source of truth for shapes)
- Primitives (`Button`, `Chip`, `Card`, `Input`, `Select`) with Vitest tests
- `/kitchen-sink` showcase route (no-indexed)

Run it locally (Node 20+):

```bash
cd web
pnpm install
pnpm dev            # http://localhost:3000
pnpm test           # Vitest unit suite
pnpm typecheck      # tsc --noEmit
pnpm build          # standalone production server
```

Run it via Docker Compose alongside the rest of the stack:

```bash
docker compose up --build web
# Open http://localhost:3001
```

The Next.js container listens on port 3000 internally and is published on host
port 3001 (Grafana already owns 3000). The `CORS_ORIGINS` env var on the API
defaults to localhost:7860 (Gradio), localhost:3001 (web on host), and the
internal Docker DNS names `web:3000` and `ui:7860`. Override with a
comma-separated list when deploying to a real domain.

The Gradio UI stays in `/ui` and keeps working — it will be removed in a single
cleanup PR once the web frontend covers the upload, browse, detail, and search
paths end-to-end.

When changing UI code in Docker, rebuild the UI service:

```bash
docker compose up -d --build ui
```

Rebuilding only the API service does not update Gradio code because the UI runs
in a separate container.

## Database Migrations

Schema changes go through Alembic. Two migrations are checked in today:

- `0001_phase8_baseline` — captures the schema as it stood at the end of
  Phase 8 (`properties`, `images`, `detected_amenities`).
- `0002_phase9_listing_metadata` — adds the Phase 9 listing-metadata columns,
  the four image SEO/ordering columns, a unique index on `slug`, and the
  partial index `ix_detected_amenities_present_room_amenity` that powers the
  Phase 11 search hot path.

Apply on a fresh database:

```bash
docker compose exec api alembic upgrade head
```

> Inside the API container, `alembic`/`python`/`uvicorn` come from the runtime
> venv on `PATH`. The `uv` binary itself is **only** present in the multi-stage
> Dockerfile's builder stage and is not copied into the runtime image — call
> `alembic` directly, not `uv run alembic`. Use `uv run alembic …` only when
> running on the host machine.

Apply on an existing PostgreSQL container that already had the Phase 8 schema
created via `Base.metadata.create_all` (the live setup before Alembic was
introduced) — stamp the baseline first so Alembic does not try to recreate
the existing tables, then upgrade:

```bash
docker compose exec api alembic stamp 0001_phase8_baseline
docker compose exec api alembic upgrade head
```

This is non-destructive: existing rows survive with `NULL` in the new
property columns and the server defaults applied to `images.is_primary` and
`images.display_order`.

Roll back the most recent migration:

```bash
docker compose exec api alembic downgrade -1
```

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

Use this checklist after a fresh `git pull` or OpenRouter-related changes. It
covers static checks, container health, and the user-facing flow.

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
├── core/                        # Detection, preprocessing, orchestration, CRUD, slug helper
├── db/                          # SQLAlchemy models, sessions, Alembic migrations
│   └── migrations/versions/     # 0001_phase8_baseline.py, 0002_phase9_listing_metadata.py
├── models/                      # VLMClient interface + OpenRouter client
├── ui/                          # Gradio frontend (legacy — to be removed after web parity)
├── web/                         # Next.js 15 + TypeScript frontend (Phase A foundation)
├── monitoring/                  # Prometheus and Grafana provisioning
├── docker/                      # API, UI, and web Dockerfiles
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
- OpenRouter migration and project cleanup.
- **Phase 9 — listing-metadata expansion**: 17 nullable property columns
  (price, rooms, area, locality, slug, …) + 4 image columns (alt_text,
  caption, is_primary, display_order) + Alembic baseline and Phase 9
  migrations + PATCH endpoints for property and image + UI listing-details
  accordion + VLM-generated `alt_text` and `room_caption`.

Next likely work:

- **TypeScript frontend revamp** (in progress — Phase A foundation shipped).
  Subsequent sub-phases: P-B browse + SEO (Server Components + JSON-LD +
  `sitemap.ts` + `llms.txt`), P-C wizard (4-step detect flow, Zustand+persist),
  P-D natural-language search, P-E cutover (Gradio UI removed in one PR).
- Phase 10 — locality enrichment via OpenStreetMap Overpass + Nominatim
  (auto-populated POI summary per property, cached 30 days).
- Phase 11 — hybrid NL search: pgvector + Postgres full-text, LLM-driven
  filter extraction over the Phase 9 metadata + per-room amenity tuples.
- Phase 12 — SEO surface: moved from FastAPI Jinja routes to Next.js
  Server Components. Per-listing HTML, JSON-LD `Accommodation`, `sitemap.ts`,
  `robots.ts`, `llms.txt`, JSONL feed for AI agents — all rendered by the
  `/web` app, with the API staying a thin REST layer.
- Cloud deployment target and managed database/storage.
- Voice-assisted property search.
