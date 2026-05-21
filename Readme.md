# Amenity Detector

Amenity Detector is an end-to-end AI engineering project for analysing property
photos. It detects visible amenities room by room, lets a user review and
correct the model output, stores the result in PostgreSQL, generates a
listing-style description, and exposes the listings for natural-language
search and crawler/agent consumption.

The project is intentionally built like a small production system: FastAPI
backend, Next.js (TypeScript) frontend, PostgreSQL with `pgvector`,
multi-model VLM access through OpenRouter, Docker Compose orchestration,
Alembic migrations, structured logs, Prometheus metrics, Grafana dashboards,
and a GitHub Actions CI gate.

## Architecture

```text
Browser
  │
  │  http://localhost:3000
  ▼
Next.js 15 (App Router, TS) ──── server-rendered HTML, JSON-LD,
  │                              /sitemap.xml, /robots.txt,
  │                              /llms.txt, /api/feed.jsonl
  │  REST over HTTP
  ▼
FastAPI                       ──── /api/v1/properties, /images,
  │             ╲                    /search, /describe, /metrics
  │ SQLAlchemy   ╲ VLMClient
  ▼               ▼
PostgreSQL       OpenRouter (4 registered VLMs)
+ pgvector
+ FTS
```

The frontend never imports backend internals. It speaks HTTP only. The backend
owns model selection, image storage, detection orchestration, embedding
indexing, and database writes.

## Quick Start

### Prerequisites

- Docker Desktop or Docker Engine with Docker Compose v2.
- `uv` for Python tests and checks.
- pnpm via corepack and Node 24 for `/web` local development.
- An OpenRouter API key with credits — sign up at https://openrouter.ai/.
- An embeddings endpoint (OpenAI-compatible) for hybrid search. The defaults
  in `.env.example` target a LiteLLM proxy with `text-embedding-3-small`.

### Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```bash
OPENROUTER_API_KEY=sk-or-v1-...your-key...
EMBEDDINGS_BASE_URL=https://your-proxy/v1
EMBEDDINGS_API_KEY=sk-...
EMBEDDINGS_MODEL=text-embedding-3-small
```

PostgreSQL defaults in `.env.example` match `docker-compose.yml`.

### Start the Stack

```bash
docker compose up --build
```

Open:

| Service        | URL                              |
|----------------|----------------------------------|
| Next.js web    | http://localhost:3000            |
| FastAPI docs   | http://localhost:8000/docs       |
| Prometheus     | http://localhost:9090            |
| Grafana        | http://localhost:3030            |

Grafana default login (change on first login):

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

1. Open http://localhost:3000.
2. Click **Upload & Detect** to enter the wizard:
   - `/detect/config` — name, model, listing metadata (price, bedrooms,
     area, locality, …). All listing fields are optional and can be
     patched later via `PATCH /api/v1/properties/{id}`.
   - `/detect/upload` — upload property images; each is processed and
     returns detected amenities + a VLM-written `alt_text` and
     `room_caption`.
   - `/detect/review` — confirm, edit, reject, or add amenities grouped
     by room.
   - `/detect/describe` — review and edit the generated description.
3. Click **Browse Properties** to list saved properties, or **Search** to
   query in natural language (e.g. *3BHK rent under 1500 EUR with fireplace
   in living room*).

Default VLM is `openai/gpt-4o-mini`. Switch to `google/gemini-pro-1.5`,
`meta-llama/llama-3.2-11b-vision-instruct`, or `qwen/qwen2-vl-72b-instruct`
from the dropdown. Live pricing: https://openrouter.ai/models.

## API Overview

| Method   | Path                                       | Purpose                                              |
|----------|--------------------------------------------|------------------------------------------------------|
| `POST`   | `/api/v1/properties/`                      | Create a property shell with optional listing metadata |
| `PATCH`  | `/api/v1/properties/{id}`                  | Update listing metadata or description (slug is immutable) |
| `POST`   | `/api/v1/properties/{id}/images`           | Upload and process one image                          |
| `POST`   | `/api/v1/properties/{id}/describe`         | Regenerate description from reviewed amenities        |
| `GET`    | `/api/v1/properties/`                      | List properties                                       |
| `GET`    | `/api/v1/properties/{id}`                  | Get full property details                             |
| `POST`   | `/api/v1/search`                           | Hybrid NL search (FTS + pgvector rerank)              |
| `GET`    | `/api/v1/images/{id}`                      | Serve stored image bytes                              |
| `PATCH`  | `/api/v1/images/{id}`                      | Update alt_text, caption, is_primary, display_order   |
| `DELETE` | `/api/v1/properties/{id}`                  | Delete a property                                     |
| `GET`    | `/api/v1/models/`                          | List available VLMs                                   |
| `GET`    | `/health`                                  | Health check                                          |
| `GET`    | `/metrics`                                 | Prometheus metrics                                    |

SEO + agent endpoints are served by the Next.js app at `/sitemap.xml`,
`/robots.txt`, `/llms.txt`, and `/api/feed.jsonl`.

### Example

Create a property and patch metadata:

```bash
curl -X POST http://localhost:8000/api/v1/properties/ \
  -H "Content-Type: application/json" \
  -d '{"name": "Frankfurt Apartment", "model_name": "openai/gpt-4o-mini"}'

curl -X PATCH http://localhost:8000/api/v1/properties/<property_id> \
  -H "Content-Type: application/json" \
  -d '{"price": 1450, "furnishing": "furnished", "num_bedrooms": 3}'
```

The create response includes an auto-generated immutable `slug` (e.g.
`frankfurt-apartment-d4e1f2`) used by the public Next.js listing pages.

## Local Development

### Backend

```bash
uv sync --group dev
uv run uvicorn api.main:app --reload
```

### Frontend

```bash
cd web
pnpm install
pnpm dev          # http://localhost:3000
pnpm test         # Vitest unit suite
pnpm typecheck    # tsc --noEmit
pnpm build        # standalone production server
```

The Next.js app is strict TypeScript with `noUncheckedIndexedAccess` and
`exactOptionalPropertyTypes` enabled. Zod schemas in `web/lib/schemas.ts`
mirror `api/schemas.py` — any drift means the API contract changed.

## Database Migrations

Schema is versioned with Alembic. Apply on a fresh database:

```bash
docker compose exec api alembic upgrade head
```

Apply on a pre-Alembic PostgreSQL container (Phase 8 schema already created
via `Base.metadata.create_all`):

```bash
docker compose exec api alembic stamp 0001_phase8_baseline
docker compose exec api alembic upgrade head
```

Roll back the latest migration:

```bash
docker compose exec api alembic downgrade -1
```

> Inside the API container, call `alembic` directly — the `uv` binary is only
> in the multi-stage builder image, not in the runtime image. Use
> `uv run alembic …` only on the host.

## Quality Checks

The GitHub Actions CI runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy . --ignore-missing-imports
uv run pytest tests/unit/ -v
```

For broader local verification, including integration tests:

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
├── core/                        # Detection, preprocessing, search pipeline, CRUD, slug helper
├── db/                          # SQLAlchemy models, sessions, Alembic migrations
├── models/                      # VLMClient interface + OpenRouter client
├── web/                         # Next.js 15 + TypeScript frontend (App Router)
├── monitoring/                  # Prometheus and Grafana provisioning
├── docker/                      # API and web Dockerfiles
├── tests/                       # Unit and integration tests
├── docker-compose.yml           # Local full-stack orchestration
├── pyproject.toml               # Python dependencies and tool config
└── .github/workflows/ci.yml     # CI checks
```

## Notes

- OpenRouter is paid. Set a spending cap in the OpenRouter dashboard before
  long evaluations.
- Runtime uploads are stored under `storage/` and are not committed.
- `ARCHITECTURE.md`, `SPEC.md`, and internal handbook files are gitignored.
