# Amenity Detector

An end-to-end pipeline that automatically identifies amenities in property images (e.g. Airbnb listings), stores results in PostgreSQL, and exposes a REST API. Built as a learning project for real-world AI Engineering practices.

**Status**: Phases 0–2 complete. FastAPI backend + PostgreSQL + VLM abstraction (Ollama / Gemini) running via Docker Compose. Gradio UI (Phase 3) is next.

---

## What it does

1. Accept one or more property images via a REST API upload
2. Run amenity detection using a Vision-Language Model (Qwen2.5-VL-7B, LLaMA 3.2 Vision, or Gemini 2.0 Flash)
3. Detect which amenities are visible per image (refrigerator, pool, sofa, etc.)
4. Store property, image, and detection records in PostgreSQL
5. Generate a natural-language description of the property
6. Allow browsing and searching properties by detected amenities

---

## Amenity Schema

Amenities are organised by room type and defined in `core/amenity_schema.py`:

| Room | Example amenities |
|---|---|
| Kitchen | refrigerator, oven, microwave, dishwasher, coffee_maker |
| Living Room | sofa, tv, fireplace, projector, gaming_console |
| Bedroom | bed, wardrobe, desk, air_conditioner, lamp |
| Bathroom | toilet, shower, bathtub, hair_dryer, washing_machine |
| Outdoor | patio, pool, garden, bbq_grill, parking_space |
| Common | wifi, heating, smoke_detector, elevator |

---

## Quick Start (Docker Compose)

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose v2)
- [uv](https://github.com/astral-sh/uv) — for running tests and local dev outside Docker
- At least one of:
  - **Gemini API key** (free, 1500 req/day) — easiest to get started
  - **Ollama** installed locally — for local GPU inference

### 1. Get a Gemini API key (free)

1. Go to [https://aistudio.google.com/](https://aistudio.google.com/)
2. Click **Get API key** → **Create API key**
3. Copy the key — you'll need it in step 3

### 2. (Optional) Install Ollama for local GPU inference

If you want to run Qwen2.5-VL-7B or LLaMA 3.2 Vision locally on your GPU:

```bash
# Install Ollama (Linux / WSL2)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the models (choose one or both)
# Qwen2.5-VL-7B: ~5GB download, fits in 6GB VRAM with 4-bit quantisation
ollama pull qwen2.5vl:7b

# LLaMA 3.2 Vision 11B: ~6.5GB — slightly over 6GB VRAM, Ollama offloads to CPU (slower)
ollama pull llama3.2-vision:11b

# Start the Ollama server (runs on http://localhost:11434)
ollama serve
```

> **Note on GPU**: The models above work on a GTX 1660 Ti (6GB VRAM). Qwen2.5-VL-7B fits
> comfortably. LLaMA 3.2 Vision 11B may be slow due to partial CPU offloading — use it to
> experience the VRAM vs speed trade-off firsthand.

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```bash
# Required for Gemini (leave empty if using Ollama only)
GEMINI_API_KEY=your_key_here

# If Ollama is running natively on your machine (outside Docker):
OLLAMA_BASE_URL=http://host.docker.internal:11434

# Database — the defaults below match docker-compose.yml, no change needed
DATABASE_URL=postgresql://amenity_user:amenity_pass@localhost:5432/amenity_db
POSTGRES_USER=amenity_user
POSTGRES_PASSWORD=amenity_pass
POSTGRES_DB=amenity_db

# Where uploaded images are stored (Docker mounts this as a volume)
IMAGE_STORAGE_DIR=./storage/images
```

### 4. Start the stack

```bash
docker compose up
```

This starts:
- **PostgreSQL 16** on port `5432` (data persists in a Docker volume)
- **FastAPI backend** on port `8000`

First run takes a minute to build the Docker image. Subsequent starts are instant.

### 5. Verify it's working

Open the interactive API docs in your browser:

```
http://localhost:8000/docs
```

Or run a quick health check:

```bash
curl http://localhost:8000/health
# {"status":"ok","database":"ok","version":"1.0.0"}
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/properties/upload` | Upload images, detect amenities, store results |
| `GET` | `/api/v1/properties/` | List all properties (paginated) |
| `GET` | `/api/v1/properties/{id}` | Full property details (images + amenities) |
| `GET` | `/api/v1/properties/search?amenities=pool,wifi` | Filter by required amenities |
| `DELETE` | `/api/v1/properties/{id}` | Remove a property |
| `GET` | `/api/v1/models/` | List available VLMs and their status |
| `GET` | `/health` | Liveness check |

### Example: upload a property

```bash
curl -X POST http://localhost:8000/api/v1/properties/upload \
  -F "name=Frankfurt House 1" \
  -F "model_name=gemini-2.0-flash" \
  -F "extra_info=2-bed flat near the city centre" \
  -F "files=@/path/to/kitchen.jpg" \
  -F "files=@/path/to/bedroom.jpg"
```

---

## Development

### Install dependencies locally (for tests and linting)

```bash
uv sync --group dev
```

### Run tests

```bash
# Unit tests only (fast, no Docker needed)
uv run pytest tests/unit/ -v

# Integration tests (uses in-memory SQLite, no Docker needed)
uv run pytest tests/integration/ -v

# Full suite
uv run pytest tests/ -v
```

To run integration tests against a real PostgreSQL (must be running):

```bash
TEST_DATABASE_URL=postgresql://amenity_user:amenity_pass@localhost:5432/amenity_db \
  uv run pytest tests/integration/ -v
```

### Run checks

```bash
uv run ruff check .          # linting
uv run ruff format .         # formatting
uv run mypy . --ignore-missing-imports  # type checking
```

### Pre-commit hooks

```bash
uv run pre-commit install    # run once; ruff + mypy run before every commit after this
```

### Database migrations

```bash
# After changing db/models.py, generate a new migration:
uv run alembic revision --autogenerate -m "describe your change"

# Apply all pending migrations (also runs automatically in the Docker container):
uv run alembic upgrade head
```

---

## Project Structure

```
amenity_detector/
├── api/                    # FastAPI app
│   ├── main.py             # App entrypoint, lifespan, router registration
│   ├── dependencies.py     # Dependency injection (DB session, model registry)
│   ├── schemas.py          # Pydantic request/response models
│   └── routers/
│       ├── properties.py   # /api/v1/properties/* endpoints
│       └── models.py       # /api/v1/models/ endpoint
│
├── core/                   # Business logic
│   ├── amenity_schema.py   # Amenity definitions by room type
│   ├── amenity_detector.py # Prompt engineering + VLM response parsing
│   ├── amenity_data_manager.py  # SQLAlchemy CRUD operations
│   └── amenity_system.py   # Upload pipeline orchestrator
│
├── models/                 # VLM abstraction layer
│   ├── base.py             # VLMClient ABC + VLMResponse dataclass
│   ├── ollama_client.py    # Ollama REST API client
│   ├── gemini_client.py    # Google Gemini API client
│   └── registry.py         # ModelRegistry (name → client mapping)
│
├── db/                     # Database layer
│   ├── models.py           # SQLAlchemy ORM: Property, PropertyImage, DetectedAmenity
│   ├── session.py          # Engine, SessionLocal, get_db()
│   └── migrations/         # Alembic migration scripts
│
├── tests/
│   ├── unit/               # Fast tests, no external dependencies
│   └── integration/        # API + DB tests (in-memory SQLite)
│
├── docker/
│   ├── Dockerfile.api      # FastAPI container
│   └── Dockerfile.ui       # Gradio container (Phase 3)
│
├── docker-compose.yml      # PostgreSQL + API services
├── pyproject.toml          # uv dependencies + tool config
└── .env.example            # Environment variable template
```

---

## Roadmap

| Phase | Status | Goal |
|---|---|---|
| 0 | ✅ Done | Cleanup: uv, pre-commit, CI, type safety |
| 1 | ✅ Done | VLM abstraction: OllamaClient, GeminiClient, ModelRegistry |
| 2 | ✅ Done | PostgreSQL + FastAPI REST API + Docker Compose |
| 3 | 🔜 Next | Gradio frontend: Upload tab + Browse tab |
| 4 | Planned | Observability: structured logging, Prometheus metrics |
| 5 | Planned | Cloud migration: managed DB, object storage, CD pipeline |

---

## CI

GitHub Actions runs on every push / pull request to `main`:
- `ruff check` — linting
- `ruff format --check` — formatting
- `mypy` — static type checking
- `pytest tests/unit/ tests/integration/` — all tests (no GPU or Docker needed)
