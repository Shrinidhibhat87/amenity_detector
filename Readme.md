# Amenity Detection and Description System

An end-to-end pipeline that automatically identifies amenities in property images and generates natural language descriptions. Given one or more images of a property (e.g. an Airbnb listing), the system detects what's present in each room, stores the results, and produces a readable property summary.

---

## What it does

1. **Detects amenities** in a property image using a vision-language model (VLM)
2. **Structures results** by room type (kitchen, bedroom, bathroom, etc.)
3. **Stores everything** in both SQLite (queryable) and CSV (portable) formats
4. **Generates a description** of the property based on what was detected
5. **Exposes results** via a Streamlit web app and a FastAPI REST API

---

## Amenity Schema

Amenities are organised by room type. The full list lives in `core/amenity_schema.py` (`AMENITY_SCHEMA`). The current categories are:

| Room | Example amenities |
|---|---|
| Kitchen | refrigerator, oven, microwave, dishwasher, coffee_maker |
| Living Room | sofa, tv, fireplace, projector, gaming_console |
| Bedroom | bed, wardrobe, desk, air_conditioner, lamp |
| Bathroom | toilet, shower, bathtub, hair_dryer, washing_machine |
| Outdoor | patio, pool, garden, bbq_grill, parking_space |
| Common | wifi, heating, smoke_detector, elevator |

---

## Architecture

```
main.py  ──▶  PropertyAmenitySystem
                  │
                  ├── AmenityDetector  (VLM calls + response parsing)
                  │       └── LlavaModel  (currently: LLaVA 1.5-7B via HuggingFace)
                  │
                  └── AmenityDataManager  (SQLite + CSV storage)

FastAPI (api/)  ──▶  PropertyAmenitySystem  (same pipeline, HTTP interface)
Streamlit (streamlit/)  ──▶  direct inference or via FastAPI
```

> **Planned (Phase 1)**: LlavaModel will be replaced by a pluggable `VLMClient` abstraction
> supporting Qwen2.5-VL-7B, LLaMA 3.2 Vision, and Gemini 2.0 Flash interchangeably via Ollama
> or the Google Generative AI API. See `SPEC.md` for the full roadmap.

---

## Getting Started

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) — fast Python package manager
- A CUDA-capable GPU is strongly recommended (LLaVA is slow on CPU)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/Shrinidhibhat87/amenity_detector.git
cd amenity_detector

# 2. Install uv if you don't have it
curl -Ls https://astral.sh/uv/install.sh | sh

# 3. Install runtime dependencies
uv sync

# 4. (Optional) Install dev tools — linting, type checking, tests
uv sync --group dev
```

### Configuration

Copy the environment template and fill in any values you need:

```bash
cp .env.example .env
```

Edit `config/config.yaml` to set your image input path and output directory before running.

### Running

**Command-line pipeline:**
```bash
uv run python main.py
```

**Streamlit web app:**
```bash
uv run streamlit run streamlit/app.py
```

**FastAPI server:**
```bash
uv run uvicorn api.service:app --reload
```

---

## Development

### Running checks locally

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy . --ignore-missing-imports

# Unit tests
uv run pytest tests/unit/ -v
```

### Pre-commit hooks

Install once to run checks automatically before every commit:

```bash
uv run pre-commit install
```

### CI

GitHub Actions runs on every push and pull request to `main`:
- `ruff check` — linting
- `ruff format --check` — formatting
- `mypy` — static type checking
- `pytest tests/unit/` — unit tests

Heavy dependencies (torch, transformers) are excluded from CI; only dev packages are installed in the runner.

---

## Data Storage

Results are written to the directory configured in `config/config.yaml → output.directory`:

| File | Format | Purpose |
|---|---|---|
| `amenities.csv` | CSV | One row per image; one column per amenity (0/1) |
| `amenities.db` | SQLite | Relational store for complex queries |

---

## Current Limitations

- **Inference speed**: ~20 seconds per image even with GPU (LLaVA 7B at 4-bit quantisation)
- **Implicit amenities**: Cannot detect things that aren't visible — e.g. WiFi, heating
- **Single model**: Only LLaVA is wired up today; the VLM abstraction layer is coming in Phase 1
- **No cloud storage**: Images and results are local only; cloud migration is planned for Phase 5

---

## Roadmap

See [`SPEC.md`](SPEC.md) for the detailed phased plan. In brief:

| Phase | Goal |
|---|---|
| 0 ✅ | Cleanup: uv migration, pre-commit, CI, type safety |
| 1 | VLM abstraction: Ollama (Qwen2.5-VL, LLaMA 3.2 Vision) + Gemini 2.0 Flash |
| 2 | PostgreSQL + FastAPI backend with proper REST endpoints |
| 3 | Gradio frontend replacing Streamlit |
| 4 | Observability: structured logging, Prometheus metrics |
| 5 | Cloud migration: managed DB, object storage, container deployment |
