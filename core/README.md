# Core Logic

`core/` holds every non-HTTP, non-persistence decision: image preprocessing,
VLM detection, description generation, CRUD orchestration, hybrid search, and
locality enrichment. It is imported by [`api/`](../api/README.md) and writes
through [`db/`](../db/README.md). It never imports FastAPI.

## Modules

| File / package | Responsibility |
|----------------|----------------|
| `amenity_system.py` | `PropertyAmenitySystem` — top-level orchestrator. Wires detector + data manager + registry. Owns `create_property_shell`, `process_one_image`, `generate_description_from_amenities`. |
| `amenity_detector.py` | `AmenityDetector` — builds prompts, calls the VLM, parses the Phase 9 JSON (room_type, per-amenity present/confidence, alt_text, caption) with legacy-flat fallback; returns `DetectionResult`. |
| `amenity_data_manager.py` | `AmenityDataManager` — all SQLAlchemy reads/writes for properties, images, amenities. Enforces the single-hero-image invariant. |
| `preprocessing.py` | `preprocess_image` — resize longest edge to ≤768px before inference. |
| `amenity_schema.py` | Canonical amenity catalogue fed into the detection prompt. |
| `slug.py` | `make_slug(name, id)` — immutable, URL-safe public slug. |
| `embeddings.py` | `EmbeddingsClient` — OpenAI-compatible embedding calls for search indexing/query. |
| `search/` | Hybrid NL search pipeline (see below). |
| `locality/` | Neighbourhood enrichment — has its own [README](locality/README.md). |
| `logging_config.py` | `setup_logging()` — shared structured logging for api + web. |

## Subsystem map

```mermaid
graph TD
    API["api/ routers"]

    subgraph orch["Detection & description"]
        SYS["PropertyAmenitySystem<br/>amenity_system.py"]
        DET["AmenityDetector<br/>amenity_detector.py"]
        PRE["preprocess_image<br/>preprocessing.py"]
        SCH["amenity_schema.py"]
        DM["AmenityDataManager<br/>amenity_data_manager.py"]
        SLUG["slug.py"]
    end

    subgraph search["search/ (hybrid)"]
        PARSER["QueryParser<br/>parser.py"]
        FILTER["SearchFilter<br/>filter.py"]
        SQL["build_candidate_query<br/>sql.py"]
        SCORE["score_candidate<br/>score.py"]
        PIPE["SearchPipeline<br/>pipeline.py"]
    end

    subgraph loc["locality/"]
        LAGENT["LocalityAgent<br/>agent.py"]
        note["geocode · overpass · gather · cache · service"]
    end

    EMB["EmbeddingsClient<br/>embeddings.py"]
    REG["ModelRegistry → VLMClient<br/>(models/)"]
    DB[("db/")]

    API --> SYS
    API --> PIPE
    API --> LAGENT

    SYS --> DET
    SYS --> DM
    SYS --> SLUG
    DET --> PRE
    DET --> SCH
    DET --> REG
    DM --> DB

    PIPE --> PARSER
    PARSER --> FILTER
    PIPE --> SQL
    SQL --> FILTER
    PIPE --> SCORE
    PIPE --> EMB
    PIPE --> DB

    LAGENT --> note
    LAGENT --> DB
```

## Detection pipeline

```mermaid
flowchart LR
    IMG["Image bytes"] --> PRE["preprocess<br/>resize ≤768px"]
    PRE --> PROMPT["build_detection_prompt<br/>(amenity_schema)"]
    PROMPT --> VLM["VLMClient → OpenRouter"]
    VLM --> PARSE{"parse JSON"}
    PARSE -->|Phase 9 shape| FULL["room_type + amenities<br/>+ alt_text + caption"]
    PARSE -->|legacy flat| FLAT["flat amenities only<br/>(alt_text/caption = None)"]
    FULL --> RES["DetectionResult"]
    FLAT --> RES
    RES --> DB[("INSERT image +<br/>detected_amenities")]
```

`DetectionResult` carries `amenities_by_room`, `flat_amenities`,
`flat_confidences`, `alt_text`, and `room_caption`. `alt_text`/`caption` are
trimmed to 500 chars at a word boundary so the columns never overflow.

## Hybrid search pipeline

Two-stage: cheap FTS/SQL recall, then embedding rerank.

```mermaid
flowchart TD
    Q["NL query"] --> P["QueryParser<br/>(LLM → SearchFilter,<br/>regex fallback)"]
    P --> F["SearchFilter<br/>price/beds/type/amenities"]
    F --> SQL["build_candidate_query<br/>scalar + amenity EXISTS clauses"]
    SQL --> CAND["Candidate rows (recall)"]
    Q --> QE["EmbeddingsClient.embed_text"]
    CAND --> FTS["FTS rank"]
    QE --> SC["score_candidate<br/>cosine(desc_embedding, query)"]
    FTS --> SC
    SC --> RANK["ranked results"]
```

`reindex_property` / `build_index_text` keep `description_embedding` and the FTS
document fresh after writes. On SQLite (tests) the vector path degrades
gracefully; pgvector is used in Postgres.

## Locality enrichment

Deterministic single-radius POI counts from OSM, plus an LLM blurb written from
the exact numbers. Full internals — module table, data flow, transit
classification, caching, and $0/compliance design — live in
[`core/locality/README.md`](locality/README.md).

```mermaid
flowchart LR
    IN["PIN + street? + country + radius"] --> GEO["geocode.py<br/>Nominatim (cached)"]
    GEO --> GATHER["gather.py<br/>count each category @ one radius"]
    GATHER --> OVER["overpass.py<br/>true count + nearest sample<br/>+ transit breakdown"]
    OVER --> AGENT["agent.py<br/>LLM blurb from exact counts"]
    AGENT --> RESULT["LocalityResult"]
    RESULT --> SVC["service.py<br/>UPSERT locality_insights"]
```
