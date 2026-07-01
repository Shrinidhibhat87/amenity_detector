# API Layer (FastAPI)

The `api/` package is the HTTP boundary. It owns routing, request/response
schemas, middleware, and dependency wiring — and nothing else. All business
logic lives in [`core/`](../core/README.md); all persistence lives in
[`db/`](../db/README.md). Routers stay thin: parse → delegate → serialise.

## Modules

| File | Responsibility |
|------|----------------|
| `main.py` | App factory + lifespan. On startup builds `ModelRegistry`, `SearchPipeline`, `QueryParser`, DB tables; attaches Prometheus instrumentation; stores singletons on `app.state`. Mounts routers, CORS, request-logging middleware. Serves `/health` and `/metrics`. |
| `routers/properties.py` | Property CRUD, per-image detection, describe, listing search. |
| `routers/images.py` | Serve stored image bytes; PATCH image metadata (`alt_text`, `caption`, `is_primary`, `display_order`). |
| `routers/search.py` | Hybrid NL search endpoint → delegates to `SearchPipeline`. |
| `routers/locality.py` | Locality preview + persist, blocking and SSE-streamed. |
| `routers/models.py` | List available VLMs from the registry. |
| `schemas.py` | Pydantic request/response models. `extra="forbid"` + `model_dump(exclude_unset=True)` on PATCH so "omit" ≠ "clear". Mirrored by `web/lib/schemas.ts`. |
| `middleware.py` | `RequestLoggingMiddleware` — structured per-request logs. |
| `dependencies.py` | `Depends()` providers: DB session, system, pipeline, registry. |
| `logging_config.py` | Compatibility shim over `core.logging_config`. |

## Component wiring

```mermaid
graph TD
    Client["Client (web / curl / crawler)"]

    subgraph app["FastAPI app (main.py)"]
        MW["CORS + RequestLoggingMiddleware"]
        subgraph routers["routers/"]
            P["properties"]
            I["images"]
            S["search"]
            L["locality"]
            M["models"]
        end
        Health["/health · /metrics"]
    end

    subgraph state["app.state singletons (lifespan)"]
        REG["ModelRegistry"]
        PIPE["SearchPipeline"]
        QP["QueryParser"]
    end

    subgraph deps["dependencies.py (Depends)"]
        DBS["DB Session"]
        SYS["PropertyAmenitySystem"]
    end

    Client --> MW --> routers
    P --> SYS
    I --> DBS
    S --> PIPE
    S --> QP
    L --> DBS
    M --> REG
    SYS --> DBS
    SYS --> REG
    PIPE --> DBS

    SYS -.core.-> core["core/"]
    PIPE -.core.-> core
    DBS -.db.-> db[("db/")]
```

## Request lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant MW as Middleware
    participant R as Router
    participant D as Depends
    participant Core as core/
    participant DB as db/

    C->>MW: HTTP request
    MW->>MW: log start, attach request id
    MW->>R: dispatch
    R->>R: validate body (Pydantic, extra=forbid → 422)
    R->>D: resolve Session / System / Pipeline
    R->>Core: delegate business logic
    Core->>DB: read/write via Session
    DB-->>Core: rows
    Core-->>R: domain result
    R->>R: serialise to response_model
    R-->>MW: response
    MW->>MW: log status + latency
    MW-->>C: HTTP response
```

## Endpoints

| Method | Path | Router | Delegates to |
|--------|------|--------|--------------|
| `POST` | `/api/v1/properties/` | properties | `create_property_shell` |
| `PATCH` | `/api/v1/properties/{id}` | properties | `AmenityDataManager.update_property` |
| `POST` | `/api/v1/properties/{id}/images` | properties | `process_one_image` |
| `POST` | `/api/v1/properties/{id}/describe` | properties | `generate_description_from_amenities` |
| `GET` | `/api/v1/properties/` · `/{id}` · `/by-slug/{slug}` · `/search` | properties | `AmenityDataManager` reads |
| `DELETE` | `/api/v1/properties/{id}` | properties | `AmenityDataManager.delete_property` |
| `POST` | `/api/v1/search` | search | `SearchPipeline.search` |
| `POST` | `/api/v1/locality` · `/properties/{id}/locality` (+ `/stream`) | locality | `LocalityAgent` |
| `GET` | `/api/v1/images/{id}` · `PATCH` | images | file serve · `update_image` |
| `GET` | `/api/v1/models/` | models | `ModelRegistry.available_models` |
| `GET` | `/health` · `/metrics` | main | DB ping · Prometheus |

### Contract invariants

- Both PATCH endpoints use `model_dump(exclude_unset=True)`: omit a key to leave
  it untouched, send `null` to clear it.
- `extra="forbid"` returns HTTP 422 on unknown keys — this is what blocks
  attempts to mutate the immutable `slug`.
- Locality streaming endpoints emit Server-Sent Events (one frame per category,
  then a final result frame); upstream OSM failures become a clean SSE `error`
  frame with CORS headers, never a masked 500.
