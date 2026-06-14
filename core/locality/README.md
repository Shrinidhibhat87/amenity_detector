# Locality Enrichment

This package fills the **"Lage"** (neighbourhood) section of a listing. Given a
PIN code (plus an optional street and a country), it returns the schools, gyms,
supermarkets, parks, transit stops, pharmacies and airports nearby — with honest
counts, a per-mode transit breakdown, and a short written blurb — using only the
free OpenStreetMap APIs and a small LLM call.

It is built like a real product surface: rate-limited and compliant OSM clients,
Postgres-backed caching ($0 to run), deterministic counts, per-category failure
isolation, and a streaming endpoint for progress feedback.

## Modules

| File | Responsibility |
|------|----------------|
| `geocode.py` | Nominatim client: PIN (+ street, country) → coordinate + bounding box. Structured query (`postalcode`/`street`/`countrycodes`), 1 req/s, mandatory User-Agent, retry/backoff, pluggable cache. |
| `overpass.py` | Overpass client: coordinate + radius + category → distance-sorted POIs. Owns the category→OSM-tag selectors, the transit sub-type classifier, and `gather()` (true count + nearest sample + transit breakdown). |
| `gather.py` | Deterministic orchestration: geocode once, then count every category at **one** radius (airports at a fixed wide radius). `iter_gather` streams one progress event per category; `gather_locality` drains it for non-streaming callers. Isolates per-category failures. |
| `agent.py` | Turns the gathered facts into prose via one constrained OpenRouter call, with a deterministic fallback. `run()` (blocking) and `run_streaming()` (yields progress + final result). Produces the `LocalityResult`. |
| `cache.py` | Postgres adapters for the geocode + Overpass caches (POIs/geocodes don't move, so a stored row means zero future network cost). |
| `service.py` | Builds an agent wired to the request's DB-backed caches, and upserts a `LocalityResult` onto a property's single `locality_insight` row. |

## Data flow

```text
PIN + street? + country + radius
        │
        ▼  geocode.py (Nominatim, structured, cached)
   centre lat/lon
        │
        ▼  gather.py  ── one radius for everyday categories ──┐
   per category, in order:                                    │  iter_gather yields
     supermarket, school, gym, park, transit, pharmacy        │  GatherProgress per
     + airport (fixed ~50 km radius)                          │  category (→ SSE)
        │  overpass.py gather(): true count + nearest 5       │
        │  + transit subtype breakdown                        │
        ▼                                                     ▼
   GatheredLocality (counts, samples, transit_breakdown)
        │
        ▼  agent.py: one LLM call writes the blurb from those exact numbers
   LocalityResult  ──►  API response / persisted on the property
```

## Key design decisions

- **Deterministic single-radius counts.** Every everyday category is measured
  once at the user's chosen radius. The agent no longer picks or widens a radius
  per category, so counts can't drift across radii, and the prose is written from
  the same numbers the UI shows. (Airports are the one exception — never within an
  everyday radius, so they get a fixed wide search.)
- **True count vs sample.** `overpass.gather()` returns the real number of matches
  at the radius *and* a capped nearest sample. The count is never the cap — "25
  supermarkets" can't secretly mean "we stopped at 25".
- **Per-category failure isolation.** A slow or failing Overpass category (a 429,
  a timeout, the heavy airport query) degrades to an empty result for that
  category instead of aborting the whole run. The router turns a geocode/Overpass
  outage into a `503` (or an SSE `error` frame) **with** CORS headers — an
  unhandled 500 is generated outside Starlette's CORS middleware and the browser
  masks it as an opaque "Failed to fetch".
- **Transit verification, never invention.** Stops are classified from
  authoritative OSM tags, order-correct (a U-Bahn stop also tagged
  `railway=station` is read as `subway`, not `rail`), and anything ambiguous is
  left unclassified — a bus-only city shows buses, never phantom U-/S-Bahn. The
  full set is summarised into a `transit_breakdown` (`{"bus": 12, "rail": 2}`).
- **Streaming.** `iter_gather` is a generator that yields a `GatherProgress` per
  category and *returns* the assembled result via `StopIteration.value`. The agent
  builds `run_streaming` on top, and the router exposes it as Server-Sent Events.
- **$0 + compliant.** Nominatim and Overpass are free; both are rate-limited and
  send a descriptive User-Agent, results are cached hard in Postgres, and the
  ODbL-required `© OpenStreetMap contributors` attribution travels with the data.

## Categories and tags

Category → OSM selector(s) live in `overpass.py` (`CATEGORY_FILTERS`):

| Category | OSM tags |
|----------|----------|
| `school` | `amenity=school` |
| `gym` | `leisure=fitness_centre`, `leisure=sports_centre` |
| `supermarket` | `shop=supermarket`/`convenience`/`grocery` |
| `park` | `leisure=park` |
| `transit` | `highway=bus_stop`, `railway=tram_stop`, `railway=station`, `station=subway` |
| `pharmacy` | `amenity=pharmacy` |
| `airport` | `aeroway=aerodrome` (fixed ~50 km radius) |

Transit sub-type (`Poi.transit_type`): `bus` / `tram` / `subway` / `light_rail` /
`rail`, or `None` when unclassifiable.

## Configuration

Environment variables (see `.env.example`):

- `NOMINATIM_USER_AGENT` — **required**; descriptive UA with contact info
  (Nominatim policy). Reused as the Overpass UA. Without it the endpoints 503.
- `NOMINATIM_BASE_URL`, `OVERPASS_BASE_URL` — default to the public endpoints.
- `OPENROUTER_API_KEY` — required for the blurb LLM.
- `LOCALITY_AGENT_MODEL` — blurb model (default `openai/gpt-4o-mini`).

## Persistence

One `locality_insights` row per property (unique `property_id`, so re-running
upserts): resolved coordinate, radius, raw POI samples (JSON), per-category
counts, `transit_breakdown`, the blurb, and the OSM attribution. The
geocode/Overpass caches live in `geocode_cache` / `poi_cache`. See migrations
`0004_locality_enrichment` and `0005_transit_breakdown`.

## Tests

- `tests/unit/test_locality_geocode.py` — structured query, country isolation,
  rate limit, retries, cache.
- `tests/unit/test_locality_overpass.py` — tag selectors, transit classifier,
  true-count-vs-sample, subtype breakdown.
- `tests/unit/test_locality_gather.py` — single radius, airport radius, failure
  isolation, streaming generator.
- `tests/unit/test_locality_agent.py` — blurb synthesis, deterministic fallback,
  streaming events.
- `tests/integration/test_locality_api.py` — endpoints, validation, persistence,
  SSE frames, the 503/error path.
