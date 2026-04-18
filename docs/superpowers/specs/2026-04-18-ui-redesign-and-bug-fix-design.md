# Design: UI Redesign + Generation Bug Fix
**Date:** 2026-04-18  
**Status:** Approved  
**Covers:** T1 (generation bug fix) + T2 (landing page + Upload/Browse redesign)

---

## Context

Phases 0–4 are complete. The app runs on FastAPI + Gradio inside Docker Compose, with Ollama serving local VLMs on the host. Two problems were identified:

1. **Generation bug (T1):** Images upload successfully but the VLM description is never generated. **Confirmed via `docker compose logs api`:** two distinct errors.

2. **UI aesthetics (T2):** The current Gradio UI is functional but visually plain. It goes straight to functional tabs with no landing page, uses default Gradio styling, and the upload flow has no step-by-step structure or editable amenity review.

---

## T1 — Generation Bug Fix

### Root Causes (confirmed from `docker compose logs api`)

**Bug 1 — Ollama unreachable from Docker:**
```
ERROR: Cannot connect to Ollama at http://localhost:11434. Is the Ollama server running?
```
Inside a Docker container, `localhost` means the container itself — not the host machine where `ollama serve` is running. The fix is to point the API container at the host via `host.docker.internal` (works on Docker Desktop / WSL2) or the host gateway IP (Linux without Docker Desktop).

**Bug 2 — Gemini free-tier quota exhausted:**
```
429 RESOURCE_EXHAUSTED: Quota exceeded for generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0
```
The API key is correctly forwarded and working — but the free-tier daily quota has been exhausted. `limit: 0` means the free tier for this Google Cloud project has been fully consumed. This is **not a code bug**; it resets daily. In the meantime, switching to a local Ollama model is the workaround.

**Note on error visibility:** The backend logs are already showing errors correctly (good logging in `core/amenity_system`). The problem is the Gradio UI shows only a generic status message — users can't see the real error. The UI should surface the error detail returned by the API.

### Fix Plan
1. Update `docker-compose.yml`: add `OLLAMA_BASE_URL: http://host.docker.internal:11434` to the `api` service environment. Add `extra_hosts: ["host.docker.internal:host-gateway"]` for Linux compatibility.
2. Surface API error details in the Gradio UI status field (pass the error message from the API response through to the user).
3. Gemini quota: document in README that free tier resets daily; no code change needed.
4. Test Ollama end-to-end after the networking fix.

---

## T2 — UI Redesign

### Visual Identity
- **Color palette:** Warm neutral (B direction) — cream/sand background (`#fafaf7` → `#f0ede4`), amber accent (`#b45309`), dark stone text (`#1c1917`), muted body text (`#78716c`).
- **Typography:** System font stack (`'Segoe UI', system-ui, sans-serif`), heavy weights for headings (800), medium for labels (600).
- **Components:** Rounded corners (8–14px), subtle shadows, warm yellow highlights (`#fef3c7`) for room headers and tags.

### Page 1 — Landing / Home Page (new)

A full-screen hero that is the first thing users see. Structure:

**Nav bar (sticky, white):**
- Left: Brand — small house SVG icon that "draws" itself on load (stroke-dasharray animation, 1.2s), then brand name fades in.
- Right: Minimal nav links (How it works, Models, Docs).

**Hero section (centered):**
- Small amber badge: "✦ AI-Powered Property Analysis"
- Large headline: "Smart Property Amenity Detection"
- Typewriter subtitle alternating between two sentences (type → pause → delete → switch):
  1. *"AI-powered image analysis for real estate listings and property management."*
  2. *"Search for a property based on amenities using your voice."*
- Two CTA buttons (centered, side by side):
  - Primary (amber filled): "↑ Upload & Detect"
  - Secondary (white + amber border): "🔍 Browse Properties"
- Feature chips row: Room Classification · Amenity Detection · Editable Results · AI Descriptions · Voice Search (coming soon)

**Implementation note:** Gradio's `gr.Blocks` does not support a true "page before tabs" pattern natively. The landing page will be implemented as a dedicated HTML block rendered via `gr.HTML` at the top of the app, with buttons that use JavaScript `document.querySelector` to programmatically click the relevant Gradio tab. This is a known Gradio pattern for hero sections.

### Page 2 — Upload & Detect (replaces current Upload tab)

**Layout:** Two columns — left (form panel, fixed ~320px), right (results panel, flex).

**Step indicator** at top: Upload → Detect → Review & Edit → Generate Description (highlights current step).

**Left panel — Property Details:**
- Upload zone (dashed border, drag-and-drop, shows thumbnails with detected room badges after upload)
- Property name text input
- Model dropdown (gemini-2.0-flash recommended, qwen2.5vl:7b, llama3.2-vision:11b)
- Optional notes textarea
- "Upload & Detect Amenities" primary button

**Right panel — Review & Edit:**
- "Edit Mode" toggle button (top right of panel)
- One "Room Card" per detected image:
  - Header: room emoji + room type label + filename
  - Amenity table: Amenity name | Present toggle (on/off switch) | Confidence badge (green/amber/red)
  - Toggles are interactive — users flip them to confirm/reject
- "Confirm & Generate Description" green button (bottom right)
- After confirmation: description appears below in a styled text box with "Edit Description" and "Save Property" actions

**Editing behavior:** Toggles are always clickable (no separate "edit mode" required — the toggle is the editing mechanism). The "Edit Mode" button label is cosmetic/optional and can be removed in implementation if redundant.

### Page 3 — Browse Properties (replaces current Browse tab)

**Search bar:** Full-width rounded pill — search icon + text input + "Search" primary button + "List All" secondary button.

**Results:** Property cards (not a raw dataframe table):
- Each card: property thumbnail placeholder, name, truncated description, amenity tag chips, metadata (model, image count) on the right.
- Selected card highlights with amber border.

**Detail panel** (below cards, appears on card click):
- Property name, ID, model, timestamp
- Room summary chips (e.g., "🍳 Kitchen — Refrigerator, Oven")
- Full generated description in styled text box
- Action buttons: View Images · Edit Amenities · Delete (red)

---

## Implementation Scope

### What changes
| File | Change |
|---|---|
| `ui/app.py` | Complete rewrite — new landing page HTML block, redesigned Upload and Browse tabs |
| `docker-compose.yml` | Add `OLLAMA_BASE_URL` env var for host networking |
| `api/routers/properties.py` | Improve error logging (log full traceback) |
| `.env.example` | Verify `GEMINI_API_KEY` is documented |

### What does NOT change
- FastAPI backend logic (no API changes needed)
- Database models or migrations
- VLM clients (bug is in deployment config, not client code)
- Tests (existing 119 tests remain valid; new UI tests will be added)

---

## Success Criteria

- `docker compose up` → landing page loads at `http://localhost:7860`
- Uploading images with Gemini model → amenities detected, review table shown, description generated
- Uploading images with a local Ollama model (qwen2.5vl:7b) → same flow works
- User can toggle amenity presence, click "Confirm & Generate", see updated description
- Browse page shows property cards; clicking a card shows full details
- `docker compose logs api` shows structured error logs (no more silent failures)
