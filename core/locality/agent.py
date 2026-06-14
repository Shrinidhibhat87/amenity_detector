"""Locality enrichment agent — a tool-calling loop over OpenRouter.

This is the one genuinely *agentic* piece of the seller pipeline. Given a
free-text location, the model decides what to do via three tools:

  - ``geocode(query)``                    — resolve the location to a coordinate.
  - ``overpass_query(category, radius_m)``— count/list nearby POIs in a category.
  - ``finalize(blurb)``                   — end the run with the written Lage text.

The model drives the multi-step decisioning the plan calls for: which categories
to look up, how wide a radius, when to *widen* on sparse results, and when it has
enough to synthesize the neighbourhood paragraph. The Python side just executes
the tools against the real OSM clients, feeds results back, and guards against a
runaway loop with a hard iteration cap (after which it synthesizes a deterministic
fallback blurb from the counts it has).

The OpenAI-SDK client is injected (pointed at OpenRouter in production via
:meth:`LocalityAgent.from_env`); unit tests inject a scripted fake.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Self

from core.locality.geocode import GeocodeResult
from core.locality.overpass import CATEGORIES, Poi

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "openai/gpt-4o-mini"
_DEFAULT_RADIUS_M = 1000
_MIN_RADIUS_M = 200
_MAX_RADIUS_M = 5000
_ATTRIBUTION = "© OpenStreetMap contributors"

_SYSTEM_PROMPT = f"""You write the "Lage" (neighbourhood) section of a property listing.

You are given a free-text location (a PIN code, a street, or a Stadtteil). Use the
tools to build a picture of what is nearby, then write a short, factual paragraph.

Workflow:
  1. Call `geocode` once to resolve the location to a coordinate.
  2. Call `overpass_query` for the relevant categories to count what is nearby.
     Available categories: {", ".join(CATEGORIES)}.
     Start around {_DEFAULT_RADIUS_M} m. If a category comes back empty or sparse,
     widen the radius (up to {_MAX_RADIUS_M} m) and query it again.
  3. Call `finalize` with a 2-4 sentence neighbourhood blurb. Mention the
     categories that actually have nearby results; do not invent places. Keep it
     concrete (e.g. "two supermarkets and a park within walking distance").

Only describe what the tools returned. Never fabricate POIs or distances."""

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": "Resolve a free-text location to a coordinate.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "overpass_query",
            "description": "Count and list nearby POIs of one category around the geocoded centre.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "radius_m": {"type": "integer"},
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize",
            "description": "Finish with the written neighbourhood blurb.",
            "parameters": {
                "type": "object",
                "properties": {"blurb": {"type": "string"}},
                "required": ["blurb"],
            },
        },
    },
]


@dataclass
class LocalityResult:
    """The agent's output — everything needed to persist + render the Lage section."""

    location_query: str
    display_name: str
    latitude: float | None
    longitude: float | None
    radius_m: int
    pois: list[Poi]
    category_counts: dict[str, int]
    blurb: str
    attribution: str = _ATTRIBUTION


@dataclass
class _RunState:
    """Mutable scratch space accumulated across tool calls within one run."""

    center: GeocodeResult | None = None
    # Latest POI list per category (a re-query at a wider radius overwrites).
    pois_by_category: dict[str, list[Poi]] = field(default_factory=dict)
    last_radius_m: int = _DEFAULT_RADIUS_M


class LocalityAgent:
    """Drives the geocode → overpass → finalize tool loop."""

    def __init__(
        self,
        *,
        openai_client: Any,
        geocode_client: Any,
        overpass_client: Any,
        model: str = _DEFAULT_MODEL,
        max_iterations: int = 8,
    ) -> None:
        self._client = openai_client
        self._geocode = geocode_client
        self._overpass = overpass_client
        self._model = model
        self._max_iterations = max_iterations

    @classmethod
    def from_env(cls, *, geocode_client: Any, overpass_client: Any) -> Self:
        from openai import OpenAI  # imported lazily so unit tests can mock it

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required to build a LocalityAgent.")
        timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "60"))
        client = OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=api_key, timeout=timeout)
        model = os.getenv("LOCALITY_AGENT_MODEL", _DEFAULT_MODEL)
        return cls(
            openai_client=client,
            geocode_client=geocode_client,
            overpass_client=overpass_client,
            model=model,
        )

    def run(self, location_query: str, *, radius_hint: int | None = None) -> LocalityResult:
        """Run the agent loop for ``location_query`` and return the enriched result.

        ``radius_hint`` is an optional preferred starting search radius (metres)
        surfaced to the model; it still decides the final radius and may widen.
        """
        state = _RunState()
        user_msg = f"Location: {location_query}"
        if radius_hint is not None:
            state.last_radius_m = _clamp_radius(radius_hint)
            user_msg += f"\nPreferred starting search radius: {state.last_radius_m} m."
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        for _ in range(self._max_iterations):
            message = self._next_message(messages)
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                # No tool call — treat any text as the blurb and stop.
                blurb = (getattr(message, "content", None) or "").strip()
                return self._build_result(
                    location_query, state, blurb or self._fallback_blurb(state)
                )

            messages.append(_assistant_message(message, tool_calls))

            for call in tool_calls:
                name = call.function.name
                args = _parse_args(call.function.arguments)

                if name == "finalize":
                    blurb = str(args.get("blurb", "")).strip() or self._fallback_blurb(state)
                    return self._build_result(location_query, state, blurb)

                output = self._dispatch(name, args, state)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(output)}
                )

        # Iteration cap hit without a finalize — synthesize from what we have.
        logger.info("locality agent hit iteration cap for %r; using fallback blurb", location_query)
        return self._build_result(location_query, state, self._fallback_blurb(state))

    def _next_message(self, messages: list[dict[str, Any]]) -> Any:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=_TOOLS,
            temperature=0,
        )
        return completion.choices[0].message

    def _dispatch(self, name: str, args: dict[str, Any], state: _RunState) -> dict[str, Any]:
        if name == "geocode":
            return self._do_geocode(args, state)
        if name == "overpass_query":
            return self._do_overpass(args, state)
        return {"error": f"unknown tool {name!r}"}

    def _do_geocode(self, args: dict[str, Any], state: _RunState) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        result = self._geocode.geocode(query)
        if result is None:
            return {"found": False}
        state.center = result
        return {
            "found": True,
            "latitude": result.latitude,
            "longitude": result.longitude,
            "display_name": result.display_name,
        }

    def _do_overpass(self, args: dict[str, Any], state: _RunState) -> dict[str, Any]:
        if state.center is None:
            return {"error": "call geocode first to establish a centre"}

        category = str(args.get("category", ""))
        if category not in CATEGORIES:
            return {"error": f"unknown category {category!r}"}

        radius = _clamp_radius(int(args.get("radius_m", _DEFAULT_RADIUS_M)))
        state.last_radius_m = radius
        pois = self._overpass.query(
            lat=state.center.latitude,
            lon=state.center.longitude,
            radius_m=radius,
            category=category,
        )
        state.pois_by_category[category] = pois
        return {
            "category": category,
            "radius_m": radius,
            "count": len(pois),
            "nearest": [
                {"name": p.name or "(unnamed)", "distance_m": round(p.distance_m)} for p in pois[:5]
            ],
        }

    def _build_result(self, location_query: str, state: _RunState, blurb: str) -> LocalityResult:
        flat: list[Poi] = [p for pois in state.pois_by_category.values() for p in pois]
        counts = {cat: len(pois) for cat, pois in state.pois_by_category.items() if pois}
        center = state.center
        return LocalityResult(
            location_query=location_query,
            display_name=center.display_name if center else "",
            latitude=center.latitude if center else None,
            longitude=center.longitude if center else None,
            radius_m=state.last_radius_m,
            pois=flat,
            category_counts=counts,
            blurb=blurb,
        )

    @staticmethod
    def _fallback_blurb(state: _RunState) -> str:
        """Deterministic blurb used when the model never produced one."""
        counts = {cat: len(pois) for cat, pois in state.pois_by_category.items() if pois}
        if not counts:
            return "No notable amenities were found nearby."
        parts = [f"{n} {cat}{'s' if n != 1 else ''}" for cat, n in counts.items()]
        return "Nearby: " + ", ".join(parts) + "."


def _clamp_radius(radius_m: int) -> int:
    return max(_MIN_RADIUS_M, min(_MAX_RADIUS_M, radius_m))


def _parse_args(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        logger.warning("locality agent got malformed tool arguments: %r", raw)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _assistant_message(message: Any, tool_calls: Any) -> dict[str, Any]:
    """Reconstruct the assistant turn (with tool_calls) for the next request."""
    return {
        "role": "assistant",
        "content": getattr(message, "content", None) or "",
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in tool_calls
        ],
    }
