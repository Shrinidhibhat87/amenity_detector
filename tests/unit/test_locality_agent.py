"""Unit tests for the locality tool-calling agent.

The agent is driven by a *scripted* fake OpenAI client: each call to
``chat.completions.create`` pops the next pre-baked response (a set of tool
calls, or a finalize). The geocode + Overpass clients are lightweight stubs so
no network is touched. We cover the happy path, the sparse-result widen path,
the iteration cap, and a geocode-miss.
"""

from __future__ import annotations

import json
from typing import Any

from core.locality.agent import LocalityAgent, LocalityResult
from core.locality.geocode import GeocodeResult
from core.locality.overpass import Poi


# ── Fake OpenAI response shapes (mirror the SDK attributes the agent reads) ──
class _FakeFunction:
    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        self.name = name
        self.arguments = json.dumps(arguments)


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: dict[str, Any]) -> None:
        self.id = call_id
        self.type = "function"
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(
        self, *, content: str | None = None, tool_calls: list[_FakeToolCall] | None = None
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _FakeCompletion:
    def __init__(self, message: _FakeMessage) -> None:
        self.choices = [type("C", (), {"message": message})()]


class _FakeCompletions:
    def __init__(self, scripted: list[_FakeMessage]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeCompletion:
        self.calls.append(kwargs)
        if not self._scripted:
            raise AssertionError("fake OpenAI ran out of scripted messages")
        return _FakeCompletion(self._scripted.pop(0))


class _FakeOpenAI:
    def __init__(self, scripted: list[_FakeMessage]) -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions(scripted)})()


# ── Stub OSM clients ─────────────────────────────────────────────────────────
class _StubGeocode:
    def __init__(self, result: GeocodeResult | None) -> None:
        self._result = result
        self.queries: list[str] = []

    def geocode(self, query: str) -> GeocodeResult | None:
        self.queries.append(query)
        return self._result


class _StubOverpass:
    """Returns scripted POIs keyed on (category, radius)."""

    def __init__(self, table: dict[tuple[str, int], list[Poi]]) -> None:
        self._table = table
        self.calls: list[tuple[str, int]] = []

    def query(
        self, *, lat: float, lon: float, radius_m: int, category: str, limit: int = 25
    ) -> list[Poi]:
        self.calls.append((category, radius_m))
        return self._table.get((category, radius_m), [])


def _poi(category: str, name: str, dist: float) -> Poi:
    return Poi(
        category=category,
        name=name,
        latitude=50.11,
        longitude=8.68,
        osm_type="node",
        osm_id=1,
        distance_m=dist,
        tags={},
    )


_FRANKFURT = GeocodeResult(
    query="60311",
    latitude=50.1109,
    longitude=8.6821,
    bbox=(50.0969, 50.1249, 8.6621, 8.7021),
    display_name="60311 Frankfurt am Main, Germany",
)


def _make_agent(
    scripted: list[_FakeMessage],
    geocode: _StubGeocode,
    overpass: _StubOverpass,
    *,
    max_iterations: int = 6,
) -> tuple[LocalityAgent, _FakeOpenAI]:
    client = _FakeOpenAI(scripted)
    agent = LocalityAgent(
        openai_client=client,
        geocode_client=geocode,
        overpass_client=overpass,
        model="openai/gpt-4o-mini",
        max_iterations=max_iterations,
    )
    return agent, client


def test_happy_path_geocode_then_pois_then_finalize() -> None:
    geocode = _StubGeocode(_FRANKFURT)
    overpass = _StubOverpass(
        {
            ("school", 1000): [_poi("school", "Goethe-Schule", 120)],
            ("park", 1000): [_poi("park", "Stadtpark", 300), _poi("park", "Grüneburgpark", 800)],
        }
    )
    scripted = [
        _FakeMessage(tool_calls=[_FakeToolCall("c1", "geocode", {"query": "60311"})]),
        _FakeMessage(
            tool_calls=[
                _FakeToolCall("c2", "overpass_query", {"category": "school", "radius_m": 1000}),
                _FakeToolCall("c3", "overpass_query", {"category": "park", "radius_m": 1000}),
            ]
        ),
        _FakeMessage(
            tool_calls=[
                _FakeToolCall(
                    "c4", "finalize", {"blurb": "Central spot with a school and two parks nearby."}
                )
            ]
        ),
    ]
    agent, _ = _make_agent(scripted, geocode, overpass)

    result = agent.run("60311")

    assert isinstance(result, LocalityResult)
    assert result.latitude == 50.1109
    assert result.display_name.startswith("60311")
    assert result.category_counts == {"school": 1, "park": 2}
    assert len(result.pois) == 3
    assert "school" in result.blurb
    assert "OpenStreetMap" in result.attribution


def test_widens_radius_on_sparse_results() -> None:
    geocode = _StubGeocode(_FRANKFURT)
    overpass = _StubOverpass(
        {
            ("school", 500): [],  # too tight → empty
            ("school", 2000): [_poi("school", "Goethe-Schule", 1500)],
        }
    )
    scripted = [
        _FakeMessage(tool_calls=[_FakeToolCall("c1", "geocode", {"query": "60311"})]),
        _FakeMessage(
            tool_calls=[
                _FakeToolCall("c2", "overpass_query", {"category": "school", "radius_m": 500})
            ]
        ),
        _FakeMessage(
            tool_calls=[
                _FakeToolCall("c3", "overpass_query", {"category": "school", "radius_m": 2000})
            ]
        ),
        _FakeMessage(
            tool_calls=[_FakeToolCall("c4", "finalize", {"blurb": "School a short ride away."})]
        ),
    ]
    agent, _ = _make_agent(scripted, geocode, overpass)

    result = agent.run("60311")

    assert ("school", 500) in overpass.calls
    assert ("school", 2000) in overpass.calls
    assert result.category_counts == {"school": 1}


def test_iteration_cap_forces_a_fallback_finalize() -> None:
    geocode = _StubGeocode(_FRANKFURT)
    overpass = _StubOverpass({("park", 1000): [_poi("park", "Stadtpark", 100)]})
    # The model never finalizes — it keeps asking for the same query forever.
    never_ending = [
        _FakeMessage(tool_calls=[_FakeToolCall("c1", "geocode", {"query": "60311"})]),
    ] + [
        _FakeMessage(
            tool_calls=[
                _FakeToolCall(f"c{i}", "overpass_query", {"category": "park", "radius_m": 1000})
            ]
        )
        for i in range(2, 20)
    ]
    agent, _ = _make_agent(never_ending, geocode, overpass, max_iterations=4)

    result = agent.run("60311")

    # Agent must terminate and synthesize a non-empty fallback blurb.
    assert isinstance(result, LocalityResult)
    assert result.blurb
    assert result.category_counts.get("park") == 1


def test_geocode_miss_still_returns_a_result() -> None:
    geocode = _StubGeocode(None)
    overpass = _StubOverpass({})
    scripted = [
        _FakeMessage(tool_calls=[_FakeToolCall("c1", "geocode", {"query": "nowhere-xyz"})]),
        _FakeMessage(tool_calls=[_FakeToolCall("c2", "finalize", {"blurb": "Could not locate."})]),
    ]
    agent, _ = _make_agent(scripted, geocode, overpass)

    result = agent.run("nowhere-xyz")

    assert result.latitude is None
    assert result.pois == []
    assert result.blurb == "Could not locate."
