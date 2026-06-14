"""Unit tests for the locality agent (blurb synthesis).

The data gathering is now deterministic (tested in test_locality_gather.py); the
agent's remaining job is to turn the gathered summary into prose. So these tests
drive a fake OpenAI client whose single completion returns a blurb, with stub
geocode + Overpass clients underneath. We cover the happy path, the deterministic
fallback when the model errors or returns nothing, and the geocode-miss path.
"""

from __future__ import annotations

from typing import Any

from core.locality.agent import LocalityAgent, LocalityResult
from core.locality.geocode import GeocodeResult
from core.locality.overpass import CategoryResult, Poi


# ── Fake OpenAI (single completion → message.content) ────────────────────────
class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeCompletion:
    def __init__(self, message: _FakeMessage) -> None:
        self.choices = [type("C", (), {"message": message})()]


class _FakeCompletions:
    def __init__(self, content: str | None, *, raises: bool = False) -> None:
        self._content = content
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeCompletion:
        self.calls.append(kwargs)
        if self._raises:
            raise RuntimeError("model unavailable")
        return _FakeCompletion(_FakeMessage(self._content))


class _FakeOpenAI:
    def __init__(self, content: str | None, *, raises: bool = False) -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content, raises=raises)})()


# ── Stub OSM clients (new structured signatures) ─────────────────────────────
class _StubGeocode:
    def __init__(self, result: GeocodeResult | None) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def geocode(
        self, postal_code: str, *, street: str | None = None, country_code: str = "DE"
    ) -> GeocodeResult | None:
        self.calls.append(
            {"postal_code": postal_code, "street": street, "country_code": country_code}
        )
        return self._result


class _StubOverpass:
    def __init__(self, totals: dict[str, int]) -> None:
        self._totals = totals

    def gather(
        self, *, lat: float, lon: float, radius_m: int, category: str, sample_limit: int = 5
    ) -> CategoryResult:
        total = self._totals.get(category, 0)
        pois = [
            Poi(
                category=category,
                name=f"{category}-{i}",
                latitude=lat,
                longitude=lon,
                osm_type="node",
                osm_id=i,
                distance_m=float((i + 1) * 100),
            )
            for i in range(min(total, sample_limit))
        ]
        return CategoryResult(category=category, total=total, pois=pois)


_AACHEN = GeocodeResult(
    query="de|52062",
    latitude=50.77,
    longitude=6.08,
    bbox=(50.7, 50.8, 6.0, 6.1),
    display_name="52062 Aachen, Germany",
)


def _make_agent(
    openai: _FakeOpenAI, geocode: _StubGeocode, overpass: _StubOverpass
) -> LocalityAgent:
    return LocalityAgent(
        openai_client=openai,
        geocode_client=geocode,
        overpass_client=overpass,
        model="openai/gpt-4o-mini",
    )


def test_happy_path_writes_blurb_from_gathered_counts() -> None:
    agent = _make_agent(
        _FakeOpenAI("Lively spot with several supermarkets and a school nearby."),
        _StubGeocode(_AACHEN),
        _StubOverpass({"school": 1, "supermarket": 4}),
    )

    result = agent.run(postal_code="52062", radius_m=3000)

    assert isinstance(result, LocalityResult)
    assert result.latitude == 50.77
    assert result.display_name.startswith("52062")
    assert result.category_counts == {"school": 1, "supermarket": 4}
    assert "supermarket" in result.blurb
    assert "OpenStreetMap" in result.attribution


def test_summary_prompt_carries_radius_in_km() -> None:
    openai = _FakeOpenAI("ok")
    agent = _make_agent(openai, _StubGeocode(_AACHEN), _StubOverpass({"park": 2}))

    agent.run(postal_code="52062", radius_m=3000)

    user_msg = openai.chat.completions.calls[0]["messages"][1]["content"]
    assert "3 km" in user_msg  # radius rendered in km, not metres


def test_empty_model_response_falls_back_to_deterministic_blurb() -> None:
    agent = _make_agent(
        _FakeOpenAI(""),  # model returns nothing
        _StubGeocode(_AACHEN),
        _StubOverpass({"supermarket": 3}),
    )

    result = agent.run(postal_code="52062", radius_m=2000)

    assert result.blurb  # non-empty
    assert "supermarkets" in result.blurb
    assert "2 km" in result.blurb


def test_model_error_falls_back_to_deterministic_blurb() -> None:
    agent = _make_agent(
        _FakeOpenAI(None, raises=True),
        _StubGeocode(_AACHEN),
        _StubOverpass({"gym": 1}),
    )

    result = agent.run(postal_code="52062")

    assert result.blurb
    assert "gym" in result.blurb


def test_geocode_miss_returns_empty_result() -> None:
    agent = _make_agent(
        _FakeOpenAI("unused"),
        _StubGeocode(None),
        _StubOverpass({}),
    )

    result = agent.run(postal_code="00000")

    assert result.latitude is None
    assert result.pois == []
    assert result.category_counts == {}
    assert result.blurb == "This location could not be found."


def test_run_streaming_emits_category_then_summary_then_result() -> None:
    agent = _make_agent(
        _FakeOpenAI("Walkable area with supermarkets and a school."),
        _StubGeocode(_AACHEN),
        _StubOverpass({"school": 1, "supermarket": 4}),
    )

    events = list(agent.run_streaming(postal_code="52062", radius_m=3000))

    types = [e["type"] for e in events]
    # One category event per gathered category, then summary, then result.
    assert types[-2:] == ["summary", "result"]
    assert types.count("category") >= 1
    cat_events = [e for e in events if e["type"] == "category"]
    assert cat_events[-1]["category"] == "airport"
    assert cat_events[0]["total"] == cat_events[-1]["done"]  # done climbs to total
    result = events[-1]["result"]
    assert isinstance(result, LocalityResult)
    assert result.category_counts == {"school": 1, "supermarket": 4}


def test_run_streaming_geocode_miss_yields_single_empty_result() -> None:
    agent = _make_agent(_FakeOpenAI("unused"), _StubGeocode(None), _StubOverpass({}))

    events = list(agent.run_streaming(postal_code="00000"))

    assert [e["type"] for e in events] == ["result"]
    assert events[0]["result"].blurb == "This location could not be found."


def test_run_passes_structured_args_to_geocode() -> None:
    geocode = _StubGeocode(_AACHEN)
    agent = _make_agent(_FakeOpenAI("ok"), geocode, _StubOverpass({"park": 1}))

    agent.run(postal_code="52062", street="Bendelstrasse", country_code="DE")

    assert geocode.calls[0] == {
        "postal_code": "52062",
        "street": "Bendelstrasse",
        "country_code": "DE",
    }
