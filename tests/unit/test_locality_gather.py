"""Unit tests for the deterministic locality gather.

These assert the architectural fix at the heart of this phase: every everyday
category is queried at ONE radius (the slider value), airports get their own wide
radius, and the counts are the true totals — never capped, never mixed across
radii. The geocode + Overpass clients are fakes so the tests run offline.
"""

from __future__ import annotations

from core.locality.gather import (
    AIRPORT_RADIUS_M,
    DEFAULT_RADIUS_M,
    EVERYDAY_CATEGORIES,
    MAX_RADIUS_M,
    MIN_RADIUS_M,
    clamp_radius,
    gather_locality,
)
from core.locality.geocode import GeocodeResult
from core.locality.overpass import CategoryResult, Poi


class _FakeGeocode:
    def __init__(self, result: GeocodeResult | None) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    def geocode(
        self, postal_code: str, *, street: str | None = None, country_code: str = "DE"
    ) -> GeocodeResult | None:
        self.calls.append(
            {"postal_code": postal_code, "street": street, "country_code": country_code}
        )
        return self._result


class _FakeOverpass:
    """Records the radius used per category and returns a scripted total."""

    def __init__(self, totals: dict[str, int]) -> None:
        self._totals = totals
        self.calls: list[dict[str, object]] = []

    def gather(
        self, *, lat: float, lon: float, radius_m: int, category: str, sample_limit: int = 5
    ) -> CategoryResult:
        self.calls.append({"category": category, "radius_m": radius_m})
        total = self._totals.get(category, 0)
        pois = [
            Poi(
                category=category,
                name=f"{category}-{i}",
                latitude=lat,
                longitude=lon,
                osm_type="node",
                osm_id=i,
                distance_m=float(i * 10),
            )
            for i in range(min(total, sample_limit))
        ]
        return CategoryResult(category=category, total=total, pois=pois)


def _center() -> GeocodeResult:
    return GeocodeResult(
        query="de|52062",
        latitude=50.77,
        longitude=6.08,
        bbox=(50.7, 50.8, 6.0, 6.1),
        display_name="52062 Aachen, Germany",
    )


def _totals() -> dict[str, int]:
    return {"school": 3, "gym": 1, "supermarket": 7, "park": 2, "transit": 12, "airport": 1}


def test_returns_none_when_geocode_fails() -> None:
    result = gather_locality(
        geocode_client=_FakeGeocode(None),
        overpass_client=_FakeOverpass({}),
        postal_code="00000",
    )
    assert result is None


def test_everyday_categories_share_one_radius() -> None:
    overpass = _FakeOverpass(_totals())
    gather_locality(
        geocode_client=_FakeGeocode(_center()),
        overpass_client=overpass,
        postal_code="52062",
        radius_m=2000,
    )

    everyday = [c for c in overpass.calls if c["category"] != "airport"]
    assert {c["category"] for c in everyday} == set(EVERYDAY_CATEGORIES)
    assert all(c["radius_m"] == 2000 for c in everyday)  # one radius, no per-category drift


def test_airport_uses_the_wide_fixed_radius() -> None:
    overpass = _FakeOverpass(_totals())
    gather_locality(
        geocode_client=_FakeGeocode(_center()),
        overpass_client=overpass,
        postal_code="52062",
        radius_m=3000,
    )

    airport = next(c for c in overpass.calls if c["category"] == "airport")
    assert airport["radius_m"] == AIRPORT_RADIUS_M


def test_counts_are_true_totals_not_capped() -> None:
    result = gather_locality(
        geocode_client=_FakeGeocode(_center()),
        overpass_client=_FakeOverpass(_totals()),
        postal_code="52062",
        sample_limit=5,
    )
    assert result is not None
    # 7 supermarkets total even though only 5 are in the sample list.
    assert result.category_counts["supermarket"] == 7
    assert len([p for p in result.pois if p.category == "supermarket"]) == 5


def test_empty_categories_dropped_from_counts() -> None:
    overpass = _FakeOverpass({"school": 2})  # everything else zero
    result = gather_locality(
        geocode_client=_FakeGeocode(_center()),
        overpass_client=overpass,
        postal_code="52062",
    )
    assert result is not None
    assert result.category_counts == {"school": 2}


def test_radius_is_clamped_into_slider_range() -> None:
    overpass = _FakeOverpass(_totals())
    result = gather_locality(
        geocode_client=_FakeGeocode(_center()),
        overpass_client=overpass,
        postal_code="52062",
        radius_m=99999,  # above MAX
    )
    assert result is not None
    assert result.radius_m == MAX_RADIUS_M


def test_clamp_radius_bounds() -> None:
    assert clamp_radius(10) == MIN_RADIUS_M
    assert clamp_radius(99999) == MAX_RADIUS_M
    assert clamp_radius(DEFAULT_RADIUS_M) == DEFAULT_RADIUS_M


class _FlakyOverpass:
    """Raises for one category, returns a normal result for the rest."""

    def __init__(self, failing: str) -> None:
        self._failing = failing

    def gather(
        self, *, lat: float, lon: float, radius_m: int, category: str, sample_limit: int = 5
    ) -> CategoryResult:
        if category == self._failing:
            raise RuntimeError("overpass timed out")
        return CategoryResult(category=category, total=2, pois=[])


def test_failing_category_is_isolated_not_fatal() -> None:
    # The airport query blows up, but everyday categories still come back.
    result = gather_locality(
        geocode_client=_FakeGeocode(_center()),
        overpass_client=_FlakyOverpass(failing="airport"),
        postal_code="52062",
    )
    assert result is not None
    assert result.results["airport"].total == 0  # degraded to empty
    assert result.category_counts["school"] == 2  # the rest survived


def test_transit_breakdown_surfaced_on_result() -> None:
    class _TransitOverpass:
        def gather(
            self, *, lat: float, lon: float, radius_m: int, category: str, sample_limit: int = 5
        ) -> CategoryResult:
            if category == "transit":
                return CategoryResult(
                    category="transit", total=14, pois=[], subtype_counts={"bus": 12, "rail": 2}
                )
            return CategoryResult(category=category, total=0, pois=[])

    result = gather_locality(
        geocode_client=_FakeGeocode(_center()),
        overpass_client=_TransitOverpass(),
        postal_code="52062",
    )
    assert result is not None
    assert result.transit_breakdown == {"bus": 12, "rail": 2}


def test_geocode_called_with_structured_args() -> None:
    geocode = _FakeGeocode(_center())
    gather_locality(
        geocode_client=geocode,
        overpass_client=_FakeOverpass(_totals()),
        postal_code="52062",
        street="Bendelstrasse",
        country_code="DE",
    )
    assert geocode.calls[0] == {
        "postal_code": "52062",
        "street": "Bendelstrasse",
        "country_code": "DE",
    }
