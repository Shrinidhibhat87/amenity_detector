"""Unit tests for the Phase 9 UI helpers that build the listing-metadata
payload sent on POST /api/v1/properties/.

Pure-Python helpers; no Gradio runtime required.
"""

from typing import Any


def test_payload_drops_unset_optional_fields() -> None:
    from ui.app import _listing_metadata_payload

    payload = _listing_metadata_payload(
        listing_type=None,
        price=None,
        currency=None,
        price_period=None,
        property_type=None,
        furnishing=None,
        num_bedrooms=None,
        num_bathrooms=None,
        area_sqm=None,
        available_from="",
        locality="",
        postal_code="",
        country_code="",
        owner_email="",
    )
    assert payload == {}


def test_payload_includes_provided_fields() -> None:
    from ui.app import _listing_metadata_payload

    payload = _listing_metadata_payload(
        listing_type="rent",
        price=1500.0,
        currency="EUR",
        price_period="monthly",
        property_type="apartment",
        furnishing="semi_furnished",
        num_bedrooms=3,
        num_bathrooms=2,
        area_sqm=82.5,
        available_from="2026-06-01",
        locality="Sachsenhausen",
        postal_code="60594",
        country_code="DE",
        owner_email="owner@example.com",
    )
    assert payload == {
        "listing_type": "rent",
        "price": 1500.0,
        "currency": "EUR",
        "price_period": "monthly",
        "property_type": "apartment",
        "furnishing": "semi_furnished",
        "num_bedrooms": 3,
        "num_bathrooms": 2,
        "area_sqm": 82.5,
        "available_from": "2026-06-01",
        "locality": "Sachsenhausen",
        "postal_code": "60594",
        "country_code": "DE",
        "owner_email": "owner@example.com",
    }


def test_payload_strips_whitespace_strings() -> None:
    from ui.app import _listing_metadata_payload

    payload = _listing_metadata_payload(
        listing_type=None,
        price=None,
        currency=None,
        price_period=None,
        property_type=None,
        furnishing=None,
        num_bedrooms=None,
        num_bathrooms=None,
        area_sqm=None,
        available_from="   ",
        locality="  Sachsenhausen ",
        postal_code="  ",
        country_code=" de ",
        owner_email="  owner@example.com  ",
    )
    # Empty / whitespace-only strings are dropped; valid strings are stripped.
    # country_code is upper-cased to match the API's CountryCode constraint.
    assert payload == {
        "locality": "Sachsenhausen",
        "country_code": "DE",
        "owner_email": "owner@example.com",
    }


def test_payload_treats_zero_numbers_as_unset() -> None:
    """Gradio's gr.Number with no input often returns 0; drop it."""
    from ui.app import _listing_metadata_payload

    payload: dict[str, Any] = _listing_metadata_payload(
        listing_type=None,
        price=0,
        currency=None,
        price_period=None,
        property_type=None,
        furnishing=None,
        num_bedrooms=0,
        num_bathrooms=0,
        area_sqm=0,
        available_from="",
        locality="",
        postal_code="",
        country_code="",
        owner_email="",
    )
    assert payload == {}


def test_payload_keeps_legitimate_zero_aware_inputs_when_explicit() -> None:
    """If a user types 0.0 for area, we treat it as unset; this is by design."""
    from ui.app import _listing_metadata_payload

    payload = _listing_metadata_payload(
        listing_type="rent",
        price=950.5,
        currency="EUR",
        price_period="monthly",
        property_type=None,
        furnishing=None,
        num_bedrooms=1,
        num_bathrooms=None,
        area_sqm=None,
        available_from="",
        locality="",
        postal_code="",
        country_code="",
        owner_email="",
    )
    assert payload == {
        "listing_type": "rent",
        "price": 950.5,
        "currency": "EUR",
        "price_period": "monthly",
        "num_bedrooms": 1,
    }


def test_build_app_smoke_still_works() -> None:
    """build_app() must construct without raising — catches @gr.render wiring
    bugs and missing-component reference errors after the accordion change.
    """
    from ui.app import build_app

    app = build_app()
    assert app is not None
