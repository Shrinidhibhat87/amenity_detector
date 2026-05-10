"""Unit tests for the Phase 9 schema additions on Property and PropertyImage.

These run against an in-memory SQLite database — they do NOT exercise the
Alembic migration; they verify the SQLAlchemy ORM definitions can persist and
retrieve the new columns. Migration tests live in tests/integration.

The Phase 9 deltas added (all nullable):
  Property: slug, listing_type, price, currency, price_period, num_bedrooms,
            num_bathrooms, area_sqm, property_type, furnishing, available_from,
            locality, postal_code, country_code, latitude, longitude, owner_email
  PropertyImage: alt_text, caption, is_primary, display_order
"""

from collections.abc import Generator
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base, Property, PropertyImage


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


class TestPropertyPhase9Fields:
    def test_all_new_fields_default_to_none(self, db_session: Session) -> None:
        prop = Property(name="Bare bones")
        db_session.add(prop)
        db_session.commit()

        assert prop.slug is None
        assert prop.listing_type is None
        assert prop.price is None
        assert prop.currency is None
        assert prop.price_period is None
        assert prop.num_bedrooms is None
        assert prop.num_bathrooms is None
        assert prop.area_sqm is None
        assert prop.property_type is None
        assert prop.furnishing is None
        assert prop.available_from is None
        assert prop.locality is None
        assert prop.postal_code is None
        assert prop.country_code is None
        assert prop.latitude is None
        assert prop.longitude is None
        assert prop.owner_email is None

    def test_round_trip_full_listing(self, db_session: Session) -> None:
        prop = Property(
            name="Frankfurt 3BHK",
            slug="frankfurt-3bhk-abcdef",
            listing_type="rent",
            price=Decimal("1500.00"),
            currency="EUR",
            price_period="monthly",
            num_bedrooms=3,
            num_bathrooms=2,
            area_sqm=Decimal("82.50"),
            property_type="apartment",
            furnishing="semi_furnished",
            available_from=date(2026, 6, 1),
            locality="Sachsenhausen",
            postal_code="60594",
            country_code="DE",
            latitude=Decimal("50.105200"),
            longitude=Decimal("8.681300"),
            owner_email="owner@example.com",
        )
        db_session.add(prop)
        db_session.commit()

        retrieved = db_session.get(Property, prop.id)
        assert retrieved is not None
        assert retrieved.slug == "frankfurt-3bhk-abcdef"
        assert retrieved.listing_type == "rent"
        assert retrieved.price == Decimal("1500.00")
        assert retrieved.currency == "EUR"
        assert retrieved.price_period == "monthly"
        assert retrieved.num_bedrooms == 3
        assert retrieved.num_bathrooms == 2
        assert retrieved.area_sqm == Decimal("82.50")
        assert retrieved.property_type == "apartment"
        assert retrieved.furnishing == "semi_furnished"
        assert retrieved.available_from == date(2026, 6, 1)
        assert retrieved.locality == "Sachsenhausen"
        assert retrieved.postal_code == "60594"
        assert retrieved.country_code == "DE"
        assert retrieved.latitude == Decimal("50.105200")
        assert retrieved.longitude == Decimal("8.681300")
        assert retrieved.owner_email == "owner@example.com"

    def test_slug_unique_constraint(self, db_session: Session) -> None:
        p1 = Property(name="A", slug="dup-slug")
        p2 = Property(name="B", slug="dup-slug")
        db_session.add_all([p1, p2])
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestPropertyImagePhase9Fields:
    def _make_property(self, db_session: Session) -> Property:
        prop = Property(name="Img Parent")
        db_session.add(prop)
        db_session.flush()
        return prop

    def test_image_new_fields_default_correctly(self, db_session: Session) -> None:
        prop = self._make_property(db_session)
        img = PropertyImage(property_id=prop.id, file_path="p/i.jpg")
        db_session.add(img)
        db_session.commit()

        assert img.alt_text is None
        assert img.caption is None
        assert img.is_primary is False
        assert img.display_order == 0

    def test_image_round_trip_with_seo_fields(self, db_session: Session) -> None:
        prop = self._make_property(db_session)
        img = PropertyImage(
            property_id=prop.id,
            file_path="p/kitchen.jpg",
            alt_text="Kitchen with stainless steel appliances",
            caption="Renovated 2025",
            is_primary=True,
            display_order=1,
        )
        db_session.add(img)
        db_session.commit()

        retrieved = db_session.get(PropertyImage, img.id)
        assert retrieved is not None
        assert retrieved.alt_text == "Kitchen with stainless steel appliances"
        assert retrieved.caption == "Renovated 2025"
        assert retrieved.is_primary is True
        assert retrieved.display_order == 1
