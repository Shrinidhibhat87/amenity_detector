"""Unit tests for the publication-status column on Property.

A property is private until someone publishes it. The column carries that
state, so the important guarantees are that a freshly created row starts as a
draft and that every status in the lifecycle round-trips through the database.

These run against in-memory SQLite and exercise the ORM definition only; the
migration itself is covered in tests/integration/test_alembic_upgrade.py.
"""

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base, Property
from db.status import PropertyStatus


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


class TestPropertyStatusColumn:
    def test_new_property_starts_as_draft(self, db_session: Session) -> None:
        prop = Property(name="Unfinished")
        db_session.add(prop)
        db_session.commit()

        assert prop.status == PropertyStatus.DRAFT

    def test_every_lifecycle_status_round_trips(self, db_session: Session) -> None:
        props = [Property(name=f"P {s}", status=s) for s in PropertyStatus.all()]
        db_session.add_all(props)
        db_session.commit()

        for prop, expected in zip(props, PropertyStatus.all(), strict=True):
            retrieved = db_session.get(Property, prop.id)
            assert retrieved is not None
            assert retrieved.status == expected

    def test_status_values_cover_the_documented_lifecycle(self) -> None:
        assert PropertyStatus.all() == (
            "draft",
            "processing",
            "ready_for_review",
            "completed",
            "published",
            "failed",
            "partially_completed",
        )
