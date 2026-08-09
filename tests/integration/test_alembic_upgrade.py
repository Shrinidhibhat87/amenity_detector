"""Integration tests for the Alembic migration chain.

These tests apply the migrations to a fresh, file-backed SQLite database (an
in-memory database doesn't survive Alembic's per-step engine recycling) and
assert that:

  - ``upgrade head`` produces every table and column the current ORM expects.
  - The Phase 9 partial index exists.
  - Existing pre-Phase-9 rows survive the 0001 -> 0002 upgrade with NULLs in
    the new columns and the boolean/integer defaults applied.
  - ``downgrade base`` walks back to an empty database.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_alembic_config(database_url: str) -> Config:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "db" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture()
def fresh_db(tmp_path: Path) -> Iterator[str]:
    db_file = tmp_path / "alembic_test.db"
    yield f"sqlite:///{db_file}"


def test_upgrade_head_creates_full_phase9_schema(fresh_db: str) -> None:
    cfg = _make_alembic_config(fresh_db)
    command.upgrade(cfg, "head")

    engine = create_engine(fresh_db)
    inspector = inspect(engine)

    assert {"properties", "images", "detected_amenities"} <= set(inspector.get_table_names())

    property_columns = {col["name"]: col for col in inspector.get_columns("properties")}
    expected_property_columns = {
        # Phase 8 baseline
        "id",
        "name",
        "description",
        "model_used",
        "extra_info",
        "created_at",
        # Phase 9 additions
        "slug",
        "listing_type",
        "price",
        "currency",
        "price_period",
        "num_bedrooms",
        "num_bathrooms",
        "area_sqm",
        "property_type",
        "furnishing",
        "available_from",
        "locality",
        "postal_code",
        "country_code",
        "latitude",
        "longitude",
        "owner_email",
        # Hybrid search additions (0003)
        "description_embedding",
    }
    # status (0006) is deliberately NOT in the set above: unlike the Phase 9
    # metadata it is NOT NULL, so it fails the nullability assertion below.
    assert "status" in property_columns
    assert property_columns["status"]["nullable"] is False
    assert expected_property_columns <= set(property_columns)

    # All Phase 9 property columns must be nullable so legacy rows survive.
    for name in expected_property_columns - {"id", "name", "created_at"}:
        assert property_columns[name]["nullable"], f"{name} should be nullable"

    image_columns = {col["name"]: col for col in inspector.get_columns("images")}
    assert {"alt_text", "caption", "is_primary", "display_order"} <= set(image_columns)
    # Defaults for existing rows on ALTER must come through as server_default.
    assert image_columns["is_primary"]["nullable"] is False
    assert image_columns["display_order"]["nullable"] is False

    # Slug must be unique. We add a named unique index in 0002.
    slug_indexes = [
        idx
        for idx in inspector.get_indexes("properties")
        if idx["name"] == "ix_properties_slug_unique"
    ]
    assert slug_indexes and slug_indexes[0].get("unique"), "slug must have a unique index"

    # The Phase 11-hot-path partial index lives on detected_amenities.
    amenity_indexes = {idx["name"] for idx in inspector.get_indexes("detected_amenities")}
    assert "ix_detected_amenities_present_room_amenity" in amenity_indexes


def test_existing_rows_survive_upgrade_with_null_phase9_fields(fresh_db: str) -> None:
    cfg = _make_alembic_config(fresh_db)

    # Stop at the Phase 8 baseline, insert a legacy row, then upgrade to head.
    command.upgrade(cfg, "0001_phase8_baseline")
    engine = create_engine(fresh_db)
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO properties (id, name, created_at) VALUES (:id, :name, :ts)"),
            {"id": "legacy-id-0001", "name": "Legacy", "ts": "2026-01-01 00:00:00"},
        )
        conn.execute(
            text("INSERT INTO images (id, property_id, file_path) VALUES (:id, :pid, :p)"),
            {"id": "legacy-img-0001", "pid": "legacy-id-0001", "p": "x/y.jpg"},
        )

    command.upgrade(cfg, "head")

    with engine.connect() as conn:
        prop_row = conn.execute(
            text(
                "SELECT slug, listing_type, num_bedrooms, latitude FROM properties WHERE id = :id"
            ),
            {"id": "legacy-id-0001"},
        ).one()
        assert prop_row.slug is None
        assert prop_row.listing_type is None
        assert prop_row.num_bedrooms is None
        assert prop_row.latitude is None

        img_row = conn.execute(
            text("SELECT alt_text, caption, is_primary, display_order FROM images WHERE id = :id"),
            {"id": "legacy-img-0001"},
        ).one()
        assert img_row.alt_text is None
        assert img_row.caption is None
        # Server defaults backfill on ALTER.
        assert int(img_row.is_primary) == 0
        assert int(img_row.display_order) == 0

        embed_row = conn.execute(
            text("SELECT description_embedding FROM properties WHERE id = :id"),
            {"id": "legacy-id-0001"},
        ).one()
        assert embed_row.description_embedding is None

        # A row that was already public before the lifecycle existed stays
        # public — 0006 backfills it to 'published' rather than 'draft'.
        status_row = conn.execute(
            text("SELECT status FROM properties WHERE id = :id"),
            {"id": "legacy-id-0001"},
        ).one()
        assert status_row.status == "published"


def test_downgrade_chain_back_to_empty(fresh_db: str) -> None:
    cfg = _make_alembic_config(fresh_db)
    command.upgrade(cfg, "head")

    command.downgrade(cfg, "base")

    engine = create_engine(fresh_db)
    inspector = inspect(engine)
    # Alembic keeps its own ``alembic_version`` bookkeeping table — that's
    # fine; what matters is none of our domain tables remain.
    domain_tables = set(inspector.get_table_names()) - {"alembic_version"}
    assert domain_tables == set()
