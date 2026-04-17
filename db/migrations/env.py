"""
Alembic environment configuration.

This file is executed by Alembic every time you run an alembic command.
Its main job is to:
  1. Set up the database URL (reading from the environment so CI/CD works the same as local)
  2. Import our ORM models' metadata so Alembic can detect schema changes automatically
  3. Support both "online" mode (real DB connection) and "offline" mode (generate SQL scripts)

Why import Base.metadata here?
  When you run `alembic revision --autogenerate`, Alembic compares the current DB schema
  against what's defined in our SQLAlchemy models. It finds the models by inspecting
  `target_metadata`. Without this import, autogenerate produces empty migrations.
"""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

# Import Base so Alembic can see all mapped tables.
# This MUST happen before `target_metadata = Base.metadata` below.
from db.models import Base

# Alembic provides a Config object from alembic.ini
config = context.config

# Set up Python logging from alembic.ini's [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Tell Alembic which tables to track — Base.metadata includes all models
# that inherit from Base (Property, PropertyImage, DetectedAmenity)
target_metadata = Base.metadata


def get_url() -> str:
    """
    Return the database URL, preferring the DATABASE_URL environment variable.

    This means the same alembic commands work in all environments:
      - Local dev: reads from .env (set DATABASE_URL=postgresql://...)
      - Docker Compose: env var is set by docker-compose.yml
      - CI: env var is set in the GitHub Actions workflow
    """
    return os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url", "")


def run_migrations_offline() -> None:
    """
    Run migrations without a real database connection (generates SQL scripts).

    Useful for: reviewing what SQL will be run, or generating SQL for DBAs to review
    before applying. Run with: alembic upgrade head --sql > migration.sql
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations with a live database connection (the normal case).

    Creates a real connection to the database, then runs any pending migrations.
    """
    connectable = create_engine(get_url())
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


# Alembic sets context.is_offline_mode() based on the --sql flag
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
