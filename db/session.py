"""
Database session factory and FastAPI dependency.

How it works:
  1. `create_engine()` sets up the connection pool using DATABASE_URL from the environment.
  2. `SessionLocal` is a session factory — calling SessionLocal() gives you a session object.
  3. `get_db()` is a FastAPI dependency that yields a session per HTTP request and ensures
     the session is always closed when the request is done (even on error).

Environment variables:
  DATABASE_URL — PostgreSQL connection string.
                 Format: postgresql://user:password@host:port/database
                 Example (local Docker): postgresql://amenity_user:amenity_pass@localhost:5432/amenity_db
                 Falls back to SQLite for local development without Docker.
                 NEVER use SQLite in production — it doesn't support concurrent writes.

Running migrations:
  After changing db/models.py, generate a migration:
      alembic revision --autogenerate -m "describe your change"
  Then apply it:
      alembic upgrade head
  See alembic.ini and db/migrations/env.py for configuration.
"""

import logging
import os
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base

logger = logging.getLogger(__name__)

# Read the database URL from the environment.
# Default to SQLite for development convenience (no Docker needed for a quick test).
# SQLite works for single-developer use; PostgreSQL is required for any multi-user setup.
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "sqlite:///./amenity_detector.db"
)

# SQLite needs check_same_thread=False because FastAPI may handle a request across
# different threads. PostgreSQL doesn't need this flag so we only add it conditionally.
_connect_args: dict[str, bool] = (
    {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    # echo=True logs every SQL statement — handy for debugging, too noisy for production.
    # Switch on temporarily if you want to see what queries are running.
    echo=False,
)

# SessionLocal is a class (not an instance). Each call to SessionLocal() creates a new session.
# autocommit=False means we control transactions manually (session.commit() / session.rollback()).
# autoflush=False means SQLAlchemy won't silently flush before queries — we control this too.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables() -> None:
    """
    Create all database tables defined in ORM models.

    Use this for local development and test setup. In production, prefer
    Alembic migrations (alembic upgrade head) so schema changes are tracked.

    This is idempotent — calling it multiple times is safe (CREATE IF NOT EXISTS).
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created (or already exist).")


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session for a single HTTP request.

    Usage in a router:
        @router.get("/properties")
        def list_properties(db: Session = Depends(get_db)):
            ...

    The `yield` makes this a context manager:
      - Code before `yield` runs at the start of the request (session created).
      - Code after `yield` runs at the end (session closed), even if an exception occurred.

    Why not just use a global session?
      Sessions are not thread-safe and should not be shared across requests.
      Each request gets its own session with its own transaction.

    Yields:
        An active SQLAlchemy Session object.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    """
    Test whether the database is reachable.

    Used by the /health endpoint to report DB status.

    Returns:
        True if the database responds, False otherwise.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning("Database health check failed: %s", e)
        return False