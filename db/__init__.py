"""
Database package for the amenity detector.

Contains:
  - models.py  : SQLAlchemy ORM table definitions (Property, PropertyImage, DetectedAmenity)
  - session.py : Engine + session factory + FastAPI dependency
  - migrations/: Alembic migration scripts (run `alembic upgrade head` to apply)
"""