"""Publication lifecycle statuses for a property.

A property moves draft -> processing -> ready_for_review -> completed ->
published as the wizard progresses, with ``failed`` and
``partially_completed`` as terminal-ish outcomes of a stage that did not
finish cleanly. Only ``published`` rows are visible on the public surfaces
(browse, search, sitemap, llms.txt, JSONL feed).

The values are plain strings rather than a database ENUM for the same reason
the Phase 9 enum-like columns are: SQLite (tests) and PostgreSQL (production)
then behave identically, and adding a status later is a code change rather
than a type migration.
"""

from typing import Final


class PropertyStatus:
    """Namespace of the allowed ``properties.status`` values."""

    DRAFT: Final = "draft"
    PROCESSING: Final = "processing"
    READY_FOR_REVIEW: Final = "ready_for_review"
    COMPLETED: Final = "completed"
    PUBLISHED: Final = "published"
    FAILED: Final = "failed"
    PARTIALLY_COMPLETED: Final = "partially_completed"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        """Every allowed status, in lifecycle order."""
        return (
            cls.DRAFT,
            cls.PROCESSING,
            cls.READY_FOR_REVIEW,
            cls.COMPLETED,
            cls.PUBLISHED,
            cls.FAILED,
            cls.PARTIALLY_COMPLETED,
        )
