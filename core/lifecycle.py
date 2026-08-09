"""Transition rules for the property publication lifecycle.

The wizard does not tell the server what state a listing is in — the server
derives it from the stages that actually completed, so a status can never
claim more than what happened. Two rules keep that honest:

  * a status only ever moves forward (a re-run of an earlier stage does not
    demote a listing that already got further), and
  * a published listing is never demoted by a later edit or a failed stage.

``failed`` and ``partially_completed`` sit outside the forward ordering: they
describe a stage that did not finish, and a retry resumes from them.
"""

from db.models import Property
from db.status import PropertyStatus

# Forward ordering of the happy path. Statuses outside this map (failed,
# partially_completed) are treated as "back at the start" so a retry resumes.
_ORDER: dict[str, int] = {
    PropertyStatus.DRAFT: 0,
    PropertyStatus.PROCESSING: 1,
    PropertyStatus.READY_FOR_REVIEW: 2,
    PropertyStatus.COMPLETED: 3,
    PropertyStatus.PUBLISHED: 4,
}


def advance_to(prop: Property, target: str) -> bool:
    """Move ``prop`` forward to ``target``; return whether the status changed.

    Raises:
        ValueError: if ``target`` is not part of the forward lifecycle.
    """
    if target not in _ORDER:
        raise ValueError(f"'{target}' is not a forward lifecycle status.")
    if prop.status == PropertyStatus.PUBLISHED:
        return False
    if _ORDER.get(prop.status, 0) >= _ORDER[target]:
        return False
    prop.status = target
    return True


def mark_stage_failed(prop: Property, *, has_usable_work: bool) -> None:
    """Record that a stage failed.

    ``has_usable_work`` distinguishes a listing that still has something to
    show (some images processed before the failure) from one that has nothing.
    A published listing is left untouched — a failed re-run must not take a
    live listing off the public surfaces.
    """
    if prop.status == PropertyStatus.PUBLISHED:
        return
    prop.status = PropertyStatus.PARTIALLY_COMPLETED if has_usable_work else PropertyStatus.FAILED


def can_publish(prop: Property) -> tuple[bool, str]:
    """Whether ``prop`` is complete enough to go public, and why not if it isn't.

    Readiness is about content, not about which status the row currently
    carries: a listing with no photo or no description is not a listing.
    """
    if not prop.images:
        return False, "A listing needs at least one image before it can be published."
    if prop.description is None or not prop.description.strip():
        return False, "A listing needs a description before it can be published."
    return True, ""
