"""Unit tests for the property lifecycle transition rules.

The rules exist so a status can never claim more than what actually happened:
it moves forward as the wizard completes stages, never backwards, and a
published listing is never demoted by a later edit.
"""

import pytest

from core.lifecycle import advance_to, can_publish, mark_stage_failed
from db.models import Property, PropertyImage
from db.status import PropertyStatus


def _prop(status: str = PropertyStatus.DRAFT, **kwargs: object) -> Property:
    return Property(name="Test", status=status, **kwargs)  # type: ignore[arg-type]


class TestAdvanceTo:
    def test_moves_forward_through_the_lifecycle(self) -> None:
        prop = _prop()
        assert advance_to(prop, PropertyStatus.PROCESSING) is True
        assert prop.status == PropertyStatus.PROCESSING
        assert advance_to(prop, PropertyStatus.COMPLETED) is True
        assert prop.status == PropertyStatus.COMPLETED

    def test_never_regresses(self) -> None:
        prop = _prop(PropertyStatus.COMPLETED)
        assert advance_to(prop, PropertyStatus.PROCESSING) is False
        assert prop.status == PropertyStatus.COMPLETED

    def test_published_is_never_demoted_by_a_later_edit(self) -> None:
        prop = _prop(PropertyStatus.PUBLISHED)
        assert advance_to(prop, PropertyStatus.COMPLETED) is False
        assert prop.status == PropertyStatus.PUBLISHED

    def test_a_failed_run_can_resume(self) -> None:
        prop = _prop(PropertyStatus.FAILED)
        assert advance_to(prop, PropertyStatus.PROCESSING) is True
        assert prop.status == PropertyStatus.PROCESSING

    def test_rejects_a_status_outside_the_forward_lifecycle(self) -> None:
        with pytest.raises(ValueError):
            advance_to(_prop(), PropertyStatus.FAILED)


class TestMarkStageFailed:
    def test_no_usable_work_yet_is_a_plain_failure(self) -> None:
        prop = _prop(PropertyStatus.PROCESSING)
        mark_stage_failed(prop, has_usable_work=False)
        assert prop.status == PropertyStatus.FAILED

    def test_partial_work_is_recorded_as_partially_completed(self) -> None:
        prop = _prop(PropertyStatus.PROCESSING)
        mark_stage_failed(prop, has_usable_work=True)
        assert prop.status == PropertyStatus.PARTIALLY_COMPLETED

    def test_a_published_listing_is_left_alone(self) -> None:
        prop = _prop(PropertyStatus.PUBLISHED)
        mark_stage_failed(prop, has_usable_work=False)
        assert prop.status == PropertyStatus.PUBLISHED


class TestCanPublish:
    def test_needs_a_description(self) -> None:
        prop = _prop(PropertyStatus.COMPLETED, description="   ")
        prop.images = [PropertyImage(file_path="p/i.jpg")]
        ok, reason = can_publish(prop)
        assert ok is False
        assert "description" in reason.lower()

    def test_needs_at_least_one_image(self) -> None:
        prop = _prop(PropertyStatus.COMPLETED, description="A nice flat.")
        ok, reason = can_publish(prop)
        assert ok is False
        assert "image" in reason.lower()

    def test_ready_listing_can_publish(self) -> None:
        prop = _prop(PropertyStatus.COMPLETED, description="A nice flat.")
        prop.images = [PropertyImage(file_path="p/i.jpg")]
        ok, reason = can_publish(prop)
        assert ok is True
        assert reason == ""
