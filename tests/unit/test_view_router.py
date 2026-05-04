"""Unit tests for the UI page-state reducer."""

from ui.view_router import BROWSE, HOME, UPLOAD, go_to


def test_go_to_home_makes_only_home_visible() -> None:
    home, upload, browse, state = go_to(HOME)

    assert home["visible"] is True
    assert upload["visible"] is False
    assert browse["visible"] is False
    assert state == HOME


def test_go_to_upload_makes_only_upload_visible() -> None:
    home, upload, browse, state = go_to(UPLOAD)

    assert upload["visible"] is True
    assert home["visible"] is False
    assert browse["visible"] is False
    assert state == UPLOAD


def test_go_to_browse_makes_only_browse_visible() -> None:
    home, upload, browse, state = go_to(BROWSE)

    assert browse["visible"] is True
    assert home["visible"] is False
    assert upload["visible"] is False
    assert state == BROWSE


def test_go_to_unknown_page_falls_back_to_home() -> None:
    home, upload, browse, state = go_to("not-a-real-page")

    assert home["visible"] is True
    assert upload["visible"] is False
    assert browse["visible"] is False
    assert state == HOME
