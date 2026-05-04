"""Pure reducer for the Gradio three-page state machine."""

HOME = "home"
UPLOAD = "upload"
BROWSE = "browse"

_VALID = {HOME, UPLOAD, BROWSE}


def go_to(page: str) -> tuple[dict[str, bool], dict[str, bool], dict[str, bool], str]:
    """Return page visibility updates plus the normalized page state."""
    if page not in _VALID:
        page = HOME
    return (
        {"visible": page == HOME},
        {"visible": page == UPLOAD},
        {"visible": page == BROWSE},
        page,
    )
