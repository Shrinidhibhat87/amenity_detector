"""Load Gradio static assets for the frontend revamp."""

from pathlib import Path

_STATIC_DIR = Path(__file__).parent / "static"


def _read(name: str) -> str:
    return (_STATIC_DIR / name).read_text(encoding="utf-8")


CSS = _read("styles.css")
JS = _read("pills.js")
HEAD = f"<script>{JS}</script>"
