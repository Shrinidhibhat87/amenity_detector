"""Animated background canvas markup (golden bokeh orbs via JS)."""

ORBS_HTML = """
<canvas id="bg-canvas" aria-hidden="true"></canvas>
"""


def bg_orbs_html() -> str:
    """Return the canvas element that pills.js animates with bokeh orbs."""
    return ORBS_HTML
