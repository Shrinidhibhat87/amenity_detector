"""Stepper HTML component."""

STEPS = ("Config", "Upload", "Detect", "Review")


def step_html(active_step: int) -> str:
    """Render the upload flow stepper."""
    parts: list[str] = ['<div class="stepper">']
    for idx, label in enumerate(STEPS):
        state = " active" if idx <= active_step else ""
        parts.append(
            f'<div class="step{state}"><span class="step-num">{idx + 1}</span>'
            f"<span>{label}</span></div>"
        )
        if idx < len(STEPS) - 1:
            parts.append(f'<span class="step-line{state}"></span>')
    parts.append("</div>")
    return "".join(parts)
