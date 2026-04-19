"""
Image preprocessing utilities.

One public function: preprocess_image. Called from AmenityDetector.detect_from_image
before the VLM call so every code path (API, tests, CLI) uses the same preprocessing.

Why only resize, why not filters:
  - Resize to 768 px on the longest edge is the biggest free speedup for VLM inference.
    Most vision models tokenize the image at fixed patch sizes and do not benefit beyond
    768 px on the input.
  - EXIF auto-orient + RGB conversion are deferred (see SPEC.md Open Questions).
    Sharpening/denoising is actively harmful for VLM quality.
"""

from PIL.Image import Image, Resampling


def preprocess_image(img: Image, max_edge: int = 768) -> Image:
    """
    Resize so the longest edge is at most ``max_edge`` pixels. Aspect ratio preserved.

    Images already smaller than ``max_edge`` on every side are returned unchanged
    (no upscaling — upscaling just invents detail that can mislead the VLM).

    Args:
        img:      The PIL image to process. Not mutated.
        max_edge: Maximum length of the longest edge. Default 768 — matches the
                  SPEC recommendation and is the sweet spot for Qwen2.5-VL / Gemini.

    Returns:
        A resized copy of the input image. If no resize was needed, a plain copy
        is returned so callers can always treat the result as independent.
    """
    width, height = img.size
    longest = max(width, height)
    if longest <= max_edge:
        return img.copy()

    scale = max_edge / longest
    new_size = (round(width * scale), round(height * scale))
    return img.resize(new_size, Resampling.LANCZOS)
