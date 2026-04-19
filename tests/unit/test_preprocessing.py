"""
Unit tests for core/preprocessing.py.

Preprocessing is the only step that runs on every image before it reaches the VLM,
so these tests lock in exactly how that resize behaves.
"""

from PIL import Image as PILImage

from core.preprocessing import preprocess_image


class TestPreprocessImage:
    def test_resizes_wide_image_longest_edge_to_max(self) -> None:
        """A 2000x1000 image should come back 768x384 (longest edge capped)."""
        img = PILImage.new("RGB", (2000, 1000), color=(0, 0, 0))
        out = preprocess_image(img, max_edge=768)
        assert out.size == (768, 384)

    def test_resizes_tall_image_longest_edge_to_max(self) -> None:
        """A 1000x2000 image should come back 384x768."""
        img = PILImage.new("RGB", (1000, 2000), color=(0, 0, 0))
        out = preprocess_image(img, max_edge=768)
        assert out.size == (384, 768)

    def test_square_image_resizes_to_max_x_max(self) -> None:
        img = PILImage.new("RGB", (2000, 2000), color=(0, 0, 0))
        out = preprocess_image(img, max_edge=768)
        assert out.size == (768, 768)

    def test_small_image_is_not_upscaled(self) -> None:
        """Images already smaller than max_edge pass through unchanged."""
        img = PILImage.new("RGB", (400, 300), color=(0, 0, 0))
        out = preprocess_image(img, max_edge=768)
        assert out.size == (400, 300)

    def test_preserves_image_mode(self) -> None:
        img = PILImage.new("RGB", (2000, 2000), color=(12, 34, 56))
        out = preprocess_image(img)
        assert out.mode == "RGB"

    def test_default_max_edge_is_768(self) -> None:
        img = PILImage.new("RGB", (4000, 2000), color=(0, 0, 0))
        out = preprocess_image(img)  # no max_edge arg
        assert max(out.size) == 768

    def test_returns_new_image_does_not_mutate_input(self) -> None:
        """preprocess_image must not mutate the original image object."""
        img = PILImage.new("RGB", (2000, 2000), color=(0, 0, 0))
        original_size = img.size
        _ = preprocess_image(img)
        assert img.size == original_size
