"""
test_convert_annotations.py — Unit tests for the BBBC041 annotation converter.

Tests the core conversion logic without requiring the actual dataset.
Run: pytest tests/ -v
"""

from src.data.convert_annotations import (
    convert_bbox_to_yolo,
    normalise_category,
    CLASS_MAP,
)


# ---------------------------------------------------------------------------
# Bounding Box Conversion Tests
# ---------------------------------------------------------------------------
class TestConvertBboxToYolo:
    """Test the pixel-to-normalised YOLO coordinate conversion."""

    def test_full_image_box(self):
        """A box covering the entire image should yield (0.5, 0.5, 1.0, 1.0)."""
        x_c, y_c, w, h = convert_bbox_to_yolo(0, 0, 100, 200, 100, 200)
        assert abs(x_c - 0.5) < 1e-6
        assert abs(y_c - 0.5) < 1e-6
        assert abs(w - 1.0) < 1e-6
        assert abs(h - 1.0) < 1e-6

    def test_quarter_box(self):
        """Top-left quarter of a 100x100 image."""
        x_c, y_c, w, h = convert_bbox_to_yolo(0, 0, 50, 50, 100, 100)
        assert abs(x_c - 0.25) < 1e-6
        assert abs(y_c - 0.25) < 1e-6
        assert abs(w - 0.5) < 1e-6
        assert abs(h - 0.5) < 1e-6

    def test_clamping_overflow(self):
        """Boxes that extend beyond image boundaries should be clamped."""
        x_c, y_c, w, h = convert_bbox_to_yolo(-10, -10, 110, 110, 100, 100)
        # After clamping: (0, 0, 100, 100) → full image
        assert abs(x_c - 0.5) < 1e-6
        assert abs(y_c - 0.5) < 1e-6
        assert abs(w - 1.0) < 1e-6
        assert abs(h - 1.0) < 1e-6

    def test_small_box(self):
        """A 10×10 box at (50,50) in a 1000×1000 image."""
        x_c, y_c, w, h = convert_bbox_to_yolo(50, 50, 60, 60, 1000, 1000)
        assert abs(x_c - 0.055) < 1e-6
        assert abs(y_c - 0.055) < 1e-6
        assert abs(w - 0.01) < 1e-6
        assert abs(h - 0.01) < 1e-6

    def test_zero_area_box(self):
        """Degenerate box (zero width) should produce zero dimensions."""
        x_c, y_c, w, h = convert_bbox_to_yolo(50, 50, 50, 60, 100, 100)
        assert w == 0.0  # Zero width → should be filtered downstream


# ---------------------------------------------------------------------------
# Category Normalisation Tests
# ---------------------------------------------------------------------------
class TestNormaliseCategory:
    """Test the annotation category string normalisation."""

    def test_canonical_names(self):
        """Canonical class names should pass through unchanged."""
        for name in CLASS_MAP.keys():
            assert normalise_category(name) == name

    def test_case_insensitive(self):
        """Category matching should be case-insensitive."""
        assert normalise_category("Red Blood Cell") == "red blood cell"
        assert normalise_category("RING") == "ring"
        assert normalise_category("Trophozoite") == "trophozoite"

    def test_aliases(self):
        """Known aliases should map to canonical names."""
        assert normalise_category("rbc") == "red blood cell"
        assert normalise_category("red_blood_cell") == "red blood cell"

    def test_skip_categories(self):
        """Categories marked for skipping should return None."""
        assert normalise_category("difficult") is None
        assert normalise_category("leukocyte") is None

    def test_whitespace_handling(self):
        """Leading/trailing whitespace should be stripped."""
        assert normalise_category("  ring  ") == "ring"
        assert normalise_category("\ttrophozoite\n") == "trophozoite"

    def test_unknown_category(self):
        """Unknown categories pass through (caught downstream)."""
        result = normalise_category("unknown_thing")
        assert result == "unknown_thing"
