"""
test_preprocess.py — Unit tests for the image preprocessing pipeline.

Tests operate on synthetically generated images so they don't require
the BBBC041 dataset to be downloaded.
"""

import numpy as np

from src.preprocessing.preprocess import (
    enhance_contrast,
    normalize_stain,
    resize_with_padding,
)


class TestResizeWithPadding:
    """Test the letterbox resize function."""

    def test_square_image(self):
        """A square image should resize without padding."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = resize_with_padding(img, target_size=640)
        assert result.shape == (640, 640, 3)

    def test_landscape_image(self):
        """A wide image should get top/bottom padding."""
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = resize_with_padding(img, target_size=640)
        assert result.shape == (640, 640, 3)

    def test_portrait_image(self):
        """A tall image should get left/right padding."""
        img = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        result = resize_with_padding(img, target_size=640)
        assert result.shape == (640, 640, 3)

    def test_already_target_size(self):
        """An image already at target size should pass through cleanly."""
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = resize_with_padding(img, target_size=640)
        assert result.shape == (640, 640, 3)

    def test_custom_target_size(self):
        """Test with a non-default target size (1024)."""
        img = np.random.randint(0, 255, (300, 500, 3), dtype=np.uint8)
        result = resize_with_padding(img, target_size=1024)
        assert result.shape == (1024, 1024, 3)

    def test_pad_colour(self):
        """Padding areas should use the specified colour."""
        # A 1x2 pixel image scaled to 10 → top and bottom padding
        img = np.array([[[0, 0, 255], [0, 0, 255]]], dtype=np.uint8)  # Red pixels
        result = resize_with_padding(img, target_size=10, pad_color=(114, 114, 114))
        assert result.shape == (10, 10, 3)
        # Corners should be pad colour
        assert list(result[0, 0]) == [114, 114, 114]


class TestNormalizeStain:
    """Test stain normalisation."""

    def test_output_shape(self):
        """Output should have the same shape as input."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = normalize_stain(img)
        assert result.shape == img.shape

    def test_output_dtype(self):
        """Output should be uint8."""
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = normalize_stain(img)
        assert result.dtype == np.uint8

    def test_uniform_image_no_crash(self):
        """A uniform (single-colour) image shouldn't cause division by zero."""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        result = normalize_stain(img)  # Should not raise
        assert result.shape == (50, 50, 3)


class TestEnhanceContrast:
    """Test CLAHE contrast enhancement."""

    def test_output_shape(self):
        """CLAHE should preserve image dimensions."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = enhance_contrast(img)
        assert result.shape == img.shape

    def test_output_dtype(self):
        """Output should be uint8."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = enhance_contrast(img)
        assert result.dtype == np.uint8

    def test_low_contrast_enhanced(self):
        """A low-contrast image should have higher variance after CLAHE."""
        # Create a very flat grey image with minimal contrast
        img = np.random.randint(120, 130, (100, 100, 3), dtype=np.uint8)
        result = enhance_contrast(img, clip_limit=3.0)
        # The enhanced image should have greater pixel variance
        assert result.std() >= img.std()
