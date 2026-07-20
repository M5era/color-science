"""Display conversion is one-way and never mutates the raw buffer."""

import numpy as np

from app.core.preview import to_display_u8


def test_clamps_scene_referred_range():
    pixels = np.array([[[-0.5, 0.0, 0.5], [1.0, 2.0, 18.0]]], dtype=np.float32)
    display = to_display_u8(pixels)
    assert display.dtype == np.uint8
    assert display.tolist() == [[[0, 0, 128], [255, 255, 255]]]


def test_source_buffer_untouched():
    pixels = np.array([[[-1.0, 0.3, 5.0]]], dtype=np.float32)
    original = pixels.copy()
    to_display_u8(pixels)
    np.testing.assert_array_equal(pixels, original)
