"""Phase 0 risk proof: float TIFF values survive loading bit-exact.

Fixtures are tiny synthetic TIFFs generated at runtime — no real footage
is ever committed or required.
"""

import numpy as np
import tifffile

from app.core.image_io import (
    load_image,
    neighbor_image,
    parse_ev_from_filename,
    sibling_images,
)


def _write_float_tiff(path, values):
    tifffile.imwrite(path, values.astype(np.float32))
    return path


def test_float32_roundtrip_bit_exact(tmp_path):
    # Scene-referred values: negatives, >1.0, tiny fractions — none may change.
    rng = np.random.default_rng(42)
    values = (rng.standard_normal((48, 64, 3)) * 4.0).astype(np.float32)
    values[0, 0] = [-0.25, 0.0, 18.5]
    values[10, 10] = [1.0000001, 0.3333333, 100.0]

    path = _write_float_tiff(tmp_path / "chart.tif", values)
    loaded = load_image(path)

    assert loaded.pixels.dtype == np.float32
    assert loaded.pixels.shape == (48, 64, 3)
    np.testing.assert_array_equal(loaded.pixels, values)  # bit-exact, no tolerance


def test_16bit_integer_normalized_exactly(tmp_path):
    values = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
    tifffile.imwrite(tmp_path / "int.tif", values)
    loaded = load_image(tmp_path / "int.tif")
    np.testing.assert_array_equal(
        loaded.pixels, np.array([[[0.0, 32768 / 65535, 1.0]]], dtype=np.float32)
    )


def test_alpha_channel_dropped(tmp_path):
    values = np.ones((4, 4, 4), dtype=np.float32)
    values[..., 3] = 0.5
    tifffile.imwrite(tmp_path / "rgba.tif", values)
    loaded = load_image(tmp_path / "rgba.tif")
    assert loaded.pixels.shape == (4, 4, 3)
    np.testing.assert_array_equal(loaded.pixels, values[..., :3])


def test_grayscale_expanded_to_rgb(tmp_path):
    values = np.full((4, 4), 2.5, dtype=np.float32)
    tifffile.imwrite(tmp_path / "gray.tif", values)
    loaded = load_image(tmp_path / "gray.tif")
    assert loaded.pixels.shape == (4, 4, 3)
    assert float(loaded.pixels[0, 0, 2]) == 2.5


def test_folder_navigation(tmp_path):
    ones = np.ones((2, 2, 3), dtype=np.float32)
    for name in ["b.tif", "a.tif", "c.tiff"]:
        _write_float_tiff(tmp_path / name, ones)
    (tmp_path / "notes.txt").write_text("ignored")

    names = [p.name for p in sibling_images(tmp_path / "a.tif")]
    assert names == ["a.tif", "b.tif", "c.tiff"]

    assert neighbor_image(tmp_path / "a.tif", +1).name == "b.tif"
    assert neighbor_image(tmp_path / "b.tif", -1).name == "a.tif"
    assert neighbor_image(tmp_path / "a.tif", -1) is None
    assert neighbor_image(tmp_path / "c.tiff", +1) is None


def test_ev_parsing():
    assert parse_ev_from_filename("0_EV_v1-800T_MatRem.tif") == 0.0
    assert parse_ev_from_filename("800T_+3EV.tif") == 3.0
    assert parse_ev_from_filename("shot_-5EV_take2.tif") == -5.0
    assert parse_ev_from_filename("stock_2.5EV.tif") == 2.5
    assert parse_ev_from_filename("stock_+1,5EV.tif") == 1.5
    assert parse_ev_from_filename("no_marker_here.tif") is None
