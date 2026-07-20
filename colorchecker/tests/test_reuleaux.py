"""Reuleaux port: coordinate round-trips and DCTL-faithful behavior."""

import numpy as np
import pytest

from app.core.reuleaux import (
    ReuleauxUserParams,
    reuleaux_to_rgb,
    reuleaux_user,
    rgb_to_reuleaux,
)


def _random_rgb(n=2000, lo=0.02, hi=1.4, seed=5):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, (n, 3))


def test_coordinate_roundtrip():
    rgb = _random_rgb()
    back = reuleaux_to_rgb(rgb_to_reuleaux(rgb))
    np.testing.assert_allclose(back, rgb, atol=1e-9)


def test_neutral_axis_maps_to_zero_sat():
    grays = np.stack([np.linspace(0.01, 1.5, 20)] * 3, axis=1)
    coords = rgb_to_reuleaux(grays)
    np.testing.assert_allclose(coords[:, 1], 0.0, atol=1e-12)  # sat
    np.testing.assert_allclose(coords[:, 2], grays[:, 0], atol=1e-12)  # val = max


def test_default_params_are_identity():
    rgb = _random_rgb()
    out = reuleaux_user(rgb, ReuleauxUserParams())
    np.testing.assert_allclose(out, rgb, atol=1e-9)


def test_grays_untouched_by_any_sliders():
    grays = np.stack([np.linspace(0.05, 1.0, 10)] * 3, axis=1)
    params = ReuleauxUserParams(
        overall_sat=1.7, overall_val=2.0,
        red=(0.1, 1.8, -1.0), green=(-0.1, 0.4, 2.0), blue=(0.15, 0.2, 1.5),
    )
    out = reuleaux_user(grays, params)
    np.testing.assert_allclose(out, grays, atol=1e-9)


def test_forward_inverse_roundtrip():
    rgb = _random_rgb(500, 0.1, 0.9)
    params = ReuleauxUserParams(
        overall_sat=1.2, red=(0.05, 1.3, 0.2), cyan=(-0.04, 0.8, -0.3)
    )
    forward = reuleaux_user(rgb, params)
    back = reuleaux_user(forward, params, invert=True)
    np.testing.assert_allclose(back, rgb, atol=2e-3)


def test_hue_slider_rotates_only_its_zone():
    # A pure red pixel and a pure cyan pixel; rotate only the red vector.
    red = np.array([[0.8, 0.1, 0.1]])
    cyan = np.array([[0.1, 0.8, 0.8]])
    params = ReuleauxUserParams(red=(0.08, 1.0, 0.0))
    red_out = reuleaux_user(red, params)
    cyan_out = reuleaux_user(cyan, params)
    assert np.abs(red_out - red).max() > 1e-3   # red moved
    np.testing.assert_allclose(cyan_out, cyan, atol=1e-9)  # cyan untouched


def test_bake_cli(tmp_path):
    import subprocess, sys
    out = tmp_path / "r.cube"
    result = subprocess.run(
        [sys.executable, "-m", "tools.reuleaux_bake", "--out", str(out),
         "--size", "17", "--red", "0.05", "1.2", "0.1"],
        capture_output=True, text=True, cwd=".",
    )
    assert result.returncode == 0, result.stderr
    from app.core.lut import parse_cube, apply_lut
    lut = parse_cube(out)
    assert lut.size == 17
    # Cube output matches the direct python transform.
    probe = np.array([[0.4, 0.3, 0.25]])
    params = ReuleauxUserParams(red=(0.05, 1.2, 0.1))
    np.testing.assert_allclose(
        apply_lut(lut, probe), reuleaux_user(probe, params), atol=2e-3
    )
