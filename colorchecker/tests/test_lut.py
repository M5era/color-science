"""Cube parsing and inspection sampling."""

import numpy as np
import pytest

from app.core.lut import apply_lut, lattice_points, neutral_curves, parse_cube, reference_gradient
from app.core.match import MatchModel, write_cube


def _identity_model():
    return MatchModel(matrix=None, rbf=None, strength=1.0)


def test_roundtrip_with_our_writer(tmp_path):
    path = tmp_path / "id.cube"
    write_cube(_identity_model(), path, size=9, domain_min=0.0, domain_max=1.0)
    lut = parse_cube(path)
    assert lut.size == 9
    probe = np.array([[0.2, 0.5, 0.8], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(apply_lut(lut, probe), probe, atol=1e-9)


def test_domain_respected(tmp_path):
    path = tmp_path / "dom.cube"
    write_cube(_identity_model(), path, size=9, domain_min=0.0, domain_max=2.0)
    lut = parse_cube(path)
    probe = np.array([[1.5, 1.5, 1.5]])
    np.testing.assert_allclose(apply_lut(lut, probe), probe, atol=1e-9)
    # Outside the domain clamps to the edge, like a real LUT box.
    np.testing.assert_allclose(apply_lut(lut, np.array([[3.0, 3.0, 3.0]])),
                               [[2.0, 2.0, 2.0]], atol=1e-9)


def test_matrix_model_cube_matches_matrix(tmp_path):
    matrix = np.array([[0.8, 0.1, 0.1], [0.0, 1.0, 0.0], [0.05, 0.0, 0.95]])
    model = MatchModel(matrix=matrix, rbf=None)
    path = tmp_path / "m.cube"
    write_cube(model, path, size=17)
    lut = parse_cube(path)
    probe = np.array([[0.25, 0.5, 0.75]])
    np.testing.assert_allclose(apply_lut(lut, probe), probe @ matrix.T, atol=1e-3)


def test_neutral_curves_identity(tmp_path):
    path = tmp_path / "id.cube"
    write_cube(_identity_model(), path, size=9)
    lut = parse_cube(path)
    inputs, outputs = neutral_curves(lut, samples=64)
    np.testing.assert_allclose(outputs, np.stack([inputs] * 3, axis=1), atol=1e-9)


def test_lattice_points_shapes(tmp_path):
    path = tmp_path / "id.cube"
    write_cube(_identity_model(), path, size=17)
    lut = parse_cube(path)
    inputs, outputs = lattice_points(lut, resolution=9)
    assert inputs.shape == (9**3, 3)
    assert outputs.shape == (9**3, 3)
    np.testing.assert_allclose(inputs, outputs, atol=1e-9)  # identity LUT


def test_bad_cube_rejected(tmp_path):
    path = tmp_path / "bad.cube"
    path.write_text("LUT_3D_SIZE 3\n0 0 0\n1 1 1\n")
    with pytest.raises(ValueError, match="Expected 27"):
        parse_cube(path)


def test_reference_gradient_shape_and_range():
    img = reference_gradient(300, 200)
    assert img.shape == (200, 300, 3)
    assert img.min() >= 0.0 and img.max() <= 1.0
    # White at top, black at bottom, saturated colors in the middle.
    assert img[0].mean() > 0.95
    assert img[-1].mean() < 0.05
    assert img[100].std() > 0.2
