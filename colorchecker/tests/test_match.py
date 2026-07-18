"""Match core: RBF + matrix fitting, validation, cube export, CSV input."""

import numpy as np
import pytest

from app.core.match import (
    load_patch_csv,
    session_patch_rows,
    solve_match,
    write_cube,
)
from app.core.project import ImageEntry, ProjectStore


def _film_like_pairs(n=500, seed=2):
    rng = np.random.default_rng(seed)
    source = rng.uniform(0.02, 0.98, (n, 3))
    matrix = np.array([[0.9, 0.08, 0.02], [0.05, 0.85, 0.10], [0.02, 0.12, 0.86]])
    target = (source @ matrix.T) ** 1.08 + 0.04 * np.sin(source * 2.5)
    return source, target


def test_matrix_plus_rbf_fits_film_curve():
    source, target = _film_like_pairs()
    result = solve_match(source, target, use_matrix=True, layers=6)
    assert result.error_after < 0.002
    assert result.error_after < result.error_before
    # Matrix stage alone helps but can't capture the nonlinearity fully.
    assert result.error_matrix is not None
    assert result.error_after < result.error_matrix


def test_pure_matrix_when_layers_zero():
    source, _ = _film_like_pairs(200)
    matrix = np.array([[1.1, 0.05, 0.0], [0.0, 0.9, 0.1], [0.05, 0.0, 1.0]])
    target = source @ matrix.T
    result = solve_match(source, target, use_matrix=True, layers=0)
    assert result.model.rbf is None
    np.testing.assert_allclose(result.model.matrix, matrix, atol=1e-10)
    assert result.error_after < 1e-12


def test_out_of_domain_extrapolates_not_clips():
    source, target = _film_like_pairs(300)
    result = solve_match(source, target, layers=4)
    bright = result.model(np.array([[1.8, 1.5, 1.4]]))
    assert bright[0].max() > 1.05  # extrapolated, not stuck at a cube wall


def test_strength_blends_toward_identity():
    source, target = _film_like_pairs(200)
    full = solve_match(source, target, layers=3, strength=1.0)
    off = solve_match(source, target, layers=3, strength=0.0)
    probe = np.array([[0.4, 0.5, 0.6]])
    np.testing.assert_allclose(off.model(probe), probe, atol=1e-12)
    half = solve_match(source, target, layers=3, strength=0.5)
    np.testing.assert_allclose(
        half.model(probe), (probe + full.model(probe)) / 2.0, atol=1e-9
    )


def test_nan_pairs_dropped_and_counted():
    source, target = _film_like_pairs(100)
    source[3] = np.nan
    target[7, 1] = np.nan
    result = solve_match(source, target, layers=2)
    assert result.pairs_dropped == 2
    assert result.pairs_used == 98


def test_length_mismatch_raises():
    source, target = _film_like_pairs(50)
    with pytest.raises(ValueError, match="pair up row for row"):
        solve_match(source[:40], target)


def test_cube_export(tmp_path):
    source, target = _film_like_pairs(200)
    result = solve_match(source, target, layers=3)
    path = tmp_path / "match.cube"
    write_cube(result.model, path, size=17, domain_min=0.0, domain_max=1.2)

    lines = path.read_text().splitlines()
    assert lines[1] == "LUT_3D_SIZE 17"
    assert lines[3] == "DOMAIN_MAX 1.2 1.2 1.2"
    data = [ln for ln in lines if ln and not ln[0].isalpha() and not ln.startswith('"')]
    assert len(data) == 17**3
    # First entry is the model at the domain origin (R fastest ordering).
    first = np.array([float(v) for v in data[0].split()])
    np.testing.assert_allclose(first, result.model(np.zeros((1, 3)))[0], atol=1e-9)


def test_load_rich_csv(tmp_path):
    path = tmp_path / "patches.csv"
    path.write_text(
        "label,ev,group,overlay,kind,patch_row,patch_col,R,G,B\n"
        '"800T, tungsten",0,5600K,Overlay 1,reflective,1,1,0.1,0.2,0.3\n'
        "800T,1,5600K,Overlay 2,emissive,1,1,1.8,1.2,1.1\n"
    )
    values, labels = load_patch_csv(path)
    np.testing.assert_array_equal(values, [[0.1, 0.2, 0.3], [1.8, 1.2, 1.1]])
    assert labels[0] == "800T, tungsten"


def test_load_headerless_csv(tmp_path):
    path = tmp_path / "bare.txt"
    path.write_text("0.1 0.2 0.3\n0.4 0.5 0.6\n")
    values, labels = load_patch_csv(path)
    assert values.shape == (2, 3)
    assert values[1][2] == 0.6


def test_session_rows_follow_export_order():
    store = ProjectStore(
        images=[
            ImageEntry(
                source_path="/a.tif", label="A", include=True,
                patch_results=[
                    {"row": 1, "col": 1, "rgb": [0.1, 0.1, 0.1], "overlay": "Overlay 1"},
                    {"row": 1, "col": 2, "rgb": [0.2, 0.2, 0.2], "overlay": "Overlay 1"},
                ],
            ),
            ImageEntry(source_path="/b.tif", label="B", include=False,
                       patch_results=[{"row": 1, "col": 1, "rgb": [9, 9, 9]}]),
            ImageEntry(
                source_path="/c.tif", label="C", include=True,
                patch_results=[{"row": 1, "col": 1, "rgb": [0.3, 0.3, 0.3]}],
            ),
        ]
    )
    values, labels = session_patch_rows(store)
    assert values.shape == (3, 3)  # B excluded
    assert values[2][0] == 0.3
    assert labels[0] == "A [Overlay 1] r1c1"


def test_zero_width_domain_rejected(tmp_path):
    source, target = _film_like_pairs(100)
    result = solve_match(source, target, layers=2)
    with pytest.raises(ValueError, match="collapses the whole LUT"):
        write_cube(result.model, tmp_path / "bad.cube", domain_min=1.0, domain_max=1.0)
