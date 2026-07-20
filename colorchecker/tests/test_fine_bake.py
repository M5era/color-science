"""Reuleaux Fine bake CLI: cube matches the stage, units match the DCTL."""

import sys

import numpy as np

from app.core.lut import apply_lut, parse_cube
from app.core.stages import ReuleauxFineStage
from tools import reuleaux_fine_bake


def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["reuleaux_fine_bake"] + argv)
    reuleaux_fine_bake.main()


def test_baked_cube_matches_stage_at_lattice(tmp_path, monkeypatch):
    out = tmp_path / "fine.cube"
    _run(
        ["--out", str(out), "--size", "9",
         "--hue-center", "20", "--hue-core", "18", "--hue-soft", "30",
         "--hue-shift", "6", "--sat", "1.4", "--val", "0.3",
         "--luma-mask", "0.4", "0.3", "0.2"],
        monkeypatch,
    )
    lut = parse_cube(out)
    assert lut.size == 9

    stage = ReuleauxFineStage()
    params = stage.identity()
    params[0:4] = np.array([20, 18, 30, 6]) / 360.0
    params[4], params[5] = 1.4, 0.3
    params[6:9] = [0.4, 0.3, 0.2]

    # at exact lattice points apply_lut returns the stored values —
    # they must equal the stage output up to the cube's text precision
    grid = np.linspace(0.0, 1.0, 9)
    pts = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"),
                   axis=-1).reshape(-1, 3)
    np.testing.assert_allclose(
        apply_lut(lut, pts), stage.apply(pts, params), atol=2e-9
    )


def test_default_bake_is_identity(tmp_path, monkeypatch):
    out = tmp_path / "identity.cube"
    _run(["--out", str(out), "--size", "7"], monkeypatch)
    lut = parse_cube(out)
    grid = np.linspace(0.05, 0.95, 5)
    pts = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"),
                   axis=-1).reshape(-1, 3)
    np.testing.assert_allclose(apply_lut(lut, pts), pts, atol=1e-6)
