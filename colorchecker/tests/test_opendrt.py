"""OpenDRT analytic port: the validation gate against Marc's
Resolve-baked cube, plus basic sanity."""

from pathlib import Path

import numpy as np
import pytest

from app.core.lut import parse_cube
from app.core.opendrt import MARC_CONFIG, OpenDRTModel, opendrt_transform

_CUBE = (Path(__file__).resolve().parents[1] / "test_luts"
         / "openDRT_LogC3_srgb_3.A001C001_260326_RPYL.cube")


@pytest.mark.skipif(not _CUBE.exists(), reason="baked cube not in repo")
def test_port_matches_resolve_baked_cube():
    """THE gate: the port must reproduce what Marc's Resolve renders.
    The cube stores float32 on a 65^3 lattice, so agreement is bounded
    by its quantization — the port sits at ~1e-5."""
    lut = parse_cube(_CUBE)
    grid = np.linspace(float(lut.domain_min[0]), float(lut.domain_max[0]),
                       lut.size)
    b, g, r = np.meshgrid(grid, grid, grid, indexing="ij")
    pts = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    baked = lut.table.reshape(-1, 3)

    port = opendrt_transform(pts)
    err = np.abs(port - baked)
    assert err.mean() < 5e-5
    assert err.max() < 5e-4


def test_neutral_axis_monotone_and_in_range():
    ramp = np.linspace(0.0, 1.0, 512)[:, None].repeat(3, axis=1)
    out = opendrt_transform(ramp)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # neutrals stay neutral (Rec.709 D65 path) and rise monotonically —
    # equality is bounded by the P3->XYZ->Rec709 matrix roundtrip
    # (row sums differ from 1.0 by ~1e-7, same as the DCTL)
    np.testing.assert_allclose(out[:, 0], out[:, 1], atol=1e-5)
    np.testing.assert_allclose(out[:, 1], out[:, 2], atol=1e-5)
    # monotone up to sub-black wiggle: inputs below LogC3's zero point
    # map to ~1e-6 outputs with ~4e-8 local dips — real DRT behavior
    assert (np.diff(out[:, 1]) >= -1e-6).all()


def test_model_wrapper_matches_function():
    x = np.random.default_rng(3).uniform(0.0, 1.0, (100, 3))
    np.testing.assert_array_equal(OpenDRTModel(MARC_CONFIG)(x),
                                  opendrt_transform(x))
