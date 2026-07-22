"""Parametric solver: stage identities, monotonicity, recovery of
known synthetic transforms, subset chains, sandwich compatibility."""

import numpy as np
import pytest

from app.core.parametric import solve_parametric
from app.core.reuleaux import ReuleauxUserParams, reuleaux_user
from app.core.stages import (
    CHAIN_PRESETS,
    LinearMatrixStage,
    LumaCurveStage,
    ReuleauxBroadStage,
    ReuleauxFineStage,
    RGBCurvesStage,
    STAGE_POOL,
)


def _source(n=800, seed=4):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.08, 0.9, (n, 3))


def test_all_stage_identities_are_passthrough():
    x = _source(200)
    for name, cls in STAGE_POOL.items():
        stage = cls()
        np.testing.assert_allclose(stage.apply(x, stage.identity()), x,
                                   atol=1e-9, err_msg=name)


def test_curves_monotone_for_any_params_in_bounds():
    stage = LumaCurveStage(6)
    lo, hi = stage.bounds()
    rng = np.random.default_rng(9)
    ramp = np.linspace(-0.2, 1.2, 200)[:, None].repeat(3, axis=1)
    for _ in range(20):
        params = rng.uniform(lo, hi)
        out = stage.apply(ramp, params)
        assert (np.diff(out[:, 0]) >= -1e-12).all()


def test_recovers_pure_reuleaux_transform():
    x = _source()
    true = ReuleauxUserParams(overall_sat=1.15, red=(0.06, 1.25, 0.15),
                              blue=(-0.04, 0.85, 0.3))
    target = reuleaux_user(x, true)
    result = solve_parametric(x, target, [ReuleauxBroadStage()])
    assert result.error_after < 5e-4
    assert result.error_after < result.error_before / 20


def test_recovers_matrix_plus_reuleaux():
    x = _source()
    matrix = np.array([[0.95, 0.04, 0.01], [0.02, 0.94, 0.04], [0.01, 0.05, 0.94]])
    true = ReuleauxUserParams(green=(0.05, 1.2, -0.2))
    target = reuleaux_user(x @ matrix.T, true)
    result = solve_parametric(
        x, target, [LinearMatrixStage(), ReuleauxBroadStage()]
    )
    assert result.error_after < 1e-3
    # waterfall exists for both stages and improves overall
    assert len(result.waterfall) == 2
    assert result.waterfall[-1][1] <= result.waterfall[0][1] + 1e-9


@pytest.mark.slow
def test_full_chain_on_film_like_target():
    x = _source()
    # contrast + split tone + hue/sat character
    contrast = np.clip(0.5 + (x - 0.45) * 1.25, None, None) ** 1.02
    toned = contrast + np.array([0.015, -0.005, -0.02]) * (1.0 - contrast)
    target = reuleaux_user(toned, ReuleauxUserParams(red=(0.03, 1.15, 0.1)))

    stages = [LumaCurveStage(6), RGBCurvesStage(6), ReuleauxBroadStage()]
    result = solve_parametric(x, target, stages)
    assert result.error_after < 0.01
    assert result.error_after < result.error_before / 5
    assert len(result.stage_reports) == 3
    assert "paste into" in result.stage_reports[2]


def test_reordered_chain_runs():
    x = _source(300)
    target = x * 0.9 + 0.02
    stages = [RGBCurvesStage(5), LumaCurveStage(5)]  # user-swapped order
    result = solve_parametric(x, target, stages)
    assert result.error_after < 0.01


def test_empty_chain_rejected():
    x = _source(50)
    with pytest.raises(ValueError, match="chain is empty"):
        solve_parametric(x, x, [])


def test_strength_blend():
    x = _source(300)
    target = x @ np.diag([1.1, 0.95, 0.9])
    full = solve_parametric(x, target, [LinearMatrixStage()], strength=1.0)
    half = solve_parametric(x, target, [LinearMatrixStage()], strength=0.5)
    probe = np.array([[0.4, 0.4, 0.4]])
    np.testing.assert_allclose(
        half.model(probe), (probe + full.model(probe)) / 2.0, atol=1e-6
    )


@pytest.mark.slow
def test_parametric_sandwich_under_drt(tmp_path):
    from tests.test_match import _drt_cube
    from app.core.lut import apply_lut

    drt = _drt_cube(tmp_path)
    x = _source(400)
    true = ReuleauxUserParams(cyan=(-0.05, 1.3, 0.2))
    target = apply_lut(drt, reuleaux_user(x, true))

    result = solve_parametric(x, target, [ReuleauxBroadStage()], output_transform=drt)
    assert result.display_referred
    assert result.error_after < 0.01
    assert result.error_after < result.error_before


@pytest.mark.slow
def test_broad_plus_fine_recovers_local_zone():
    """Broad handles the global move; a Fine zone near red (overlapping
    the identity window, so the local solver can find it) mops up a
    local push the 6 fixed anchors can't express as sharply."""
    x = _source()
    fine_true = ReuleauxFineStage()
    p_true = fine_true.identity().copy()
    p_true[[0, 1, 2, 3, 4]] = [0.05, 0.08, 0.12, 0.02, 1.4]
    target = fine_true.apply(
        reuleaux_user(x, ReuleauxUserParams(overall_sat=1.1)), p_true
    )

    result = solve_parametric(
        x, target, [ReuleauxBroadStage(), ReuleauxFineStage()]
    )
    assert result.error_after < result.error_before / 10
    assert len(result.stage_reports) == 2
    assert "Reuleaux Fine zone" in result.stage_reports[1]


def test_presets_reference_pool():
    for preset, names in CHAIN_PRESETS.items():
        for name in names:
            assert name in STAGE_POOL, f"{preset} references unknown stage {name}"
