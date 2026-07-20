"""Torch backend: mirror parity with the numpy stages, gradient flow,
and the capability scipy lacks — *finding* a distant Fine zone."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.core.parametric import solve_parametric
from app.core.stages import (
    LinearMatrixStage,
    LumaCurveStage,
    ReuleauxBroadStage,
    ReuleauxFineStage,
    RGBCurvesStage,
    STAGE_POOL,
)
from app.core.torch_stages import torch_apply, torch_chain


def _source(n=500, seed=3):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.08, 0.9, (n, 3))


def _random_params(stage, rng):
    lo, hi = stage.bounds()
    # stay off the exact bounds; extreme reuleaux params can push spow
    # into regions where float noise dominates
    return lo + (hi - lo) * rng.uniform(0.25, 0.75, lo.size)


def test_torch_mirrors_match_numpy_stages():
    rng = np.random.default_rng(11)
    x = _source(400)
    x_t = torch.as_tensor(x, dtype=torch.float64)
    for name, cls in STAGE_POOL.items():
        stage = cls()
        for trial in range(3):
            params = _random_params(stage, rng)
            expected = stage.apply(x, params)
            got = torch_apply(
                stage, x_t, torch.as_tensor(params, dtype=torch.float64)
            ).numpy()
            np.testing.assert_allclose(
                got, expected, atol=1e-9,
                err_msg=f"{name} trial {trial}",
            )


def test_torch_mirrors_match_at_identity():
    x = _source(200)
    x_t = torch.as_tensor(x, dtype=torch.float64)
    for name, cls in STAGE_POOL.items():
        stage = cls()
        got = torch_apply(
            stage, x_t, torch.as_tensor(stage.identity(), dtype=torch.float64)
        ).numpy()
        np.testing.assert_allclose(got, x, atol=1e-9, err_msg=name)


def test_gradients_flow_and_are_finite():
    x = torch.as_tensor(_source(100), dtype=torch.float64)
    # include grays and an emissive value — the numerically nasty inputs
    extra = torch.tensor(
        [[0.18, 0.18, 0.18], [0.0, 0.0, 0.0], [1.4, 1.3, 1.2]],
        dtype=torch.float64,
    )
    x = torch.cat([x, extra])
    stages = [LinearMatrixStage(), LumaCurveStage(5), RGBCurvesStage(5),
              ReuleauxBroadStage(), ReuleauxFineStage()]
    params = [
        torch.tensor(s.identity(), dtype=torch.float64, requires_grad=True)
        for s in stages
    ]
    out = torch_chain(stages, x, params)
    out.square().mean().backward()
    for stage, p in zip(stages, params):
        assert p.grad is not None, stage.name
        assert torch.isfinite(p.grad).all(), stage.name


def test_torch_backend_finds_distant_zone():
    """The placement test: a Fine zone in the greens, far from the
    identity start at red. Scipy's finite differences see zero gradient
    (no window overlap) and cannot move the zone; the torch backend's
    restarts must find it."""
    x = _source(700)
    fine = ReuleauxFineStage()
    p_true = fine.identity().copy()
    #        center  flat  soft  shift  sat
    p_true[[0, 1, 2, 3, 4]] = [1.0 / 3.0, 0.07, 0.10, 0.03, 1.5]
    target = fine.apply(x, p_true)

    scipy_result = solve_parametric(x, target, [ReuleauxFineStage()])
    torch_result = solve_parametric(
        x, target, [ReuleauxFineStage()], backend="torch"
    )

    assert torch_result.backend == "torch"
    # scipy stays parked (zone never found); torch must crack it
    assert torch_result.error_after < scipy_result.error_after / 3
    assert torch_result.error_after < torch_result.error_before / 10


def test_torch_backend_matches_scipy_on_global_stages():
    """No Fine stage -> no placement problem; torch must not be worse
    than scipy alone (it feeds the same joint refine)."""
    x = _source(400)
    matrix = np.array([[0.96, 0.03, 0.01], [0.02, 0.95, 0.03], [0.01, 0.04, 0.95]])
    target = x @ matrix.T
    scipy_result = solve_parametric(x, target, [LinearMatrixStage()])
    torch_result = solve_parametric(
        x, target, [LinearMatrixStage()], backend="torch"
    )
    assert torch_result.error_after <= scipy_result.error_after * 1.5 + 1e-9
    assert torch_result.error_after < 1e-6


def test_unknown_backend_rejected():
    x = _source(50)
    with pytest.raises(ValueError, match="Unknown backend"):
        solve_parametric(x, x, [LinearMatrixStage()], backend="numpy")
