"""LUT matching (Plan C item 1), the LGG prep stage's strong identity
prior, and the noise-gain artifact KPI."""

import numpy as np
import pytest

from app.core.chromogen import ColourSaturationStage, ContrastBoostStage
from app.core.diagnostics import noise_gain
from app.core.lut import parse_cube
from app.core.lut_match import sample_lut_domain, solve_lut_match
from app.core.match import write_cube
from app.core.parametric import solve_parametric
from app.core.stages import CHAIN_PRESETS, STAGE_POOL, LiftGammaGainStage


def _with(stage, **by_name):
    p = stage.identity().copy()
    for name, value in by_name.items():
        p[stage.param_names.index(name)] = value
    return p


def _chromogen_look_cube(tmp_path, size=17):
    """A known chromogen-style look baked to a .cube."""
    sat = ColourSaturationStage()
    con = ContrastBoostStage()
    p_sat = _with(sat, **{"R/G": 1.15, "Y/B": 1.45, "Chroma": -0.3})
    p_con = _with(con, **{"Contrast Boost": 0.5})

    def look(rgb):
        return con.apply(sat.apply(rgb, p_sat), p_con)

    path = tmp_path / "look.cube"
    write_cube(look, path, size=size)
    return parse_cube(path), look


# ------------------------------------------------------------ sampling

def test_sample_covers_domain_and_neutrals():
    lut, _ = _chromogen_look_cube(pytest.importorskip("pathlib").Path("/tmp"))
    pts = sample_lut_domain(lut, n=1000, seed=3)
    assert pts.min() >= 0.0 and pts.max() <= 1.0
    neutral = pts[np.all(pts == pts[:, :1], axis=1)]
    assert len(neutral) >= 64  # the neutral ramp is always included


# ---------------------------------------------------------- lut match

def test_lut_match_recovers_chromogen_look(tmp_path):
    lut, look = _chromogen_look_cube(tmp_path)
    stages = [STAGE_POOL[n]() for n in
              CHAIN_PRESETS["Chromogen match (LGG prep → Chromogen chain)"]]
    result = solve_lut_match(lut, stages, n_samples=800)

    assert result.error_after < result.error_before / 10
    assert result.error_after < 0.01
    # the KPI ships with the result
    assert len(result.stage_noise_gain) == len(stages)
    assert result.chain_noise_gain["median"] > 0.0

    # the strongly-anchored prep stage stayed home: the look contains
    # no exposure/WB change, so LGG must remain ~identity
    lgg = result.model.params[0]
    np.testing.assert_allclose(lgg, LiftGammaGainStage().identity(),
                               atol=0.05)


def test_lgg_prior_yields_when_target_is_exposure(tmp_path):
    """A target that IS an exposure/WB change: now the prep stage must
    do the work (nothing else in the chain can express a neutral gain
    cleanly), prior or not."""
    rng = np.random.default_rng(8)
    x = rng.uniform(0.1, 0.8, (600, 3))
    target = x * 1.25  # one third of a stop of linear gain, neutral

    stages = [LiftGammaGainStage(), ColourSaturationStage()]
    result = solve_parametric(x, target, stages)
    assert result.error_after < 5e-3
    gains = result.model.params[0][2:5]
    np.testing.assert_allclose(gains, 1.25, atol=0.1)


@pytest.mark.parametrize("backend", ["torch"])
def test_lut_match_through_backprop(tmp_path, backend):
    pytest.importorskip("torch")
    lut, _ = _chromogen_look_cube(tmp_path, size=13)
    stages = [STAGE_POOL[n]() for n in
              CHAIN_PRESETS["Chromogen match (LGG prep → Chromogen chain)"]]
    result = solve_lut_match(lut, stages, n_samples=400, backend=backend)
    assert result.backend == "torch"
    assert result.error_after < result.error_before / 10


# ------------------------------------------------------------ the KPI

def test_noise_gain_flags_amplifiers():
    x = np.random.default_rng(4).uniform(0.1, 0.9, (400, 3))
    identity = noise_gain(lambda v: v, x)
    assert abs(identity["median"] - 1.0) < 1e-9

    doubler = noise_gain(lambda v: v * 2.0, x)
    assert abs(doubler["median"] - 2.0) < 1e-9

    # the old sector-sat failure mode: power law near zero saturation
    stage = ColourSaturationStage()
    strong = _with(stage, **{"R/G": 3.0, "Y/B": 3.0})
    boosted = noise_gain(lambda v: stage.apply(v, strong), x)
    neutral = noise_gain(lambda v: stage.apply(v, stage.identity()), x)
    assert boosted["median"] > neutral["median"] * 1.3


def test_stage_labels_read_like_grading_notes():
    from app.core.chromogen import (NeutralTintStage, SectorSkewStage,
                                    SectorSquashStage)

    tint = NeutralTintStage()
    p = tint.identity(); p[0], p[1] = 220.0, -0.3
    assert tint.label(p) == "cool lows"
    p[0], p[1] = 40.0, 0.3
    assert tint.label(p) == "warm highs"

    skew = SectorSkewStage()
    p = skew.identity(); p[0], p[1], p[3] = 120.0, 20.0, -0.5
    assert skew.label(p) == "skew dark greens toward cyan"

    squash = SectorSquashStage()
    p = squash.identity(); p[0], p[1] = 10.0, 0.7
    assert squash.label(p) == "squash reds"

    lgg = LiftGammaGainStage()
    assert lgg.label(lgg.identity()) == "prep (idle)"
    p = lgg.identity(); p[2:5] = [1.3, 1.3, 1.3]
    assert "exposure" in lgg.label(p)


def test_cli_lut_match(tmp_path, monkeypatch, capsys):
    import sys

    from tools import lut_match as cli

    lut_path = tmp_path / "look.cube"
    lut, look = _chromogen_look_cube(tmp_path)
    out = tmp_path / "fitted.cube"
    monkeypatch.setattr(sys, "argv", [
        "lut_match", "--lut", str(tmp_path / "look.cube"),
        "--samples", "600", "--out", str(out),
    ])
    cli.main()
    text = capsys.readouterr().out
    assert "noise gain" in text
    assert "paste into" in text
    assert out.exists()
    fitted = parse_cube(out)
    assert fitted.size == 33
