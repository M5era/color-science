"""Free-order chain search: pool contents, greedy construction,
max-nodes cap, stop conditions, and the CLI search/deliver path."""

import sys

import numpy as np
import pytest

from app.core.chain_search import default_pool, search_chain
from app.core.chromogen import (
    ColourSaturationStage,
    ContrastCurveStage,
    NeutralTintStage,
)
from app.core.filmic import FilmicContrastStage
from app.core.stages import LiftGammaGainStage


def _with(stage, **by_name):
    p = stage.identity().copy()
    for name, value in by_name.items():
        p[stage.param_names.index(name)] = value
    return p


def _source(n=250, seed=5):
    return np.random.default_rng(seed).uniform(0.08, 0.9, (n, 3))


def _simple_look(x):
    """A look made from two cleanly-separable broad pool tools: a global
    sat boost and a warm tint. (Contrast Curve is deliberately kept out
    of this fixture — in per-RGB mode it also moves saturation, so the
    recovery-mechanics tests would otherwise have two tools competing to
    explain the sat; Contrast Curve discovery is covered on its own by
    the display-domain tests below.)"""
    sat = ColourSaturationStage()
    tint = NeutralTintStage()
    out = sat.apply(x, _with(sat, **{"R/G": 1.35, "Y/B": 1.35}))
    return tint.apply(out, _with(tint, Hue=40.0, Amount=0.5))


def test_default_pool_is_chromogen_without_lgg_or_neutral_tint():
    pool = default_pool()
    assert LiftGammaGainStage not in pool
    names = {cls.name for cls in pool}
    assert "Brilliance Reduction" in names
    assert "Split Tone" in names
    assert "Neutral Tint" in names          # re-admitted after the falloff fix
    assert "Filmic Contrast" in names       # replaces Contrast Curve as tone
    assert "Contrast Curve" not in names    # retired from the ML audition
    assert len(pool) == 11


@pytest.mark.slow
def test_search_recovers_simple_look_and_logs():
    x = _source()
    result = search_chain(x, _simple_look(x), max_nodes=4, min_gain=0.01)
    # ~10x recovery; the exact figure moved a hair when Filmic Contrast
    # replaced Contrast Curve in the pool (different greedy trajectory)
    assert result.error_after < result.error_before / 8
    assert 1 <= len(result.model.stages) <= 4
    assert all(not isinstance(s, LiftGammaGainStage)
               for s in result.model.stages)
    assert result.search_log
    assert isinstance(result.search_log[-1], str)  # stop reason logged


@pytest.mark.slow
def test_search_respects_max_nodes():
    x = _source()
    tint = NeutralTintStage()
    target = _simple_look(
        tint.apply(x, _with(tint, Hue=220.0, Amount=-0.5))
    )
    result = search_chain(x, target, max_nodes=2, min_gain=0.001)
    assert len(result.model.stages) == 2
    assert "max_nodes" in result.search_log[-1]


@pytest.mark.slow
def test_search_allows_reusing_a_stage_type():
    """Two sat boosts compose to more than one slider-max can do —
    the search must be free to pick the same tool twice."""
    x = _source()
    sat = ColourSaturationStage()
    p = _with(sat, **{"R/G": 1.9, "Y/B": 1.9})
    target = sat.apply(sat.apply(x, p), p)  # ~3.6x, beyond the 0..2 range
    # keep the per-channel tools out of this fixture: Filmic (per-RGB
    # contrast) and Split Tone v3 (per-channel curves) both move channel
    # spread, so they'd compete to explain the sat boost and muddy this
    # pure reuse-mechanic check.
    from app.core.chromogen import SplitToneStage
    pool = [c for c in default_pool()
            if c is not FilmicContrastStage and c is not SplitToneStage]
    result = search_chain(x, target, max_nodes=3, min_gain=0.005, pool=pool)
    names = [s.name for s in result.model.stages]
    assert names.count("Colour Saturation") >= 2
    assert result.error_after < result.error_before / 5


@pytest.mark.slow
def test_search_display_domain_analytic_drt_finds_contrast():
    """The genesis lesson: with the analytic DRT and a display-domain
    loss, a contrasty look must surface the tone tool (now Filmic
    Contrast) — no cube inversion deleting the tone evidence."""
    from app.core.opendrt import OpenDRTModel

    drt = OpenDRTModel()
    x = _source(300)
    con = FilmicContrastStage()
    target = drt(con.apply(x, _with(con, Contrast=1.7)))
    result = search_chain(x, target, max_nodes=3, min_gain=0.005,
                          display_transform=drt)
    names = [s.name for s in result.model.stages]
    assert "Filmic Contrast" in names
    assert result.error_after < result.error_before / 5
    assert result.pairs_unreachable == 0  # nothing is ever dropped


@pytest.mark.slow
def test_grey_locked_tone_matches_neutrals_exactly():
    """Marc: 'contrast adjusted based on grey scale only'. The tone
    node is fitted on neutrals, frozen, and no second tone node
    enters the chain — so the grey-scale match is exact and stays."""
    from app.core.opendrt import OpenDRTModel

    drt = OpenDRTModel()
    neutrals = np.linspace(0.02, 0.95, 40)[:, None].repeat(3, axis=1)
    x = np.concatenate([_source(250), neutrals])
    con = FilmicContrastStage()
    target = drt(con.apply(x, _with(con, Contrast=1.7)))

    result = search_chain(x, target, max_nodes=4, min_gain=0.01,
                          display_transform=drt)
    assert isinstance(result.model.stages[0], FilmicContrastStage)
    names = [s.name for s in result.model.stages]
    assert names.count("Filmic Contrast") == 1  # left the pool after node 1
    mask = np.all(x == x[:, :1], axis=1)
    fitted_display = drt(result.model(x[mask]))
    np.testing.assert_allclose(fitted_display, target[mask], atol=5e-3)


@pytest.mark.slow
def test_local_search_drops_redundant_node():
    """A one-tool look with a permissive min_gain tempts the greedy build
    to bolt on marginal extra nodes; local_search's prune pass drops them
    back out (never lengthening the chain) while still explaining it."""
    x = _source()
    sat = ColourSaturationStage()
    target = sat.apply(x, _with(sat, **{"R/G": 1.4, "Y/B": 1.4}))
    greedy = search_chain(x, target, max_nodes=3, min_gain=1e-4,
                          neutral_tone=False)
    local = search_chain(x, target, max_nodes=3, min_gain=1e-4,
                         neutral_tone=False, local_search=True)
    assert len(local.model.stages) <= len(greedy.model.stages)
    assert local.error_after < local.error_before / 5


@pytest.mark.slow
def test_local_search_unfreezes_tone_and_prunes_on_tint():
    """A look whose NEUTRALS are tinted (crossover): the frozen tone can't
    tint, so local_search must un-freeze it to co-adapt with the Split
    Tone, and its noise-gain-aware prune should trim redundant nodes."""
    from app.core.chromogen import SplitToneStage
    from app.core.opendrt import OpenDRTModel

    drt = OpenDRTModel()
    neutrals = np.linspace(0.05, 0.95, 40)[:, None].repeat(3, axis=1)
    x = np.concatenate([_source(200), neutrals])
    con = FilmicContrastStage()
    tint = SplitToneStage()
    y = con.apply(x, _with(con, Contrast=1.6))
    # asymmetric per-channel shadow moves tint the neutral ramp
    y = tint.apply(y, _with(tint, **{"Black R": -0.4, "Black B": 0.3,
                                     "Dark R": 1.25, "Dark B": 0.8}))
    target = drt(y)

    greedy = search_chain(x, target, max_nodes=4, min_gain=0.005,
                          display_transform=drt)
    local = search_chain(x, target, max_nodes=4, min_gain=0.005,
                         display_transform=drt, local_search=True)
    # prune never lengthens the chain
    assert len(local.model.stages) <= len(greedy.model.stages)
    # the frozen tone node co-adapted with the tint (moved off its grey fit)
    assert not np.allclose(greedy.model.params[0], local.model.params[0],
                           atol=1e-4)
    # and the joint result is no worse than the locked-then-disturbed one
    assert local.error_after <= greedy.error_after * 1.2 + 1e-4


def test_search_refuses_identity_target():
    x = _source()
    with pytest.raises(ValueError, match="nothing worth adding"):
        search_chain(x, x.copy(), max_nodes=3)


@pytest.mark.slow
def test_cli_search_and_deliver(tmp_path, monkeypatch):
    from pathlib import Path

    from app.core.lut import parse_cube
    from tools import lut_match as cli
    from tools import stage_bake

    look = tmp_path / "look.cube"
    monkeypatch.setattr(sys, "argv", [
        "stage_bake", "--stage", "Colour Saturation",
        "--set", "R/G=1.4", "--set", "Y/B=1.4",
        "--out", str(look), "--size", "9",
    ])
    stage_bake.main()

    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "lut_match", "--lut", str(look), "--search",
        "--max-nodes", "2", "--samples", "200", "--deliver",
    ])
    cli.main()

    fitted = downloads / "look_fit.cube"
    assert fitted.exists()
    parse_cube(fitted)  # valid cube
    # the default template has ColourSaturation nodes, so the drx side
    # must have been written too
    assert (downloads / "look_fit.drx").exists()
    # the chain spec is persisted as crash insurance
    import json
    spec = json.loads((downloads / "look_fit.chain.json").read_text())
    assert spec["stages"] and len(spec["params"]) == len(spec["stages"])


@pytest.mark.slow
def test_search_broad_bias_prefers_broad_tools():
    """A global sat boost with a slight hue-local wrinkle: with a
    heavy bias the search must explain it with broad tools only."""
    x = _source(200)
    sat = ColourSaturationStage()
    target = sat.apply(x, _with(sat, **{"R/G": 1.5, "Y/B": 1.4}))
    result = search_chain(x, target, max_nodes=3, min_gain=0.005,
                          broad_bias=0.9)
    assert all(not s.local_tool for s in result.model.stages)
    assert result.error_after < result.error_before / 5


@pytest.mark.slow
def test_manual_pre_chain_is_hard_frozen():
    """Marc's workflow pivot: he dials Exposure/Filmic/Split himself and
    hands over the DCTL values; the search must apply them as an
    untouchable prefix, drop the tone tools from the pool, and explain
    only the color on top."""
    from app.core.filmic import ExposureStage

    x = _source()
    exp, fc, sat = (ExposureStage(), FilmicContrastStage(),
                    ColourSaturationStage())
    p_exp = _with(exp, Exposure=-0.3)
    p_fc = _with(fc, Contrast=1.5, **{"White Point": 0.9})
    y = fc.apply(exp.apply(x, p_exp), p_fc)
    target = sat.apply(y, _with(sat, **{"R/G": 1.4, "Y/B": 1.4}))

    res = search_chain(x, target, max_nodes=3, min_gain=0.005,
                       pre_chain=([exp, fc], [p_exp, p_fc]))
    names = [s.name for s in res.model.stages]
    assert names[:2] == ["Exposure", "Filmic Contrast"]
    np.testing.assert_array_equal(res.model.params[0], p_exp)  # bit-frozen
    np.testing.assert_array_equal(res.model.params[1], p_fc)
    assert not any(n in ("Filmic Contrast", "Split Tone", "Exposure",
                         "Contrast Curve") for n in names[2:])
    assert res.error_after < res.error_before / 5
