"""Free-order chain search: pool contents, greedy construction,
max-nodes cap, stop conditions, and the CLI search/deliver path."""

import sys

import numpy as np
import pytest

from app.core.chain_search import default_pool, search_chain
from app.core.chromogen import (
    ColourSaturationStage,
    ContrastBoostStage,
    NeutralTintStage,
)
from app.core.stages import LiftGammaGainStage


def _with(stage, **by_name):
    p = stage.identity().copy()
    for name, value in by_name.items():
        p[stage.param_names.index(name)] = value
    return p


def _source(n=250, seed=5):
    return np.random.default_rng(seed).uniform(0.08, 0.9, (n, 3))


def _simple_look(x):
    """A look made from two pool tools: global sat boost + contrast."""
    sat = ColourSaturationStage()
    con = ContrastBoostStage()
    out = sat.apply(x, _with(sat, **{"R/G": 1.35, "Y/B": 1.35}))
    return con.apply(out, _with(con, **{"Contrast Boost": 0.5}))


def test_default_pool_is_chromogen_without_lgg():
    pool = default_pool()
    assert LiftGammaGainStage not in pool
    names = {cls.name for cls in pool}
    assert "Brilliance Reduction" in names
    assert "Neutral Tint" in names
    assert len(pool) == 10


def test_search_recovers_simple_look_and_logs():
    x = _source()
    result = search_chain(x, _simple_look(x), max_nodes=4, min_gain=0.01)
    assert result.error_after < result.error_before / 10
    assert 1 <= len(result.model.stages) <= 4
    assert all(not isinstance(s, LiftGammaGainStage)
               for s in result.model.stages)
    assert result.search_log
    assert isinstance(result.search_log[-1], str)  # stop reason logged


def test_search_respects_max_nodes():
    x = _source()
    tint = NeutralTintStage()
    target = _simple_look(
        tint.apply(x, _with(tint, Hue=220.0, Amount=-0.5))
    )
    result = search_chain(x, target, max_nodes=2, min_gain=0.001)
    assert len(result.model.stages) == 2
    assert "max_nodes" in result.search_log[-1]


def test_search_allows_reusing_a_stage_type():
    """Two sat boosts compose to more than one slider-max can do —
    the search must be free to pick the same tool twice."""
    x = _source()
    sat = ColourSaturationStage()
    p = _with(sat, **{"R/G": 1.9, "Y/B": 1.9})
    target = sat.apply(sat.apply(x, p), p)  # ~3.6x, beyond the 0..2 range
    result = search_chain(x, target, max_nodes=3, min_gain=0.005)
    names = [s.name for s in result.model.stages]
    assert names.count("Colour Saturation") >= 2
    assert result.error_after < result.error_before / 5


def test_search_display_domain_analytic_drt_finds_contrast():
    """The genesis lesson: with the analytic DRT and a display-domain
    loss, a contrasty look must surface Contrast Boost — no cube
    inversion deleting the tone evidence at the extremes."""
    from app.core.opendrt import OpenDRTModel

    drt = OpenDRTModel()
    x = _source(300)
    con = ContrastBoostStage()
    target = drt(con.apply(x, _with(con, **{"Contrast Boost": 0.6})))
    result = search_chain(x, target, max_nodes=3, min_gain=0.005,
                          display_transform=drt)
    names = [s.name for s in result.model.stages]
    assert "Contrast Boost" in names
    assert result.error_after < result.error_before / 5
    assert result.pairs_unreachable == 0  # nothing is ever dropped


def test_grey_locked_tone_matches_neutrals_exactly():
    """Marc: 'contrast adjusted based on grey scale only'. The tone
    node is fitted on neutrals, frozen, and no second Contrast Boost
    enters the chain — so the grey-scale match is exact and stays."""
    from app.core.opendrt import OpenDRTModel

    drt = OpenDRTModel()
    neutrals = np.linspace(0.02, 0.95, 40)[:, None].repeat(3, axis=1)
    x = np.concatenate([_source(250), neutrals])
    con = ContrastBoostStage()
    target = drt(con.apply(x, _with(con, **{"Contrast Boost": 0.7})))

    result = search_chain(x, target, max_nodes=4, min_gain=0.01,
                          display_transform=drt)
    assert isinstance(result.model.stages[0], ContrastBoostStage)
    names = [s.name for s in result.model.stages]
    assert names.count("Contrast Boost") == 1  # left the pool after node 1
    mask = np.all(x == x[:, :1], axis=1)
    fitted_display = drt(result.model(x[mask]))
    np.testing.assert_allclose(fitted_display, target[mask], atol=5e-3)


def test_search_refuses_identity_target():
    x = _source()
    with pytest.raises(ValueError, match="nothing worth adding"):
        search_chain(x, x.copy(), max_nodes=3)


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
