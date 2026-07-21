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
