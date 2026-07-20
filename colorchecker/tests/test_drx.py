"""PowerGrade (.drx) template patching against Marc's example grade."""

from pathlib import Path

import pytest

_TEMPLATES = Path(__file__).resolve().parents[1] / "templates"
TEMPLATE = _TEMPLATES / "example_powergrade_1.6.1.T.drx"
FULL_TEMPLATE = _TEMPLATES / "liftgammagain_1.2.1.T.drx"
CONTRAST_TEMPLATE = _TEMPLATES / "contrast_boost_1.6.4.T.drx"

pytestmark = pytest.mark.skipif(
    not TEMPLATE.exists(), reason="example powergrade template not present"
)
pytest.importorskip("zstandard")


def test_template_nodes_and_sliders():
    from app.core.drx import DrxTemplate

    drx = DrxTemplate(TEMPLATE)
    names = [n.dctl_name for n in drx.nodes]
    for expected in ("SectorSaturation", "SectorBrightness", "SectorSquash",
                     "SectorSkew", "NeutralTint", "HighlightBleach",
                     "ColourCrosstalk", "ColourSaturation"):
        assert expected in names
    # our DCTL nodes expose 12 generic float sliders
    sat = next(n for n in drx.nodes if n.dctl_name == "ColourSaturation")
    assert set(range(12)) <= set(sat.sliders)


def test_full_template_has_contrast_boost_and_lgg():
    """liftgammagain_1.2.1.T.drx is the full stack: all 9 Chromogen
    tools (incl. ContrastBoost, missing from older templates) + LGG."""
    from app.core.drx import DrxTemplate

    drx = DrxTemplate(FULL_TEMPLATE)
    names = [n.dctl_name for n in drx.nodes]
    for expected in ("SectorSaturation", "SectorBrightness", "SectorSquash",
                     "SectorSkew", "NeutralTint", "HighlightBleach",
                     "ColourCrosstalk", "ColourSaturation", "ContrastBoost",
                     "LiftGammaGain", "OpenDRT"):
        assert expected in names, f"{expected} missing from full template"

    cb = next(n for n in drx.nodes if n.dctl_name == "ContrastBoost")
    assert set(range(4)) <= set(cb.sliders)  # Boost/GreyPiv/HighPiv/Chroma
    lgg = next(n for n in drx.nodes if n.dctl_name == "LiftGammaGain")
    assert set(range(5)) <= set(lgg.sliders)  # Lift/Gamma/Gain RGB


def test_contrast_boost_template_parses():
    from app.core.drx import DrxTemplate

    drx = DrxTemplate(CONTRAST_TEMPLATE)
    names = [n.dctl_name for n in drx.nodes]
    assert "ContrastBoost" in names
    assert "OpenDRT" in names


def test_contrast_boost_patch_roundtrips(tmp_path):
    from app.core.drx import DrxTemplate

    drx = DrxTemplate(FULL_TEMPLATE)
    lengths = [len(p) for _, p in drx.bodies]
    cb = next(n for n in drx.nodes if n.dctl_name == "ContrastBoost")
    drx.set_slider(cb, 0, 0.35)   # Contrast Boost
    drx.set_slider(cb, 3, 0.8)    # Chroma
    assert [len(p) for _, p in drx.bodies] == lengths

    out = tmp_path / "patched_full.drx"
    drx.write(out)
    again = DrxTemplate(out)
    cb2 = next(n for n in again.nodes if n.dctl_name == "ContrastBoost")
    assert cb2.sliders[0] == 0.35
    assert cb2.sliders[3] == 0.8


def test_patch_is_fixed_width_and_roundtrips(tmp_path):
    from app.core.drx import DrxTemplate

    drx = DrxTemplate(TEMPLATE)
    lengths = [len(p) for _, p in drx.bodies]
    tint = next(n for n in drx.nodes if n.dctl_name == "NeutralTint")
    drx.set_slider(tint, 0, 220.0)   # Hue
    drx.set_slider(tint, 1, -0.4)    # Amount -> cool lows
    assert [len(p) for _, p in drx.bodies] == lengths  # nothing shifted

    out = tmp_path / "patched.drx"
    drx.write(out)
    again = DrxTemplate(out)
    tint2 = next(n for n in again.nodes if n.dctl_name == "NeutralTint")
    assert tint2.sliders[0] == 220.0
    assert tint2.sliders[1] == -0.4
    # untouched nodes preserved exactly
    sq = next(n for n in again.nodes if n.dctl_name == "SectorSquash")
    assert sq.sliders[2] == 60.0  # Falloff default


def test_unknown_slider_rejected():
    from app.core.drx import DrxTemplate

    drx = DrxTemplate(TEMPLATE)
    node = drx.nodes[0]
    with pytest.raises(KeyError, match="sliderFloatParam99"):
        drx.set_slider(node, 99, 1.0)


def test_lut_match_to_drx_end_to_end(tmp_path, monkeypatch, capsys):
    """The full Plan C pipeline: LUT in -> fitted PowerGrade out."""
    import sys

    import numpy as np

    from app.core.drx import DrxTemplate
    from tests.test_lut_match import _chromogen_look_cube
    from tools import lut_match as cli

    _chromogen_look_cube(tmp_path)
    out_drx = tmp_path / "fitted.drx"
    monkeypatch.setattr(sys, "argv", [
        "lut_match", "--lut", str(tmp_path / "look.cube"),
        "--samples", "500",
        "--drx-out", str(out_drx), "--drx-template", str(TEMPLATE),
    ])
    cli.main()
    text = capsys.readouterr().out
    assert "drx node ColourSaturation" in text
    # LGG/Contrast nodes don't exist in the example template at all
    assert "NO NODE OF THIS TYPE IN TEMPLATE" in text

    fitted = DrxTemplate(out_drx)
    sat = next(n for n in fitted.nodes if n.dctl_name == "ColourSaturation")
    # the fit recovers the baked look's Y/B boost (1.45): the patched
    # node must carry a clearly-raised Y/B slider, not the default.
    # (Plumbing smoke test at 500 samples — exact recovery is covered
    # by test_lut_match_recovers_chromogen_look, so R/G is loose.)
    assert sat.sliders[1] > 1.2
    assert abs(sat.sliders[0] - 1.15) < 0.35


def test_lut_match_full_template_maps_every_stage(tmp_path, monkeypatch,
                                                  capsys):
    """With Marc's full-stack template the default Chromogen-match
    chain (LGG + 5 Chromogen tools incl. ContrastBoost) maps with
    ZERO unmatched stages."""
    import sys

    from app.core.drx import DrxTemplate
    from tests.test_lut_match import _chromogen_look_cube
    from tools import lut_match as cli

    _chromogen_look_cube(tmp_path)
    out_drx = tmp_path / "fitted_full.drx"
    monkeypatch.setattr(sys, "argv", [
        "lut_match", "--lut", str(tmp_path / "look.cube"),
        "--samples", "500",
        "--drx-out", str(out_drx), "--drx-template", str(FULL_TEMPLATE),
    ])
    cli.main()
    text = capsys.readouterr().out
    assert "NO NODE OF THIS TYPE IN TEMPLATE" not in text
    for node in ("LiftGammaGain", "ColourSaturation", "ColourCrosstalk",
                 "ContrastBoost", "HighlightBleach", "NeutralTint"):
        assert f"drx node {node}" in text

    fitted = DrxTemplate(out_drx)
    cb = next(n for n in fitted.nodes if n.dctl_name == "ContrastBoost")
    # template default was 0.516; the fitted (near-identity look on
    # this axis) must have overwritten it
    assert cb.sliders[0] != pytest.approx(0.516279, abs=1e-6)


def test_lut_match_full_stack_preset_duplicates_nodes(tmp_path,
                                                      monkeypatch,
                                                      capsys):
    """The 'Chromogen film look (full stack)' preset wants THREE
    ColourSaturations and TWO NeutralTints — the graph rebuild must
    materialize the extra instances by duplication and wire them in
    chain order."""
    import sys

    from app.core.drx import DrxTemplate
    from app.core.drx_graph import graph_bodies
    from tests.test_lut_match import _chromogen_look_cube
    from tools import lut_match as cli

    _chromogen_look_cube(tmp_path)
    out_drx = tmp_path / "fitted_stack.drx"
    monkeypatch.setattr(sys, "argv", [
        "lut_match", "--lut", str(tmp_path / "look.cube"),
        "--samples", "400",
        "--preset", "Chromogen film look (full stack)",
        "--drx-out", str(out_drx), "--drx-template", str(FULL_TEMPLATE),
    ])
    cli.main()
    text = capsys.readouterr().out
    assert "NO NODE OF THIS TYPE IN TEMPLATE" not in text
    assert "[duplicated]" in text

    fitted = DrxTemplate(out_drx)
    graphs = graph_bodies(fitted)
    graph = max(graphs.values(),
                key=lambda g: sum(1 for n in g.nodes if n.dctl_name))
    main_names = [graph.node(nid).dctl_name for nid in graph.main_line()]
    assert main_names.count("ColourSaturation") == 3
    assert main_names.count("NeutralTint") == 2
    # pure serial: chain covers every node, mixer gone
    assert len(main_names) == len(graph.nodes)
    assert not any(n.is_mixer for n in graph.nodes)
    # fitted chain order: preset says LGG first, tail stays display-side
    assert main_names[0] == "LiftGammaGain"
    assert main_names[-2:] == ["OpenDRT", "MONO-3D-Cube-v1.1"]
