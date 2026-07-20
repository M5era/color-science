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
    assert "drx node ColourSaturation#0" in text
    assert "NO NODE IN TEMPLATE" in text  # LGG/Contrast not in template

    fitted = DrxTemplate(out_drx)
    sat = next(n for n in fitted.nodes if n.dctl_name == "ColourSaturation")
    # the fit recovers the baked look's Y/B boost (1.45): the patched
    # node must carry a clearly-raised Y/B slider, not the default
    assert sat.sliders[1] > 1.2
    assert abs(sat.sliders[0] - 1.15) < 0.2


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
    assert "NO NODE IN TEMPLATE" not in text
    for node in ("LiftGammaGain#0", "ColourSaturation#0",
                 "ColourCrosstalk#0", "ContrastBoost#0",
                 "HighlightBleach#0", "NeutralTint#0"):
        assert f"drx node {node}" in text

    fitted = DrxTemplate(out_drx)
    cb = next(n for n in fitted.nodes if n.dctl_name == "ContrastBoost")
    # template default was 0.516; the fitted (near-identity look on
    # this axis) must have overwritten it
    assert cb.sliders[0] != pytest.approx(0.516279, abs=1e-6)
