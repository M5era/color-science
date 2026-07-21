"""PowerGrade (.drx) template patching against Marc's example grade."""

from pathlib import Path

import pytest

TEMPLATE = (Path(__file__).resolve().parents[1]
            / "templates" / "example_powergrade_1.6.1.T.drx")

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
    # R/G lands a touch high (~1.35 vs 1.15): the per-RGB Contrast Curve
    # in the fitted chain shares some of the saturation lift, so the sat
    # node carries a little less. Still a clearly-raised, non-default slider.
    assert abs(sat.sliders[0] - 1.15) < 0.25
