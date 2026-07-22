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


KITCHEN_SINK = (Path(__file__).resolve().parents[1]
                / "templates" / "all_nodes_1.10.3.T.drx")


def test_lut_match_to_drx_end_to_end(tmp_path, monkeypatch, capsys):
    """The full Plan C pipeline: LUT in -> from-scratch PowerGrade out.

    Exercises the real deliverable path — a free-order --search fit
    exported by cloning exactly the fitted chain (+ DRT) out of the
    kitchen-sink template, each node carrying its short label."""
    import sys

    from app.core.drx import DrxTemplate
    from app.core.protobuf import Message
    from tests.test_lut_match import _chromogen_look_cube
    from tools import lut_match as cli

    if not KITCHEN_SINK.exists():
        pytest.skip("kitchen-sink template not present")

    _chromogen_look_cube(tmp_path)
    out_drx = tmp_path / "fitted.drx"
    monkeypatch.setattr(sys, "argv", [
        "lut_match", "--lut", str(tmp_path / "look.cube"),
        "--search", "--max-nodes", "4", "--samples", "500",
        "--drx-out", str(out_drx), "--drx-template", str(KITCHEN_SINK),
    ])
    cli.main()
    text = capsys.readouterr().out
    assert "from scratch" in text

    fitted = DrxTemplate(out_drx)
    names = [n.dctl_name for n in fitted.nodes]
    # the generated stack is EXACTLY the fitted look nodes + the DRT,
    # nothing else — no leftover identity nodes from the kitchen sink
    assert names[-1] == "OpenDRT"
    assert "LiftGammaGain" not in names          # --search never emits LGG
    assert len(names) == len(cli_stage_count(text)) + 1  # look nodes + DRT

    # every generated node carries a non-empty display label (field 6)
    body = _node_container(fitted)
    labels = [n.as_message().find(6) for n in body.find(7)]
    assert all(lbl and lbl[0].value for lbl in labels)


def cli_stage_count(text: str) -> list[str]:
    """The 'drx node <Type>#<k>' lines list one entry per look node."""
    return [ln for ln in text.splitlines() if ln.strip().startswith("drx node")]


def _node_container(fitted):
    """The body-1 container holding the node stack (field 7 + field 8)."""
    from app.core.protobuf import Message

    for _prefix, payload in fitted.bodies:
        body = Message.parse(bytes(payload))
        for cont in body.find(1):
            contm = cont.as_message()
            if contm.find(7) and contm.find(8):
                return contm
    raise AssertionError("no node-stack container found")
