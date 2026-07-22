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


def test_build_grade_from_scratch_with_labels(tmp_path):
    """From-scratch PowerGrade generation (the export deliverable path):
    build_grade clones EXACTLY the requested chain (+ DRT) out of the
    kitchen-sink template, in order, each node carrying its label. Node
    types repeat as often as asked. This is what the CLI --drx-out path
    calls under the hood."""
    from app.core.drx import DrxTemplate
    from app.core.drx_build import build_grade

    if not KITCHEN_SINK.exists():
        pytest.skip("kitchen-sink template not present")

    out_drx = tmp_path / "fitted.drx"
    names = ["ContrastCurve", "ColourSaturation", "SectorBrightness",
             "SectorBrightness"]
    labels = ["Con", "BstYB", "DrkBlu", "BrtOrg"]
    build_grade(KITCHEN_SINK, names, out_drx, labels=labels + ["OpenDRT"])

    fitted = DrxTemplate(out_drx)
    dnames = [n.dctl_name for n in fitted.nodes]
    # exactly the requested chain + DRT, in order — no leftover template nodes
    assert dnames == names + ["OpenDRT"]
    # every generated node carries a non-empty display label (field 6)
    body = _node_container(fitted)
    got = [n.as_message().find(6) for n in body.find(7)]
    assert [g[0].value.decode() for g in got] == labels + ["OpenDRT"]


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
