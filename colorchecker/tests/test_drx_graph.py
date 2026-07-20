"""Node-graph surgery on .drx grades: the protobuf re-serializer.

The safety gate is byte-identical round-trip on every template body —
only then are add/duplicate/reorder/relabel trusted at all.
"""

from pathlib import Path

import pytest

_TEMPLATES = Path(__file__).resolve().parents[1] / "templates"
FULL_TEMPLATE = _TEMPLATES / "liftgammagain_1.2.1.T.drx"
EXAMPLE_TEMPLATE = _TEMPLATES / "example_powergrade_1.6.1.T.drx"

pytestmark = pytest.mark.skipif(
    not FULL_TEMPLATE.exists(), reason="templates not present"
)
pytest.importorskip("zstandard")


def _graphs(path):
    from app.core.drx import DrxTemplate
    from app.core.drx_graph import graph_bodies

    drx = DrxTemplate(path)
    return drx, graph_bodies(drx)


# ------------------------------------------------------------- gate

def test_roundtrip_is_byte_identical_for_every_template_body():
    from app.core.drx import DrxTemplate
    from app.core.drx_graph import GradeGraph

    checked = 0
    for tpl in sorted(_TEMPLATES.glob("*.drx")):
        drx = DrxTemplate(tpl)
        for bi, (_, payload) in enumerate(drx.bodies):
            payload = bytes(payload)
            g = GradeGraph.parse(payload)
            if g is None:
                continue
            assert g.serialize() == payload, f"{tpl.name} body {bi}"
            checked += 1
    assert checked >= 8  # 4 templates x 2 graph bodies


# ------------------------------------------------------- topology

def test_full_template_topology():
    """Marc's full stack: entry feeds TWO branches (main chain + a
    LiftGammaGain lane) into a layer mixer; exit from the 3D cube."""
    _, graphs = _graphs(FULL_TEMPLATE)
    # the main grade body is the one with our DCTL stage nodes
    graph = max(graphs.values(),
                key=lambda g: sum(1 for n in g.nodes
                                  if n.dctl_name == "SectorSaturation"))
    assert len(graph.entry_targets()) == 2
    names = {n.dctl_name for n in graph.nodes if n.dctl_name}
    assert {"SectorSaturation", "ContrastBoost", "LiftGammaGain",
            "OpenDRT"} <= names
    assert any(n.is_mixer for n in graph.nodes)

    main = graph.main_line()
    main_names = [graph.node(nid).dctl_name for nid in main]
    assert main_names[0] == "SectorSaturation"
    assert main_names[-1] == "MONO-3D-Cube-v1.1"
    # the walk steps THROUGH the mixer to reach the cube
    assert "OpenDRT" in main_names


def test_labels_and_sliders_readable_from_graph():
    _, graphs = _graphs(FULL_TEMPLATE)
    graph = max(graphs.values(),
                key=lambda g: sum(1 for n in g.nodes if n.dctl_name))
    cb = next(n for n in graph.nodes if n.dctl_name == "ContrastBoost")
    assert cb.label == "BoostCon"
    assert 0 in cb.sliders() and 3 in cb.sliders()


# -------------------------------------------------------- surgery

def test_duplicate_and_serial_rebuild_roundtrips(tmp_path):
    from app.core.drx import DrxTemplate
    from app.core.drx_graph import GradeGraph, graph_bodies, write_graph

    drx, graphs = _graphs(FULL_TEMPLATE)
    body_index, graph = max(
        graphs.items(), key=lambda kv: sum(1 for n in kv[1].nodes
                                           if n.dctl_name))
    tint = next(n for n in graph.nodes if n.dctl_name == "NeutralTint")
    clone = graph.duplicate_node(tint.node_id)
    assert clone.node_id not in (n.node_id for n in graph.nodes
                                 if n is not clone)
    clone.set_slider(0, 220.0)
    clone.label = "CoolLo"

    stage_ids = [n.node_id for n in graph.nodes if n.dctl_name
                 and n.dctl_name != "MONO-3D-Cube-v1.1"
                 and n.dctl_name != "OpenDRT"]
    tail = [n.node_id for n in graph.nodes
            if n.dctl_name in ("OpenDRT", "MONO-3D-Cube-v1.1")]
    order = stage_ids + tail
    graph.serial_rebuild(order)

    assert not any(n.is_mixer for n in graph.nodes)   # mixer dropped
    assert len(graph.entry_targets()) == 1            # single entry
    assert graph.entry_targets()[0] == order[0]
    assert graph.exit_source() == order[-1]
    assert [n.badge for n in graph.nodes] == list(range(1, len(order) + 1))
    assert graph.main_line() == order

    # write -> reload -> everything still parses and carries the edits
    write_graph(drx, body_index, graph)
    out = tmp_path / "rebuilt.drx"
    drx.write(out)

    again = DrxTemplate(out)
    graphs2 = graph_bodies(again)
    g2 = graphs2[body_index]
    assert g2.main_line() == order
    tints = [n for n in g2.nodes if n.dctl_name == "NeutralTint"]
    assert len(tints) == 2
    assert any(n.label == "CoolLo" and n.sliders()[0] == 220.0
               for n in tints)
    # the old byte scanner still reads the patched file (compat)
    assert sum(1 for n in again.nodes if n.dctl_name == "NeutralTint") >= 2


def test_rebuild_as_chain_duplicates_and_resets(tmp_path):
    from app.core.drx import DrxTemplate
    from app.core.drx_graph import graph_bodies, rebuild_as_chain, write_graph
    from app.core.stages import STAGE_POOL

    drx, graphs = _graphs(FULL_TEMPLATE)
    body_index, graph = max(
        graphs.items(), key=lambda kv: sum(1 for n in kv[1].nodes
                                           if n.dctl_name))

    # a chain wanting TWO NeutralTints and TWO ColourSaturations
    tint = STAGE_POOL["Neutral Tint"]()
    sat = STAGE_POOL["Colour Saturation"]()
    p_tint1 = list(tint.identity()); p_tint1[0], p_tint1[1] = 40.0, 0.5
    p_tint2 = list(tint.identity()); p_tint2[0], p_tint2[1] = 220.0, 0.4
    p_sat = list(sat.identity())
    want = [
        ("ColourSaturation", p_sat, "Sat1"),
        ("NeutralTint", p_tint1, "WarmMids"),
        ("NeutralTint", p_tint2, "CoolMids"),
        ("ColourSaturation", p_sat, "Sat2"),
    ]
    stage_names = {"ColourSaturation", "NeutralTint", "SectorSaturation",
                   "SectorBrightness", "SectorSquash", "SectorSkew",
                   "HighlightBleach", "ColourCrosstalk", "ContrastBoost",
                   "LiftGammaGain"}
    identity_lookup = {
        cls().name.replace(" ", ""): list(cls().identity())
        for cls in (STAGE_POOL[n] for n in STAGE_POOL)
        if cls().name.replace(" ", "") in stage_names
    }
    reports = rebuild_as_chain(graph, want, stage_names, identity_lookup)

    # template has 1 ColourSaturation + 1 NeutralTint -> the first of
    # each type is assigned, the second of each is a duplicate
    fitted_names = [r.dctl_name for r in reports[:4]]
    assert fitted_names == ["ColourSaturation", "NeutralTint",
                            "NeutralTint", "ColourSaturation"]
    actions = [r.action for r in reports[:4]]
    assert actions == ["fitted", "fitted", "duplicated", "duplicated"]

    # chain order in the rebuilt graph: fitted block in want order
    main = graph.main_line()
    main_names = [graph.node(nid).dctl_name for nid in main]
    idx = main_names.index("ColourSaturation")
    assert main_names[idx:idx + 4] == ["ColourSaturation", "NeutralTint",
                                       "NeutralTint", "ColourSaturation"]
    # leftovers identity-reset and still in chain, tail preserved
    assert main_names[-2:] == ["OpenDRT", "MONO-3D-Cube-v1.1"]
    leftover = [r for r in reports if r.action == "identity"]
    assert {r.dctl_name for r in leftover} >= {"SectorSaturation",
                                              "ContrastBoost",
                                              "LiftGammaGain"}
    cb = next(n for n in graph.nodes if n.dctl_name == "ContrastBoost")
    ident_cb = identity_lookup["ContrastBoost"]
    assert cb.sliders()[0] == pytest.approx(ident_cb[0])

    # labels landed
    labels = [graph.node(nid).label for nid in main]
    for want_label in ("Sat1", "WarmMids", "CoolMids", "Sat2"):
        assert want_label in labels

    # full file write survives a reload
    write_graph(drx, body_index, graph)
    out = tmp_path / "chain.drx"
    drx.write(out)
    again, graphs2 = _graphs(out)
    assert graphs2[body_index].main_line() == main


def test_serial_rebuild_rejects_unknown_and_duplicate_ids():
    _, graphs = _graphs(EXAMPLE_TEMPLATE)
    graph = max(graphs.values(),
                key=lambda g: sum(1 for n in g.nodes if n.dctl_name))
    ids = [n.node_id for n in graph.nodes]
    with pytest.raises(KeyError):
        graph.serial_rebuild(ids + [99999])
    with pytest.raises(ValueError):
        graph.serial_rebuild([ids[0], ids[0]])
