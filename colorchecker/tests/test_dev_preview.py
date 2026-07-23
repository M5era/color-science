"""The standalone DCTL dev preview tool (tools/dev_preview.py):
pure chain math, the synthetic chart, hot reload, and an offscreen
window smoke test through the real widget paths.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app.core.lut import CubeLUT
from app.core.opendrt import oetf_arri_logc3
from app.core.stages import STAGE_POOL
from tools import dev_preview as dp


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def pool():
    return dict(STAGE_POOL)


def _contrast_node(pool, **overrides):
    """A Contrast Curve node with named slider overrides."""
    stage = pool["Contrast Curve"]()
    params = np.asarray(stage.identity(), dtype=np.float64).tolist()
    for name, value in overrides.items():
        params[stage.param_names.index(name)] = value
    return {"stage": "Contrast Curve", "params": params, "bypass": False}


def _gain_lut(gain=0.5, size=5):
    """A tiny linear-gain 3D LUT (trilinear interp reproduces it exactly)."""
    g = np.linspace(0.0, 1.0, size)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    table = np.stack([r, gg, b], axis=-1) * gain
    return CubeLUT(title="gain", size=size, domain_min=np.zeros(3),
                   domain_max=np.ones(3), table=table)


# ------------------------------------------------------------------
# pure core
# ------------------------------------------------------------------

def test_stage_specs_cover_pool(pool):
    specs = {s["name"]: s for s in dp.stage_specs(pool)}
    assert set(specs) == set(pool)
    for name, cls in pool.items():
        stage = cls()
        ident = np.asarray(stage.identity())
        params = specs[name]["params"]
        assert len(params) == ident.size
        for p, ident_v in zip(params, ident):
            assert np.isfinite([p["lo"], p["hi"], p["identity"]]).all()
            assert p["lo"] <= p["identity"] <= p["hi"]
            assert p["identity"] == pytest.approx(float(ident_v))


def test_encode_logc3_is_inverse_of_the_port():
    lin = np.concatenate([np.linspace(0.0, 0.05, 200),
                          np.linspace(0.05, 8.0, 200)])
    np.testing.assert_allclose(
        oetf_arri_logc3(dp.encode_logc3(lin)), lin, atol=1e-6)


def test_build_chart_shape_and_range():
    chart = dp.build_chart(320, 180)
    assert chart.shape == (180, 320, 3)
    assert chart.dtype == np.float32
    assert np.isfinite(chart).all()
    assert chart.min() >= 0.0 and chart.max() <= 1.0
    # the bottom strip is the raw 0..1 code ramp (the curve domain)
    np.testing.assert_allclose(chart[-1, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(chart[-1, -1], 1.0, atol=1e-6)


def test_apply_chain_identity_and_bypass(pool):
    rng = np.random.default_rng(7)
    x = rng.uniform(0.05, 0.9, (50, 3))
    assert np.array_equal(dp.apply_chain(x, [], pool), x)

    node = _contrast_node(pool, Contrast=1.6)
    bypassed = dict(node, bypass=True)
    np.testing.assert_array_equal(dp.apply_chain(x, [bypassed], pool), x)
    assert not np.allclose(dp.apply_chain(x, [node], pool), x)


def test_apply_chain_composes_in_order(pool):
    rng = np.random.default_rng(8)
    x = rng.uniform(0.05, 0.9, (50, 3))
    a = _contrast_node(pool, Contrast=1.4)
    sat_stage = pool["Colour Saturation"]()
    b_params = np.asarray(sat_stage.identity()).tolist()
    b_params[0] = 1.5
    b = {"stage": "Colour Saturation", "params": b_params, "bypass": False}

    manual = sat_stage.apply(
        pool["Contrast Curve"]().apply(x, np.asarray(a["params"])),
        np.asarray(b_params))
    np.testing.assert_allclose(dp.apply_chain(x, [a, b], pool), manual)


def test_render_chain_frames(pool):
    chart = dp.build_chart(160, 90)
    plain = dp.render_chain(chart, [], pool)
    assert plain.shape == (90, 160, 3) and plain.dtype == np.uint8

    node = _contrast_node(pool, Contrast=1.8)
    graded = dp.render_chain(chart, [node], pool)
    assert not np.array_equal(graded, plain)

    half = dp.render_chain(chart, [node], pool, scale=0.5)
    assert half.shape == (45, 80, 3)

    drt = dp.load_stage_modules()[1]
    through = dp.render_chain(chart, [node], pool, drt=drt)
    assert through.shape == plain.shape
    assert not np.array_equal(through, graded)


def test_chain_curve_identity_is_diagonal(pool):
    ramp, resp = dp.chain_curve([], pool, n=64)
    for ch in range(3):
        np.testing.assert_allclose(resp[:, ch], ramp, atol=1e-12)
    node = _contrast_node(pool, Contrast=1.8)
    _, resp = dp.chain_curve([node], pool, n=64)
    assert not np.allclose(resp[:, 0], ramp)


def test_chain_report_lists_nodes(pool):
    node = _contrast_node(pool, Contrast=1.5)
    text = dp.chain_report(
        [node, dict(node, bypass=True)], pool)
    assert "[1]" in text and "[2]" in text
    assert "bypassed" in text
    assert "Contrast" in text


# ------------------------------------------------------------------
# B-side LUT rendering
# ------------------------------------------------------------------

def test_render_chain_applies_lut(pool):
    chart = dp.build_chart(160, 90)
    plain = dp.render_chain(chart, [], pool)
    halved = dp.render_chain(chart, [], pool, lut=_gain_lut(0.5))
    expected = (np.clip(chart * 0.5, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    assert np.abs(halved.astype(int) - expected.astype(int)).max() <= 1
    assert not np.array_equal(halved, plain)


# ------------------------------------------------------------------
# HD curve core (port of the HD Curve Probe/Display DCTLs)
# ------------------------------------------------------------------

def test_hd_probe_identity_sweeps_logc3():
    stops, out = dp.hd_probe(lambda c: c, n=128)
    assert stops[0] == pytest.approx(-dp.HD_STOP_RANGE)
    assert stops[-1] == pytest.approx(dp.HD_STOP_RANGE)
    expected = dp.encode_logc3(0.18 * 2.0 ** stops)
    for ch in range(3):
        np.testing.assert_allclose(out[:, ch], expected)
    # mid grey: 0 stops encodes to the LogC3 mid-grey code value
    mid = dp.encode_logc3(np.asarray(0.18))
    assert mid == pytest.approx(dp.MID_GREY, abs=5e-4)


def test_eotf_and_density_helpers():
    np.testing.assert_allclose(dp.eotf_to_linear(0.5, "none"), 0.5)
    np.testing.assert_allclose(dp.eotf_to_linear(0.5, "g24"), 0.5 ** 2.4)
    np.testing.assert_allclose(dp.eotf_to_linear(0.02, "srgb"),
                               0.02 / 12.92)
    assert dp.density_of(1.0) == pytest.approx(0.0)
    assert dp.density_of(0.01) == pytest.approx(2.0)
    assert np.isfinite(dp.density_of(0.0))    # clamped by DENSITY_EPS


def test_find_active_stop_range_detects_clip():
    stops = np.linspace(-8.0, 8.0, 512)
    # transitions between plateaus only inside -4..+4 stops
    v = np.clip((stops + 4.0) / 8.0, 0.0, 1.0) * 0.85 + 0.05
    rgb = np.repeat(v[:, None], 3, axis=1)
    smin, smax = dp.find_active_stop_range(stops, rgb)
    assert smin == pytest.approx(-4.0, abs=0.35)
    assert smax == pytest.approx(4.0, abs=0.35)

    # a flat curve has no active range -> falls back to the full sweep
    flat = np.full_like(rgb, 0.5)
    assert dp.find_active_stop_range(stops, flat) == (-8.0, 8.0)


def test_nice_tick_step_matches_dctl():
    assert dp.nice_tick_step(16.5, 8) == pytest.approx(2.0)
    assert dp.nice_tick_step(100.0, 5) == pytest.approx(20.0)
    assert dp.nice_tick_step(3.0, 8) == pytest.approx(0.5)
    assert dp.nice_tick_step(0.0, 8) == 1.0


def test_hd_side_percent_and_density():
    side = dp.hd_side(lambda c: c, "A", y_mode="percent", clamp=False)
    assert side["y_max"] == 100.0
    assert side["stop_min"] == pytest.approx(-dp.HD_STOP_RANGE)
    expected = dp.encode_logc3(0.18 * 2.0 ** side["stops"]) * 100.0
    np.testing.assert_allclose(side["y"][:, 0], expected)

    dense = dp.hd_side(lambda c: c, "A", y_mode="density", eotf="g24",
                       clamp=True)
    assert dense["y_max"] >= 0.5
    assert dense["y_max"] % 0.5 == pytest.approx(0.0)
    assert dense["stop_min"] >= -dp.HD_STOP_RANGE
    assert dense["stop_max"] <= dp.HD_STOP_RANGE
    assert np.isfinite(dense["y"]).all()


# ------------------------------------------------------------------
# viewer zoom (trackpad pinch / pan / reset)
# ------------------------------------------------------------------

def test_viewer_zoom_pan_and_reset(qapp):
    from PySide6.QtCore import QPointF
    v = dp.Viewer()
    v.resize(800, 500)
    v.set_frame(dp.to_qimage(dp.render_chain(dp.build_chart(160, 90),
                                             [], dict(STAGE_POOL))))
    base = v._base_rect()
    assert v._image_rect() == base            # zoom 1 = fit

    center = QPointF(400, 250)
    v._apply_zoom(2.0, center)
    x, y, w, h = v._image_rect()
    assert w == 2 * base[2] and h == 2 * base[3]

    # zooming at a corner keeps that corner's content anchored: the
    # pan is pushed toward it (and stays within the clamp)
    v._apply_zoom(1.5, QPointF(0, 0))
    assert v._zoom == pytest.approx(3.0)
    mx = (base[2] * v._zoom - base[2]) / 2
    assert abs(v._pan[0]) <= mx + 1e-6

    # zooming out below 1 clamps to fit and recenters
    v._apply_zoom(0.01, center)
    assert v._zoom == 1.0 and v._pan == [0.0, 0.0]
    assert v._image_rect() == base


# ------------------------------------------------------------------
# JSON chain import/export
# ------------------------------------------------------------------

def test_chain_json_roundtrip(pool):
    specs = {s["name"]: s for s in dp.stage_specs(pool)}
    node = _contrast_node(pool, Contrast=1.44)
    exported = dp.chain_to_export([node, dict(node, bypass=True)], specs)
    assert exported[0]["params"]["Contrast"] == pytest.approx(1.44)
    assert exported[1]["bypass"] is True

    chain, notes = dp.chain_from_import(exported, specs)
    assert notes == []
    assert chain[0]["params"] == node["params"]
    assert chain[1]["bypass"] is True


def test_chain_import_is_forgiving(pool):
    specs = {s["name"]: s for s in dp.stage_specs(pool)}
    nodes = [
        {"stage": "Nope Stage", "params": {}},
        {"stage": "Contrast Curve",
         "params": {"Contrast": 1.3, "Bogus Slider": 9.0}},
    ]
    chain, notes = dp.chain_from_import(nodes, specs)
    assert len(chain) == 1                      # unknown stage skipped
    assert len(notes) == 2                      # both problems reported
    spec = specs["Contrast Curve"]
    i = [ps["name"] for ps in spec["params"]].index("Contrast")
    assert chain[0]["params"][i] == pytest.approx(1.3)
    # every other param fell back to identity
    for j, ps in enumerate(spec["params"]):
        if j != i:
            assert chain[0]["params"][j] == pytest.approx(ps["identity"])


def test_window_import_export_files(qapp, tmp_path):
    win = dp.MainWindow(dp.build_chart(160, 90), persist=False)
    try:
        win.add_node("Contrast Curve")
        panel = win.panels()[0]
        i = [ps["name"] for ps in panel.spec["params"]].index("Contrast")
        panel.rows[i].set_value(1.62)
        win.hd_box.setChecked(True)
        win.hd_clamp.setChecked(False)

        out = tmp_path / "chain.json"
        win._export_chain(str(out))
        assert out.is_file()

        win._remove_node(win.panels()[0])
        win.hd_box.setChecked(False)
        assert win.chain() == []

        win._import_chain(str(out))
        assert [n["stage"] for n in win.chain()] == ["Contrast Curve"]
        assert win.chain()[0]["params"][i] == pytest.approx(1.62)
        assert win.hd_box.isChecked()
        assert not win.hd_clamp.isChecked()

        # paste-import: same JSON straight from the clipboard
        win._remove_node(win.panels()[0])
        assert win.chain() == []
        QApplication.clipboard().setText(out.read_text())
        win._paste_chain()
        assert [n["stage"] for n in win.chain()] == ["Contrast Curve"]
        assert win.chain()[0]["params"][i] == pytest.approx(1.62)

        # invalid clipboard shows the banner instead of crashing
        QApplication.clipboard().setText("not json {")
        win._paste_chain()
        assert "not valid chain JSON" in win.banner.text()
    finally:
        win.close()


# ------------------------------------------------------------------
# hot reload
# ------------------------------------------------------------------

def test_stage_host_reload_on_touch():
    host = dp.StageHost()
    assert host.version == 0 and host.error == ""
    assert not host.poll()          # nothing changed

    target = dp.CORE_DIR / "windows.py"
    stat = target.stat()
    os.utime(target, (stat.st_atime, stat.st_mtime + 1))
    try:
        assert host.poll()
        assert host.version == 1 and host.error == ""
        assert set(host.pool) == set(STAGE_POOL)
    finally:
        os.utime(target, (stat.st_atime, stat.st_mtime))
        host.poll()                 # restore + leave modules importable


def test_stage_host_survives_broken_core(monkeypatch):
    host = dp.StageHost()
    old_pool = host.pool
    target = dp.CORE_DIR / "windows.py"
    stat = target.stat()
    os.utime(target, (stat.st_atime, stat.st_mtime + 1))
    try:
        # an import error during reload (e.g. a syntax error saved in
        # app/core) must keep the OLD pool live and surface the trace
        monkeypatch.setattr(
            dp, "load_stage_modules",
            lambda: (_ for _ in ()).throw(SyntaxError("boom")))
        assert not host.poll()
        assert "boom" in host.error
        assert host.pool is old_pool

        monkeypatch.undo()
        os.utime(target, (stat.st_atime, stat.st_mtime))
        assert host.poll()                  # next save recovers
        assert host.error == ""
    finally:
        os.utime(target, (stat.st_atime, stat.st_mtime))
        host.poll()


# ------------------------------------------------------------------
# window smoke (offscreen, real widget paths)
# ------------------------------------------------------------------

def test_window_smoke(qapp):
    win = dp.MainWindow(dp.build_chart(160, 90), persist=False)
    try:
        assert set(win.specs) == set(win.host.pool)
        assert win.chain() == []

        win.add_node("Contrast Curve")
        win.add_node("Colour Saturation")
        assert [n["stage"] for n in win.chain()] == \
            ["Contrast Curve", "Colour Saturation"]

        # slider path: pushing Contrast off identity shows up in chain()
        panel = win.panels()[0]
        i = [ps["name"] for ps in panel.spec["params"]].index("Contrast")
        panel.rows[i].set_value(1.7)
        assert win.chain()[0]["params"][i] == pytest.approx(1.7)
        assert "Contrast" in win.report.toPlainText()

        # per-param reset button restores identity and re-renders
        identity = panel.spec["params"][i]["identity"]
        panel.rows[i]._reset.click()
        assert win.chain()[0]["params"][i] == pytest.approx(identity)

        # per-node reset button restores every param at once
        panel.rows[i].set_value(1.7)
        other = 0 if i != 0 else 1
        panel.rows[other].set_value(panel.spec["params"][other]["hi"])
        panel._reset_btn.click()
        for j, ps in enumerate(panel.spec["params"]):
            assert win.chain()[0]["params"][j] == \
                pytest.approx(ps["identity"])
        panel.rows[i].set_value(1.7)    # leave a non-identity value

        # bypass, reorder, remove
        panel.on_box.setChecked(False)
        assert win.chain()[0]["bypass"] is True
        win._move_node(win.panels()[1], -1)
        assert win.chain()[0]["stage"] == "Colour Saturation"
        win._remove_node(win.panels()[0])
        assert [n["stage"] for n in win.chain()] == ["Contrast Curve"]

        # a real render through the worker path, synchronously
        frame = dp.render_chain(
            win._img, win.chain(), win.host.pool, drt=win.host.drt)
        assert frame.dtype == np.uint8
        assert frame.shape == (90, 160, 3)

        # curve overlay toggle goes through without error
        win.curve_box.setChecked(True)
        assert win.viewer._curve is not None

        # HD curves overlay: one chart, A and B through the wipe's
        # exact paths, merged into one coordinate system
        win.hd_box.setChecked(True)
        hd = win.viewer._hd
        assert hd is not None and hd["mode"] == "wipe"
        assert hd["a"]["title"] == "A · chain"
        assert hd["b"]["title"] == "B · original"
        assert hd["stop_min"] <= min(hd["a"]["stop_min"],
                                     hd["b"]["stop_min"]) + 1e-9
        assert hd["stop_max"] >= max(hd["a"]["stop_max"],
                                     hd["b"]["stop_max"]) - 1e-9

        # a B-side LUT changes the B frame and the B curve title
        win._lut = _gain_lut(0.5)
        win._lut_path = "/tmp/gain.cube"
        win._need_original = True
        win.request_render()
        assert win.viewer._hd["b"]["title"] == "B · gain"

        # the LUT stands on its own: even with openDRT enabled for A,
        # the B curve is the bare probe through the LUT, no DRT on top
        assert win.drt_box.isChecked()
        from app.core.lut import apply_lut
        bare = dp.hd_side(lambda c: apply_lut(win._lut, c), "x")
        np.testing.assert_allclose(win.viewer._hd["b"]["y"], bare["y"])
        b_lut = dp.render_chain(win._img, [], win.host.pool,
                                lut=win._lut)
        b_plain = dp.render_chain(win._img, [], win.host.pool)
        assert not np.array_equal(b_lut, b_plain)

        win._clear_lut()
        assert win.viewer._hd["b"]["title"] == "B · original"

        # density mode, Both overlay and plot shapes recompute fine
        win.hd_y.setCurrentIndex(1)
        assert win.viewer._hd["y_mode"] == "density"
        win.hd_mode.setCurrentIndex(1)
        assert win.viewer._hd["mode"] == "both"
        win.hd_shape.setCurrentIndex(2)
        assert win.viewer._hd["shape"] == 2
    finally:
        win.close()


def test_self_restart_on_tool_edit(qapp, monkeypatch):
    win = dp.MainWindow(dp.build_chart(160, 90), persist=False)
    calls = []
    monkeypatch.setattr(dp.os, "execv",
                        lambda exe, argv: calls.append((exe, argv)))
    stat = dp.TOOL_FILE.stat()
    os.utime(dp.TOOL_FILE, (stat.st_atime, stat.st_mtime + 1))
    try:
        win._poll_self_restart()
        assert len(calls) == 1
        exe, argv = calls[0]
        assert exe == dp.sys.executable
        assert argv[:3] == [dp.sys.executable, "-m",
                            "tools.dev_preview"]
        # unchanged mtime afterwards: no restart loop
        win._poll_self_restart()
        assert len(calls) == 1
    finally:
        os.utime(dp.TOOL_FILE, (stat.st_atime, stat.st_mtime))
        win.close()


def test_self_restart_guards_broken_file(qapp, monkeypatch):
    win = dp.MainWindow(dp.build_chart(160, 90), persist=False)
    calls = []
    monkeypatch.setattr(dp.os, "execv",
                        lambda *a: calls.append(a))
    monkeypatch.setattr(
        dp.py_compile, "compile",
        lambda *a, **k: (_ for _ in ()).throw(
            dp.py_compile.PyCompileError(
                SyntaxError, SyntaxError("boom"), "bad")))
    stat = dp.TOOL_FILE.stat()
    os.utime(dp.TOOL_FILE, (stat.st_atime, stat.st_mtime + 1))
    try:
        win._poll_self_restart()
        assert not calls                    # no restart into a crash
        assert win.banner.isVisible() or win.banner.text()
    finally:
        os.utime(dp.TOOL_FILE, (stat.st_atime, stat.st_mtime))
        win.close()


def test_window_rebuild_keeps_values_by_name(qapp):
    win = dp.MainWindow(dp.build_chart(160, 90), persist=False)
    try:
        win.add_node("Contrast Curve")
        panel = win.panels()[0]
        i = [ps["name"] for ps in panel.spec["params"]].index("Contrast")
        panel.rows[i].set_value(1.55)
        win._rebuild_panels(win.chain(), win.specs)
        assert win.chain()[0]["params"][i] == pytest.approx(1.55)
    finally:
        win.close()
