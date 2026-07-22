"""The standalone DCTL dev preview tool (tools/dev_preview.py):
pure chain math, the synthetic chart, hot reload, and an offscreen
window smoke test through the real widget paths.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

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
    finally:
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
