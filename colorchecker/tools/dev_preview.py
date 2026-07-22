"""Standalone DCTL dev preview — `npm run dev` for the Python ports.

    python3 -m tools.dev_preview --image shot.tif [--width 1100]

A native PySide6 window: stack any STAGE_POOL stages as nodes (sliders
generated from param_names/bounds/identity, so future ports appear in
the menu automatically), drag sliders and watch the compound effect on
a LogC3 test image. Extras:

- openDRT preview toggle (the analytic port, Marc's exact config)
- Draw Curve overlay: the enabled chain's R/G/B response on a 0..1
  code-value ramp, drawn over the image a la FilmicContrast
- A/B wipe against the untouched frame (drag the divider)
- HOT RELOAD: saving any file under app/core/ re-imports the stages
  and re-renders, keeping slider values by param name; an import error
  shows as a red banner instead of killing the app
- paste-ready describe() report per node (exact DCTL slider units)

Input is a TIFF (bit-exact app.core.image_io load); with no --image a
built-in synthetic LogC3 chart (hue x EV sweep over grey ramps) is used.
The chain, params and toggles persist across restarts (QSettings).
"""

import argparse
import importlib
import json
import sys
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np

CORE_DIR = Path(__file__).resolve().parents[1] / "app" / "core"

# LogC3 mid-grey code value (chromogen.py's calibration constant)
MID_GREY = 0.391


# --------------------------------------------------------------------
# pure core (no Qt): chain math, curve, synthetic chart, hot reload
# --------------------------------------------------------------------

def load_stage_modules():
    """(Re)import the stage + openDRT modules fresh from disk.

    Purging app.core.* first is what makes hot reload work: the next
    import re-executes the edited source. References other modules
    still hold to the OLD classes stay valid — only the pool returned
    here is rebuilt.
    """
    for name in [m for m in list(sys.modules)
                 if m == "app.core" or m.startswith("app.core.")]:
        del sys.modules[name]
    stages = importlib.import_module("app.core.stages")
    opendrt = importlib.import_module("app.core.opendrt")
    return dict(stages.STAGE_POOL), opendrt.OpenDRTModel()


def stage_specs(pool):
    """UI-facing description of every stage: [{name, params: [...]}].

    Slider names come from param_names (the DCTL slider order); stages
    without them (Matrix, the curve stages) get generic p0..pN.
    """
    specs = []
    for name, cls in pool.items():
        stage = cls()
        ident = np.asarray(stage.identity(), dtype=np.float64)
        lo, hi = (np.asarray(b, dtype=np.float64) for b in stage.bounds())
        names = list(getattr(stage, "param_names", []))
        if len(names) != ident.size:
            names = [f"p{i}" for i in range(ident.size)]
        specs.append({
            "name": name,
            "params": [
                {"name": names[i], "lo": float(lo[i]), "hi": float(hi[i]),
                 "identity": float(ident[i])}
                for i in range(ident.size)
            ],
        })
    return specs


def apply_chain(x, chain, pool):
    """Apply the enabled nodes in order to (..., 3) values (log domain)."""
    shape = np.shape(x)
    y = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    for node in chain:
        if node.get("bypass"):
            continue
        stage = pool[node["stage"]]()
        y = stage.apply(y, np.asarray(node["params"], dtype=np.float64))
    return y.reshape(shape)


def render_chain(img, chain, pool, drt=None, scale=1.0):
    """Chain -> optional openDRT -> clipped uint8 RGB preview frame.

    scale < 1 renders on a downsampled copy (the while-dragging path);
    the caller displays it scaled up, trading resolution for latency.
    """
    src = np.asarray(img, dtype=np.float64)
    if scale != 1.0:
        src = cv2.resize(src, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    y = apply_chain(src, chain, pool)
    if drt is not None:
        y = drt(y.reshape(-1, 3)).reshape(src.shape)
    return (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def chain_curve(chain, pool, n=256):
    """(ramp, response): the chain applied to a neutral 0..1 code ramp.

    Same domain as the DCTL Draw Curve scopes — log code in, log code
    out, NOT through the DRT.
    """
    ramp = np.linspace(0.0, 1.0, n)
    grey = np.repeat(ramp[:, None], 3, axis=1)
    return ramp, apply_chain(grey, chain, pool)


def chain_report(chain, pool):
    """Paste-ready describe() text for every enabled node, in order."""
    parts = []
    for i, node in enumerate(chain):
        if node.get("bypass"):
            parts.append(f"[{i + 1}] {node['stage']} (bypassed)")
            continue
        stage = pool[node["stage"]]()
        params = np.asarray(node["params"], dtype=np.float64)
        parts.append(f"[{i + 1}] {stage.describe(params)}")
    return "\n\n".join(parts)


def encode_logc3(lin):
    """linear -> LogC3 code (EI800), the exact inverse of
    opendrt.oetf_arri_logc3 (verified in tests)."""
    lin = np.asarray(lin, dtype=np.float64)
    log = 0.247190 * np.log10(np.maximum(5.555556 * lin + 0.052272, 1e-10)) \
        + 0.385537
    return np.where(lin < 0.010591, 5.367655 * lin + 0.092809, log)


def _hue_rgb(h):
    """Hue [0,1) -> linear RGB with max component 1 (red at 0)."""
    h6 = (h % 1.0) * 6.0
    r = np.clip(np.abs(h6 - 3.0) - 1.0, 0.0, 1.0)
    g = np.clip(2.0 - np.abs(h6 - 2.0), 0.0, 1.0)
    b = np.clip(2.0 - np.abs(h6 - 4.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def build_chart(width=960, height=540):
    """Synthetic LogC3 test chart, float32 (H, W, 3).

    Top 70%: hue sweep (x) times exposure sweep (y, +4..-4 EV around
    mid grey) at constant saturation. Below: a grey EV ramp (-6..+6
    stops, encoded from linear) and a raw 0..1 CODE ramp — the code
    ramp is exactly the Draw Curve domain, so curve moves can be read
    straight off the image.
    """
    chart = np.zeros((height, width, 3), dtype=np.float64)
    h_field = int(height * 0.70)
    h_strip = (height - h_field) // 2

    hue = _hue_rgb(np.linspace(0.0, 1.0, width))
    ev = np.linspace(4.0, -4.0, h_field)
    sat = 0.75
    base = (1.0 - sat) + sat * hue                          # (W, 3)
    lin = 0.18 * (2.0 ** ev)[:, None, None] * base[None]    # (H, W, 3)
    chart[:h_field] = encode_logc3(lin)

    ev_ramp = 0.18 * 2.0 ** np.linspace(-6.0, 6.0, width)
    chart[h_field:h_field + h_strip] = \
        encode_logc3(ev_ramp)[None, :, None]

    chart[h_field + h_strip:] = np.linspace(0.0, 1.0, width)[None, :, None]
    return chart.astype(np.float32)


def load_preview_image(path, max_width):
    """Bit-exact TIFF load (app.core.image_io), downsampled for speed."""
    from app.core.image_io import load_image
    pixels = load_image(path).pixels
    if pixels.shape[1] > max_width:
        s = max_width / pixels.shape[1]
        pixels = cv2.resize(pixels, None, fx=s, fy=s,
                            interpolation=cv2.INTER_AREA)
    return pixels


class StageHost:
    """The live stage pool + openDRT model, with mtime hot reload.

    poll() re-imports app.core when any of its files changed. On an
    import error the OLD pool stays live and `error` carries the
    traceback (the UI shows it as a banner); the next successful save
    clears it. `version` bumps only on successful reloads.
    """

    def __init__(self):
        self.pool, self.drt = load_stage_modules()
        self.version = 0
        self.error = ""
        self._mtimes = self._snapshot()

    def _snapshot(self):
        return {p.name: p.stat().st_mtime for p in CORE_DIR.glob("*.py")}

    def poll(self):
        """Returns True if a successful reload happened."""
        snap = self._snapshot()
        if snap == self._mtimes:
            return False
        self._mtimes = snap
        try:
            self.pool, self.drt = load_stage_modules()
        except Exception:
            self.error = traceback.format_exc()
            return False
        self.version += 1
        self.error = ""
        return True


# --------------------------------------------------------------------
# Qt shell
# --------------------------------------------------------------------

from PySide6.QtCore import (  # noqa: E402
    QRect, QSettings, Qt, QThread, QTimer, Signal,
)
from PySide6.QtGui import QColor, QImage, QPainter, QPen  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSplitter, QToolButton,
    QVBoxLayout, QWidget,
)

SLIDER_STEPS = 1000


def to_qimage(arr):
    """uint8 (H, W, 3) RGB -> detached QImage."""
    arr = np.ascontiguousarray(arr)
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class RenderWorker(QThread):
    """Latest-wins renderer: only the newest pending request is served,
    so a burst of slider moves never queues up stale frames."""

    done = Signal(object)   # {"frame": uint8, "original": uint8|None,
                            #  "interactive": bool}
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending = None
        self._lock = threading.Lock()
        self._stop = False

    def submit(self, request):
        with self._lock:
            self._pending = request

    def stop(self):
        self._stop = True
        self.wait(5000)

    def run(self):
        while not self._stop:
            with self._lock:
                request, self._pending = self._pending, None
            if request is None:
                self.msleep(10)
                continue
            try:
                drt = request["drt"]
                frame = render_chain(
                    request["img"], request["chain"], request["pool"],
                    drt=drt, scale=request["scale"])
                original = None
                if request["want_original"]:
                    original = render_chain(
                        request["img"], [], request["pool"], drt=drt)
                self.done.emit({"frame": frame, "original": original,
                                "interactive": request["scale"] != 1.0})
            except Exception:
                self.failed.emit(traceback.format_exc())


class Viewer(QWidget):
    """The image area: rendered frame, A/B wipe divider, curve overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = None          # QImage (graded)
        self._original = None       # QImage (untouched, same drt setting)
        self._wipe_on = False
        self._wipe = 0.5
        self._curve = None          # (ramp, (n, 3) response) or None
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_frame(self, frame, original=None):
        self._frame = frame
        if original is not None:
            self._original = original
        self.update()

    def clear_original(self):
        self._original = None
        self.update()

    def set_wipe_enabled(self, on):
        self._wipe_on = on
        self.update()

    def set_curve(self, curve):
        self._curve = curve
        self.update()

    def _image_rect(self):
        if self._frame is None:
            return None
        iw, ih = self._frame.width(), self._frame.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = int(iw * s), int(ih * s)
        return ((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(24, 24, 24))
        rect = self._image_rect()
        if rect is None:
            p.end()
            return
        x, y, w, h = rect
        p.drawImage(*self._draw_args(x, y, w, h, self._frame))

        if self._wipe_on and self._original is not None:
            split = x + int(w * self._wipe)
            p.save()
            p.setClipRect(split, y, x + w - split, h)
            p.drawImage(*self._draw_args(x, y, w, h, self._original))
            p.restore()
            p.setPen(QPen(QColor(255, 255, 255, 200), 1))
            p.drawLine(split, y, split, y + h)

        if self._curve is not None:
            self._paint_curve(p, x, y, w, h)
        p.end()

    @staticmethod
    def _draw_args(x, y, w, h, image):
        return (QRect(x, y, w, h), image)

    def _paint_curve(self, p, x, y, w, h):
        ramp, resp = self._curve
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(x, y, w, h, QColor(0, 0, 0, 70))
        p.setPen(QPen(QColor(255, 255, 255, 90), 1, Qt.DashLine))
        p.drawLine(x, y + h, x + w, y)                       # identity
        gx = x + int(w * MID_GREY)                           # mid grey
        p.drawLine(gx, y, gx, y + h)
        for ch, color in enumerate((QColor(255, 80, 80),
                                    QColor(80, 255, 80),
                                    QColor(90, 140, 255))):
            p.setPen(QPen(color, 2))
            pts = []
            for i in range(len(ramp)):
                v = min(max(resp[i, ch], 0.0), 1.0)
                pts.append((x + ramp[i] * w, y + h - v * h))
            for a, b in zip(pts, pts[1:]):
                p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))

    def mousePressEvent(self, event):
        self._drag_wipe(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._drag_wipe(event)

    def _drag_wipe(self, event):
        rect = self._image_rect()
        if not self._wipe_on or rect is None:
            return
        x, _, w, _ = rect
        self._wipe = min(max((event.position().x() - x) / w, 0.0), 1.0)
        self.update()


class _ResetLabel(QLabel):
    """QLabel whose double-click fires a reset callback (a plain
    instance-attribute override never reaches PySide6's virtual
    dispatch, hence the subclass)."""

    def __init__(self, text, on_reset, parent=None):
        super().__init__(text, parent)
        self._on_reset = on_reset

    def mouseDoubleClickEvent(self, event):
        self._on_reset()


class ParamRow(QWidget):
    """label + slider + spinbox for one stage param, in DCTL units.
    Double-click the label to reset the param to identity."""

    def __init__(self, spec, value, on_change, parent=None):
        super().__init__(parent)
        self._spec = spec
        self._on_change = on_change
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)

        self._label = _ResetLabel(
            spec["name"], lambda: self.set_value(spec["identity"]))
        self._label.setMinimumWidth(90)
        self._label.setToolTip("double-click: reset to identity "
                               f"({spec['identity']:g})")

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, SLIDER_STEPS)
        self._slider.valueChanged.connect(self._slider_moved)
        self._slider.sliderReleased.connect(
            lambda: self._on_change(interactive=False))

        span = spec["hi"] - spec["lo"]
        self._spin = QDoubleSpinBox()
        self._spin.setRange(spec["lo"], spec["hi"])
        self._spin.setDecimals(4)
        self._spin.setSingleStep(span / 200.0 if span else 0.01)
        self._spin.setKeyboardTracking(False)
        self._spin.valueChanged.connect(self._spin_moved)
        self._spin.setMinimumWidth(80)

        row.addWidget(self._label)
        row.addWidget(self._slider, 1)
        row.addWidget(self._spin)
        self.set_value(value, notify=False)

    def value(self):
        return self._spin.value()

    def set_value(self, v, notify=True, interactive=False):
        v = min(max(v, self._spec["lo"]), self._spec["hi"])
        span = self._spec["hi"] - self._spec["lo"]
        t = int(round((v - self._spec["lo"]) / span * SLIDER_STEPS)) \
            if span else 0
        for w in (self._slider, self._spin):
            w.blockSignals(True)
        self._slider.setValue(t)
        self._spin.setValue(v)
        for w in (self._slider, self._spin):
            w.blockSignals(False)
        if notify:
            self._on_change(interactive=interactive)

    def _slider_moved(self, t):
        span = self._spec["hi"] - self._spec["lo"]
        v = self._spec["lo"] + t / SLIDER_STEPS * span
        self._spin.blockSignals(True)
        self._spin.setValue(v)
        self._spin.blockSignals(False)
        # half-res render while the handle is held down, full-res on release
        self._on_change(interactive=self._slider.isSliderDown())

    def _spin_moved(self, v):
        self.set_value(v, notify=False)
        self._on_change(interactive=False)


class NodePanel(QGroupBox):
    """One chain node: checkable box (unchecked = bypass), param rows,
    move up/down and remove buttons."""

    def __init__(self, spec, params, on_change, on_move, on_remove,
                 bypass=False, parent=None):
        super().__init__(spec["name"], parent)
        self.spec = spec

        col = QVBoxLayout(self)
        head = QHBoxLayout()
        # a checkable QGroupBox would also disable the move/remove
        # buttons while bypassed — hence an explicit checkbox
        self.on_box = QCheckBox("on")
        self.on_box.setChecked(not bypass)
        self.on_box.toggled.connect(lambda _: on_change(interactive=False))
        head.addWidget(self.on_box)
        head.addStretch(1)
        for text, action in (("↑", lambda: on_move(self, -1)),
                             ("↓", lambda: on_move(self, +1)),
                             ("✕", lambda: on_remove(self))):
            b = QToolButton()
            b.setText(text)
            b.clicked.connect(action)
            head.addWidget(b)
        col.addLayout(head)

        self.rows = [ParamRow(ps, params[i], on_change)
                     for i, ps in enumerate(spec["params"])]
        for r in self.rows:
            col.addWidget(r)

    def node(self):
        return {"stage": self.spec["name"],
                "params": [r.value() for r in self.rows],
                "bypass": not self.on_box.isChecked()}

    def reset(self):
        for r in self.rows:
            r.set_value(r._spec["identity"], notify=False)


class MainWindow(QMainWindow):
    def __init__(self, img, title="DCTL dev preview", persist=True):
        super().__init__()
        self.setWindowTitle(title)
        self._img = np.asarray(img, dtype=np.float32)
        self._persist = persist
        self._settings = QSettings("colorchecker", "dev_preview")
        self._need_original = True

        self.host = StageHost()
        self.specs = {s["name"]: s for s in stage_specs(self.host.pool)}

        self.worker = RenderWorker()
        self.worker.done.connect(self._frame_ready)
        self.worker.failed.connect(self._show_error)
        self.worker.start()

        self._build_ui()
        if persist:
            self._restore_state()
        self._reload_timer = QTimer(self)
        self._reload_timer.timeout.connect(self._poll_reload)
        self._reload_timer.start(500)
        self.request_render()

    # ---------------- UI scaffolding ----------------

    def _build_ui(self):
        self.viewer = Viewer()

        right = QWidget()
        col = QVBoxLayout(right)

        self.banner = QLabel()
        self.banner.setStyleSheet(
            "background:#5a1f1f;color:#ffb0b0;padding:6px;")
        self.banner.setWordWrap(True)
        self.banner.hide()

        add_row = QHBoxLayout()
        self.stage_menu = QComboBox()
        self.stage_menu.addItems(list(self.specs))
        add = QPushButton("Add node")
        add.clicked.connect(lambda: self.add_node(
            self.stage_menu.currentText()))
        add_row.addWidget(self.stage_menu, 1)
        add_row.addWidget(add)

        toggles = QHBoxLayout()
        self.drt_box = QCheckBox("openDRT")
        self.drt_box.setChecked(True)
        self.drt_box.toggled.connect(self._drt_toggled)
        self.curve_box = QCheckBox("Draw curve")
        self.curve_box.toggled.connect(
            lambda _: self.request_render(interactive=False))
        self.wipe_box = QCheckBox("A/B wipe")
        self.wipe_box.toggled.connect(self.viewer.set_wipe_enabled)
        for w in (self.drt_box, self.curve_box, self.wipe_box):
            toggles.addWidget(w)
        toggles.addStretch(1)

        self.chain_col = QVBoxLayout()
        self.chain_col.addStretch(1)
        chain_host = QWidget()
        chain_host.setLayout(self.chain_col)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(chain_host)

        actions = QHBoxLayout()
        reset = QPushButton("Reset all")
        reset.clicked.connect(self._reset_all)
        copy = QPushButton("Copy report")
        copy.clicked.connect(self._copy_report)
        actions.addWidget(reset)
        actions.addWidget(copy)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(160)

        col.addWidget(self.banner)
        col.addLayout(add_row)
        col.addLayout(toggles)
        col.addWidget(scroll, 1)
        col.addLayout(actions)
        col.addWidget(self.report)

        split = QSplitter()
        split.addWidget(self.viewer)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setSizes([900, 380])
        self.setCentralWidget(split)
        self.resize(1320, 760)

    # ---------------- chain management ----------------

    def panels(self):
        return [self.chain_col.itemAt(i).widget()
                for i in range(self.chain_col.count() - 1)]

    def chain(self):
        return [p.node() for p in self.panels()]

    def add_node(self, stage_name, params=None, bypass=False):
        spec = self.specs[stage_name]
        if params is None or len(params) != len(spec["params"]):
            params = [ps["identity"] for ps in spec["params"]]
        panel = NodePanel(spec, params, self.request_render,
                          self._move_node, self._remove_node, bypass)
        self.chain_col.insertWidget(self.chain_col.count() - 1, panel)
        self.request_render()

    def _move_node(self, panel, delta):
        i = self.panels().index(panel)
        j = i + delta
        if 0 <= j < len(self.panels()):
            self.chain_col.removeWidget(panel)
            self.chain_col.insertWidget(j, panel)
            self.request_render()

    def _remove_node(self, panel):
        self.chain_col.removeWidget(panel)
        panel.deleteLater()
        self.request_render()

    def _reset_all(self):
        for p in self.panels():
            p.reset()
        self.request_render()

    # ---------------- rendering ----------------

    def request_render(self, interactive=False):
        chain = self.chain()
        self.worker.submit({
            "img": self._img,
            "chain": chain,
            "pool": self.host.pool,
            "drt": self.host.drt if self.drt_box.isChecked() else None,
            "scale": 0.5 if interactive else 1.0,
            "want_original": self._need_original,
        })
        self._need_original = False
        self._update_scopes(chain)
        # persist only settled states — not every half-res drag frame
        if self._persist and not interactive:
            self._save_state(chain)

    def _update_scopes(self, chain):
        try:
            if self.curve_box.isChecked():
                self.viewer.set_curve(chain_curve(chain, self.host.pool))
            else:
                self.viewer.set_curve(None)
            self.report.setPlainText(chain_report(chain, self.host.pool))
        except Exception:
            self._show_error(traceback.format_exc())

    def _frame_ready(self, result):
        self.viewer.set_frame(
            to_qimage(result["frame"]),
            to_qimage(result["original"])
            if result["original"] is not None else None)

    def _drt_toggled(self, _):
        self._need_original = True      # the wipe frame is per-DRT-setting
        self.request_render()

    def _show_error(self, text):
        self.banner.setText(text)
        self.banner.show()

    def _copy_report(self):
        QApplication.clipboard().setText(self.report.toPlainText())

    # ---------------- hot reload ----------------

    def _poll_reload(self):
        reloaded = self.host.poll()
        if self.host.error:
            self._show_error(self.host.error)
            return
        if not reloaded:
            return
        self.banner.hide()
        old_specs = self.specs
        self.specs = {s["name"]: s for s in stage_specs(self.host.pool)}
        current = self.stage_menu.currentText()
        self.stage_menu.clear()
        self.stage_menu.addItems(list(self.specs))
        self.stage_menu.setCurrentText(current)
        self._rebuild_panels(self.chain(), old_specs)
        self._need_original = True
        self.request_render()
        self.statusBar().showMessage("reloaded app/core ✓", 2000)

    def _rebuild_panels(self, chain, old_specs):
        """Recreate the node panels against the fresh specs, keeping
        param values by NAME (an edited stage may add/drop sliders)."""
        for p in self.panels():
            self.chain_col.removeWidget(p)
            p.deleteLater()
        for node in chain:
            spec = self.specs.get(node["stage"])
            if spec is None:        # stage vanished in the edit
                continue
            old_names = [ps["name"] for ps in
                         old_specs.get(node["stage"], spec)["params"]]
            old = dict(zip(old_names, node["params"]))
            params = [old.get(ps["name"], ps["identity"])
                      for ps in spec["params"]]
            self.add_node(node["stage"], params, node.get("bypass", False))

    # ---------------- persistence ----------------

    def _save_state(self, chain):
        self._settings.setValue("chain", json.dumps(chain))
        self._settings.setValue("drt", self.drt_box.isChecked())
        self._settings.setValue("curve", self.curve_box.isChecked())
        self._settings.setValue("wipe", self.wipe_box.isChecked())

    def _restore_state(self):
        raw = self._settings.value("chain", "")
        for box, key in ((self.drt_box, "drt"), (self.curve_box, "curve"),
                         (self.wipe_box, "wipe")):
            v = self._settings.value(key)
            if v is not None:
                box.setChecked(v in (True, "true"))
        if not raw:
            return
        try:
            for node in json.loads(raw):
                if node["stage"] in self.specs:
                    self.add_node(node["stage"], node["params"],
                                  node.get("bypass", False))
        except Exception:
            pass                    # a stale blob must never block launch

    def closeEvent(self, event):
        self.worker.stop()
        super().closeEvent(event)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="LogC3 TIFF test image "
                        "(default: built-in synthetic chart)")
    parser.add_argument("--width", type=int, default=1100,
                        help="max preview width in px (default 1100)")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    if args.image:
        img = load_preview_image(args.image, args.width)
        title = f"DCTL dev preview — {Path(args.image).name}"
    else:
        img = build_chart(min(args.width, 960), 540)
        title = "DCTL dev preview — synthetic LogC3 chart"
    win = MainWindow(img, title=title)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
