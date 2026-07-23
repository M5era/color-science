"""Standalone DCTL dev preview — `npm run dev` for the Python ports.

    python3 -m tools.dev_preview --image shot.tif [--width 1100]

A native PySide6 window: stack any STAGE_POOL stages as nodes (sliders
generated from param_names/bounds/identity, so future ports appear in
the menu automatically), drag sliders and watch the compound effect on
a LogC3 test image. Extras:

- openDRT preview toggle (the analytic port, Marc's exact config)
- Draw Curve overlay: the enabled chain's R/G/B response on a 0..1
  code-value ramp, drawn over the image a la FilmicContrast
- A/B wipe against the untouched frame (drag the divider), or against
  a .cube LUT of your choice ("B LUT..." button): B becomes
  image -> LUT with NOTHING on top — a LUT is a complete transform
  of its own, so openDRT is never applied to the B side
- Trackpad zoom a la Safari: pinch zooms around the cursor (or
  Cmd+scroll), two-finger scroll pans while zoomed, double-click
  resets to fit. The HD-curve / Draw-curve overlays zoom along with
  the image, so you can pinch into a chart region (e.g. mid grey)
  to inspect the match up close
- HD curves overlay: ONE sensitometric chart with A (chain) and B
  (LUT / original) in the same coordinate system — the HD Curve
  Probe+Display DCTL logic in the UI: +-8.2574-stop LogC3 sweep
  around mid grey, Output % or Density Y axis (with EOTF
  linearization), density-space active range clamping, Plot Shape
  (Fill Width / Square / Datasheet 1:1). A/B mode "Wipe" splits the
  curves at the draggable wipe divider (A left, B right); "Both"
  overlays them with B muted
- HOT RELOAD: saving any file under app/core/ re-imports the stages
  and re-renders, keeping slider values by param name; an import error
  shows as a red banner instead of killing the app. Saving
  tools/dev_preview.py ITSELF auto-restarts the process in place
  (chain, sliders, LUT and toggles come back via QSettings); a syntax
  error shows as a banner and the old UI stays alive
- paste-ready describe() report per node (exact DCTL slider units)
- JSON chain import/export ("Import..."/"Export..."): the whole node
  tree in one file, params stored BY NAME (robust to slider
  additions), optionally with the B-side LUT path, openDRT toggle and
  HD-curve settings — a preset restores the entire comparison setup

Input is a TIFF (bit-exact app.core.image_io load); with no --image a
built-in synthetic LogC3 chart (hue x EV sweep over grey ramps) is used.
The chain, params and toggles persist across restarts (QSettings).
"""

import argparse
import importlib
import json
import os
import py_compile
import sys
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np

from app.core.lut import apply_lut, parse_cube

CORE_DIR = Path(__file__).resolve().parents[1] / "app" / "core"
TOOL_FILE = Path(__file__).resolve()

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


def render_chain(img, chain, pool, drt=None, scale=1.0, lut=None):
    """Chain -> optional LUT -> optional openDRT -> uint8 RGB frame.

    scale < 1 renders on a downsampled copy (the while-dragging path);
    the caller displays it scaled up, trading resolution for latency.
    The B side of the wipe uses chain=[] with a CubeLUT.
    """
    src = np.asarray(img, dtype=np.float64)
    if scale != 1.0:
        src = cv2.resize(src, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    y = apply_chain(src, chain, pool)
    if lut is not None:
        y = apply_lut(lut, y)
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


def chain_to_export(chain, specs):
    """Chain -> JSON-ready node list with params keyed BY NAME (robust
    to future slider additions/reordering)."""
    nodes = []
    for node in chain:
        spec = specs[node["stage"]]
        names = [ps["name"] for ps in spec["params"]]
        nodes.append({
            "stage": node["stage"],
            "bypass": bool(node.get("bypass", False)),
            "params": {n: float(v) for n, v in zip(names, node["params"])},
        })
    return nodes


def chain_from_import(nodes, specs):
    """JSON node list -> (positional chain, warnings). Unknown stages
    are skipped, missing params fall back to identity, unknown param
    names are ignored — each with a warning so nothing fails silently."""
    chain, notes = [], []
    for nd in nodes:
        spec = specs.get(nd.get("stage"))
        if spec is None:
            notes.append(f"unknown stage skipped: {nd.get('stage')!r}")
            continue
        by_name = dict(nd.get("params", {}))
        known = {ps["name"] for ps in spec["params"]}
        unknown = sorted(set(by_name) - known)
        if unknown:
            notes.append(f"{nd['stage']}: ignored params {unknown}")
        params = [float(by_name.get(ps["name"], ps["identity"]))
                  for ps in spec["params"]]
        chain.append({"stage": nd["stage"], "params": params,
                      "bypass": bool(nd.get("bypass", False))})
    return chain, notes


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


# --------------------------------------------------------------------
# HD curve core — port of the HD Curve Probe (v2.3) + Display (v2.23)
# DCTL pair. The probe sweeps the LogC3 dynamic range around 18% mid
# grey; the display's axis logic (density, EOTF linearization, active
# range clamping) runs here on the sampled response.
# --------------------------------------------------------------------

# Symmetric stop range of the LogC3 probe strip (probe DCTL v2.1+):
# bounded by where LogC3 clips at code 1.0.
HD_STOP_RANGE = 8.2574
HD_SAMPLES = 256
LOG10_2 = 0.3010299957          # 1 stop in log10-exposure decades

# Active range detection (display DCTL): the scan runs in density
# space with per-channel relative thresholds against the plateau
# values; highlights converge slowly in density, hence the finer
# threshold on that side.
ACTIVE_RANGE_THRESH_SHADOWS = 0.02
ACTIVE_RANGE_THRESH_HIGHLIGHTS = 0.005
ACTIVE_RANGE_MIN_SPAN_STOPS = 1.0
DENSITY_EPS = 1.0e-5


def density_of(value):
    return -np.log10(np.maximum(np.asarray(value, dtype=np.float64),
                                DENSITY_EPS))


def eotf_to_linear(code, eotf):
    """Linearize display code values so density measures light, not
    code. eotf: "none" (as-is / already linear), "g24", "srgb"."""
    c = np.maximum(np.asarray(code, dtype=np.float64), 0.0)
    if eotf == "g24":
        return c ** 2.4
    if eotf == "srgb":
        return np.where(c <= 0.04045, c / 12.92,
                        ((c + 0.055) / 1.055) ** 2.4)
    return c


def hd_probe(transform, n=HD_SAMPLES):
    """(stops, out (n, 3)): the probe strip through `transform`.

    `transform` maps (n, 3) LogC3 codes to output codes — it stands in
    for everything between the probe and display nodes in Resolve."""
    stops = np.linspace(-HD_STOP_RANGE, HD_STOP_RANGE, n)
    grey = encode_logc3(0.18 * 2.0 ** stops)
    codes = np.repeat(grey[:, None], 3, axis=1)
    return stops, np.asarray(transform(codes), dtype=np.float64)


def find_active_stop_range(stops, rgb):
    """Stop range where any channel actively transitions between its
    plateaus (display DCTL find_active_stop_range, vectorised).

    Left boundary: earliest channel to depart its shadow plateau.
    Right boundary: last channel to converge onto its highlight
    plateau. Falls back to the full range if nothing is detected."""
    d = density_of(rgb)
    n = d.shape[0]
    d_span = d.max(axis=0) - d.min(axis=0)
    sh_thresh = ACTIVE_RANGE_THRESH_SHADOWS * d_span
    hl_thresh = ACTIVE_RANGE_THRESH_HIGHLIGHTS * d_span

    left, right = n, -1
    for ch in range(3):
        if sh_thresh[ch] > 1.0e-4:
            hits = np.nonzero(
                np.abs(d[:, ch] - d[0, ch]) > sh_thresh[ch])[0]
            if hits.size:
                left = min(left, int(hits[0]))
        if hl_thresh[ch] > 1.0e-4:
            hits = np.nonzero(
                np.abs(d[:, ch] - d[-1, ch]) > hl_thresh[ch])[0]
            if hits.size:
                right = max(right, int(hits[-1]))

    if left >= n or right < 0 or left >= right:
        return float(stops[0]), float(stops[-1])

    smin, smax = float(stops[left]), float(stops[right])
    if smax - smin < ACTIVE_RANGE_MIN_SPAN_STOPS:
        center = 0.5 * (smin + smax)
        smin = max(center - 0.5 * ACTIVE_RANGE_MIN_SPAN_STOPS,
                   float(stops[0]))
        smax = min(smin + ACTIVE_RANGE_MIN_SPAN_STOPS, float(stops[-1]))
    return smin, smax


def nice_tick_step(span, target_count):
    """Display DCTL nice_tick_step: a 1/2/5/10 step near span/target."""
    if span <= 0.0:
        return 1.0
    raw = span / float(target_count)
    mag = 10.0 ** np.floor(np.log10(max(raw, 1.0e-6)))
    norm = raw / mag
    if norm < 1.5:
        return 1.0 * mag
    if norm < 3.5:
        return 2.0 * mag
    if norm < 7.5:
        return 5.0 * mag
    return 10.0 * mag


def hd_side(transform, title, y_mode="percent", eotf="none",
            clamp=True, n=HD_SAMPLES):
    """Chart-ready data for one HD curve panel.

    y is the display-space value per sample and channel: output % for
    "percent", density (of the EOTF-linearized signal) for "density".
    The density Y axis auto-scales to the highest density within the
    displayed stop range, rounded up to the next 0.5 (display DCTL)."""
    stops, rgb = hd_probe(transform, n)
    if clamp:
        smin, smax = find_active_stop_range(stops, rgb)
    else:
        smin, smax = float(stops[0]), float(stops[-1])

    if y_mode == "density":
        y = density_of(eotf_to_linear(rgb, eotf))
        sel = (stops >= smin) & (stops <= smax)
        # display DCTL: next 0.5 above the black point (+0.05 headroom)
        y_max = max(float(np.ceil((y[sel].max() + 0.05) * 2.0) / 2.0), 0.5)
    else:
        y = rgb * 100.0
        y_max = 100.0
    return {"title": title, "stops": stops, "y": y,
            "stop_min": smin, "stop_max": smax, "y_max": y_max}


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
    QEvent, QRect, QSettings, Qt, QThread, QTimer, Signal,
)
from PySide6.QtGui import (  # noqa: E402
    QColor, QFont, QFontMetrics, QImage, QPainter, QPen,
)
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QSplitter,
    QToolButton, QVBoxLayout, QWidget,
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
                    # A LUT on B is a complete transform: no DRT on top.
                    # Without a LUT, B is the untouched frame viewed
                    # through the same DRT setting as A.
                    lut = request.get("lut")
                    original = render_chain(
                        request["img"], [], request["pool"],
                        drt=None if lut is not None else drt, lut=lut)
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
        self._hd = None             # {"y_mode", "sides": [side, side]}
        self._zoom = 1.0            # trackpad pinch zoom (1 = fit)
        self._pan = [0.0, 0.0]      # px offset of the zoomed image center
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

    def set_hd(self, hd):
        self._hd = hd
        self.update()

    def _base_rect(self):
        """Viewport-fit rect at zoom 1 (also the overlay canvas)."""
        if self._frame is None:
            return None
        iw, ih = self._frame.width(), self._frame.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = int(iw * s), int(ih * s)
        return ((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def _image_rect(self):
        """The zoom/pan-transformed rect the image is drawn into."""
        base = self._base_rect()
        if base is None:
            return None
        x, y, w, h = base
        zw, zh = w * self._zoom, h * self._zoom
        cx = x + w / 2.0 + self._pan[0]
        cy = y + h / 2.0 + self._pan[1]
        return (int(round(cx - zw / 2.0)), int(round(cy - zh / 2.0)),
                int(round(zw)), int(round(zh)))

    # ---- trackpad zoom (Safari-style pinch, scroll pans, dbl-click
    # resets; Cmd/Ctrl + scroll wheel also zooms)

    def _apply_zoom(self, factor, pos):
        old = self._zoom
        self._zoom = min(max(self._zoom * factor, 1.0), 16.0)
        f = self._zoom / old
        base = self._base_rect()
        if base is not None:
            bx, by, bw, bh = base
            bcx, bcy = bx + bw / 2.0, by + bh / 2.0
            cx, cy = bcx + self._pan[0], bcy + self._pan[1]
            # keep the point under the cursor fixed while scaling
            self._pan[0] = pos.x() + f * (cx - pos.x()) - bcx
            self._pan[1] = pos.y() + f * (cy - pos.y()) - bcy
        self._clamp_pan()
        self.update()

    def _clamp_pan(self):
        if self._zoom <= 1.0:
            self._pan = [0.0, 0.0]
            return
        base = self._base_rect()
        if base is None:
            return
        _, _, w, h = base
        mx = (w * self._zoom - w) / 2.0
        my = (h * self._zoom - h) / 2.0
        self._pan[0] = min(max(self._pan[0], -mx), mx)
        self._pan[1] = min(max(self._pan[1], -my), my)

    def event(self, ev):
        if ev.type() == QEvent.Type.NativeGesture \
                and ev.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            self._apply_zoom(1.0 + ev.value(), ev.position())
            return True
        return super().event(ev)

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.ControlModifier:      # Cmd on macOS
            self._apply_zoom(2.0 ** (ev.angleDelta().y() / 600.0),
                             ev.position())
        elif self._zoom > 1.0:
            d = ev.pixelDelta()
            if d.isNull():
                d = ev.angleDelta() / 4
            self._pan[0] += d.x()
            self._pan[1] += d.y()
            self._clamp_pan()
            self.update()
        ev.accept()

    def mouseDoubleClickEvent(self, event):
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self.update()

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

        # overlays live on the same zoomed canvas — pinch into the HD
        # chart to inspect a curve region up close (fonts scale along)
        if self._curve is not None:
            self._paint_curve(p, x, y, w, h)
        if self._hd is not None:
            self._paint_hd(p, x, y, w, h)
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

    # HD curve chart colors (display DCTL palette)
    _HD_BG = QColor(13, 13, 13)             # 0.05
    _HD_GRID = QColor(56, 56, 56)           # 0.22 (major / border)
    _HD_GRID_MINOR = QColor(36, 36, 36)     # 0.14 (half-decade lines)
    _HD_LABEL = QColor(217, 217, 217)       # 0.85
    _HD_MID_GREY = QColor(217, 166, 51)     # 0.85/0.65/0.20
    _HD_RGB = (QColor(255, 38, 13), QColor(13, 255, 13),
               QColor(26, 140, 255))

    def _paint_hd(self, p, x, y, w, h):
        """One sensitometric chart over the image, in the display
        DCTL's datasheet layout (log exposure on top, camera stops
        along the bottom), with A and B in the same coordinate system:
        split at the wipe divider, or both at once with B muted."""
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(x, y, w, h, QColor(0, 0, 0, 150))
        pad = max(int(min(w, h) * 0.02), 8)
        font = p.font()
        font.setPointSizeF(max(8.0, h * 0.022))
        p.setFont(font)
        split_x = x + int(w * self._wipe)
        # A/B titles in the overlay corners, clear of the chart bands
        fm = QFontMetrics(font)
        p.setPen(self._HD_LABEL)
        p.drawText(QRect(x + pad, y + pad, w - 2 * pad, fm.height()),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   self._hd["a"]["title"])
        p.setPen(self._HD_LABEL if self._hd["mode"] == "wipe"
                 else QColor(150, 150, 150))
        p.drawText(QRect(x + pad, y + pad, w - 2 * pad, fm.height()),
                   Qt.AlignRight | Qt.AlignVCenter,
                   self._hd["b"]["title"])
        top = y + pad + fm.height() + 2
        self._paint_hd_plot(p, x + pad, top, w - 2 * pad,
                            h - pad - (top - y), self._hd, split_x)

    def _paint_hd_plot(self, p, px, py, pw, ph, hd, split_x):
        density = hd["y_mode"] == "density"
        label_font = QFont(p.font())
        legend_font = QFont(label_font)
        legend_font.setPointSizeF(label_font.pointSizeF() * 0.72)
        fm = QFontMetrics(label_font)
        lfm = QFontMetrics(legend_font)
        label_h, legend_h = fm.height(), lfm.height()
        gap = max(int(label_h * 0.25), 2)

        smin, smax = hd["stop_min"], hd["stop_max"]
        # The datasheet X axis lives in log10-exposure decades.
        xmin, xmax = smin * LOG10_2, smax * LOG10_2
        xspan = max(xmax - xmin, 1.0e-4)
        y_max = hd["y_max"]
        ystep = 0.5 if density else nice_tick_step(y_max, 8)
        ylab = (lambda v: f"{v:.1f}") if density else (lambda v: f"{v:.0f}")

        # Margins hug their content (display DCTL v2.6): left fits the
        # widest Y label, top/bottom fit one numbers row + legend row.
        ylab_w = 0
        v = 0.0
        while v <= y_max + 1.0e-6:
            ylab_w = max(ylab_w, fm.horizontalAdvance(ylab(v)))
            v += ystep
        m_left = ylab_w + gap + 6
        m_right = max(int(label_h * 0.8), 8)
        m_top = legend_h + label_h + gap
        m_bot = label_h + legend_h + gap
        avail_x, avail_y = px + m_left, py + m_top
        avail_w, avail_h = pw - m_left - m_right, ph - m_top - m_bot
        if avail_w < 50 or avail_h < 50:
            return

        # Plot Shape (display DCTL): Fill Width uses all space; Square
        # keeps a square; Datasheet 1:1 sizes one density unit equal to
        # one log-exposure decade in pixels so slopes read as true
        # gamma. 1:1 needs the Density Y axis (Output % has no fixed
        # log-unit size) and falls back to Fill Width otherwise.
        plot_w, plot_h = avail_w, avail_h
        if hd["shape"] == 2 and density:
            ppu = min(avail_w / xspan, avail_h / y_max)
            plot_w = int(round(xspan * ppu))
            plot_h = int(round(y_max * ppu))
        elif hd["shape"] == 1:
            plot_w = plot_h = min(avail_w, avail_h)
        ax = avail_x + (avail_w - plot_w) // 2
        ay = avail_y + (avail_h - plot_h) // 2

        p.fillRect(ax - m_left, ay - m_top, plot_w + m_left + m_right,
                   plot_h + m_top + m_bot, self._HD_BG)

        def x_px(decades):
            return ax + int(round((decades - xmin) / xspan * plot_w))

        def y_px(frac):
            return ay + plot_h - int(round(frac * plot_h))

        # X grid: half-decade minor lines, full-decade major.
        half_lo = int(np.ceil(xmin * 2.0))
        half_hi = int(np.floor(xmax * 2.0))
        for i in range(half_lo, half_hi + 1):
            color = self._HD_GRID if i % 2 == 0 else self._HD_GRID_MINOR
            p.setPen(QPen(color, 1))
            tx = x_px(i * 0.5)
            p.drawLine(tx, ay, tx, ay + plot_h)

        # Y grid + labels (left, right-aligned, centered on the tick)
        v = 0.0
        while v <= y_max + 1.0e-6:
            ty = y_px(v / y_max)
            p.setPen(QPen(self._HD_GRID, 1))
            p.drawLine(ax, ty, ax + plot_w, ty)
            p.setPen(self._HD_LABEL)
            p.setFont(label_font)
            p.drawText(QRect(ax - m_left, ty - label_h // 2,
                             m_left - gap - 4, label_h),
                       Qt.AlignRight | Qt.AlignVCenter, ylab(v))
            v += ystep

        # Mid grey (18% input / 0 stops) at its exact position.
        if xmin <= 0.0 <= xmax:
            p.setPen(QPen(self._HD_MID_GREY, 2))
            gx = x_px(0.0)
            p.drawLine(gx, ay, gx, ay + plot_h)

        # Border: same color as the grid, twice as thick.
        p.setPen(QPen(self._HD_GRID, 2))
        p.drawRect(ax, ay, plot_w, plot_h)

        # Top band: log exposure numbers with the "Log Exp" legend
        # above them. Square views label integers at full decades only
        # (the DCTL's square/right-overlay rule); other shapes label
        # every half decade with one decimal.
        square = hd["shape"] == 1
        p.setFont(label_font)
        p.setPen(self._HD_LABEL)
        top_y = ay - gap - label_h
        for i in range(half_lo, half_hi + 1):
            if square and i % 2 != 0:
                continue
            tick = i * 0.5
            text = f"{tick:.0f}" if square else f"{tick:.1f}"
            p.drawText(QRect(x_px(tick) - 60, top_y, 120, label_h),
                       Qt.AlignHCenter | Qt.AlignVCenter, text)
        p.setFont(legend_font)
        p.drawText(QRect(ax, top_y - legend_h, plot_w, legend_h),
                   Qt.AlignHCenter | Qt.AlignVCenter, "Log Exp")

        # Bottom band: camera stops, every other integer, "Stops"
        # legend centered below.
        p.setFont(label_font)
        bot_y = ay + plot_h + gap
        for i in range(int(np.ceil(smin)), int(np.floor(smax)) + 1):
            if i % 2 != 0:
                continue
            text = "0" if i == 0 else f"{i:+d}"
            p.drawText(QRect(x_px(i * LOG10_2) - 60, bot_y, 120, label_h),
                       Qt.AlignHCenter | Qt.AlignVCenter, text)
        p.setFont(legend_font)
        p.drawText(QRect(ax, bot_y + label_h, plot_w, legend_h),
                   Qt.AlignHCenter | Qt.AlignVCenter, "Stops")

        # R/G/B curves (blue under green under red, like the DCTL's
        # blend order). "Wipe": A left of the divider, B right of it —
        # both in this one coordinate system, meeting at the split.
        # "Both": B muted underneath, A on top.
        def draw_side(side, colors, clip=None):
            if clip is not None:
                p.save()
                p.setClipRect(clip)
            stops, yv = side["stops"], side["y"]
            sel = np.nonzero((stops >= smin) & (stops <= smax))[0]
            for ch in (2, 1, 0):
                p.setPen(QPen(colors[ch], 2))
                pts = []
                for i in sel:
                    sx = ax + (stops[i] * LOG10_2 - xmin) / xspan * plot_w
                    frac = min(max(yv[i, ch] / y_max, 0.0), 1.0)
                    pts.append((sx, ay + plot_h - frac * plot_h))
                for a, b in zip(pts, pts[1:]):
                    p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            if clip is not None:
                p.restore()

        if hd["mode"] == "wipe":
            sx = min(max(split_x, ax), ax + plot_w)
            draw_side(hd["a"], self._HD_RGB,
                      QRect(ax, ay, sx - ax, plot_h))
            draw_side(hd["b"], self._HD_RGB,
                      QRect(sx, ay, ax + plot_w - sx, plot_h))
            p.setPen(QPen(QColor(255, 255, 255, 200), 1))
            p.drawLine(sx, ay, sx, ay + plot_h)
        else:
            muted = tuple(QColor(c.red(), c.green(), c.blue(), 105)
                          for c in self._HD_RGB)
            draw_side(hd["b"], muted)
            draw_side(hd["a"], self._HD_RGB)

    def mousePressEvent(self, event):
        self._drag_wipe(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._drag_wipe(event)

    def _drag_wipe(self, event):
        rect = self._image_rect()
        hd_wipe = self._hd is not None and self._hd["mode"] == "wipe"
        if not (self._wipe_on or hd_wipe) or rect is None:
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
    """label + slider + spinbox + reset button for one stage param, in
    DCTL units. The ⟲ button (or double-clicking the label) resets the
    param to identity."""

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

        self._reset = QToolButton()
        self._reset.setText("⟲")
        self._reset.setToolTip(
            f"reset to identity ({spec['identity']:g})")
        self._reset.clicked.connect(
            lambda: self.set_value(self._spec["identity"]))

        row.addWidget(self._label)
        row.addWidget(self._slider, 1)
        row.addWidget(self._spin)
        row.addWidget(self._reset)
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
    reset-to-identity, move up/down and remove buttons."""

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

        def reset_node():
            self.reset()
            on_change(interactive=False)

        for text, tip, action in (
                ("⟲", "reset all params to identity", reset_node),
                ("↑", "move node up", lambda: on_move(self, -1)),
                ("↓", "move node down", lambda: on_move(self, +1)),
                ("✕", "remove node", lambda: on_remove(self))):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(action)
            head.addWidget(b)
            if text == "⟲":
                self._reset_btn = b
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
        self._lut = None            # CubeLUT on the wipe's B side
        self._lut_path = ""
        self._self_mtime = TOOL_FILE.stat().st_mtime

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

        # B side of the wipe: untouched frame, or a .cube LUT
        lut_row = QHBoxLayout()
        lut_btn = QPushButton("B LUT…")
        lut_btn.setToolTip("Load a .cube LUT for the B side of the "
                           "wipe and the B HD curve. The LUT stands "
                           "on its own: openDRT is never applied on "
                           "top of it.")
        lut_btn.clicked.connect(self._pick_lut)
        self.lut_label = QLabel("B: original")
        self.lut_label.setStyleSheet("color:#aaa;")
        lut_clear = QToolButton()
        lut_clear.setText("✕")
        lut_clear.setToolTip("Clear the B LUT (back to the untouched "
                             "frame)")
        lut_clear.clicked.connect(self._clear_lut)
        lut_row.addWidget(lut_btn)
        lut_row.addWidget(self.lut_label, 1)
        lut_row.addWidget(lut_clear)

        # HD curves overlay (the HD Curve DCTL logic, A/B side by side)
        hd_row = QHBoxLayout()
        self.hd_box = QCheckBox("HD curves")
        self.hd_box.setToolTip("A and B sensitometric charts side by "
                               "side: ±8.26-stop LogC3 probe sweep "
                               "through chain (A) and LUT (B)")
        self.hd_y = QComboBox()
        self.hd_y.addItems(["Output %", "Density"])
        self.hd_eotf = QComboBox()
        self.hd_eotf.addItems(["None / Linear", "Gamma 2.4", "sRGB"])
        self.hd_eotf.setToolTip("Output Signal: linearize through this "
                                "EOTF before measuring density")
        self.hd_mode = QComboBox()
        self.hd_mode.addItems(["Wipe", "Both"])
        self.hd_mode.setToolTip("Wipe: A left / B right of the "
                                "draggable divider, in one coordinate "
                                "system. Both: A and B overlaid, B "
                                "muted.")
        self.hd_shape = QComboBox()
        self.hd_shape.addItems(["Fill Width", "Square", "Datasheet 1:1"])
        self.hd_shape.setToolTip("Datasheet 1:1: one density unit = "
                                 "one log-exposure decade, so slopes "
                                 "read as true gamma (needs the "
                                 "Density Y axis)")
        self.hd_clamp = QCheckBox("Clamp range")
        self.hd_clamp.setChecked(True)
        self.hd_clamp.setToolTip("Zoom the stops axis to the union of "
                                 "the A and B active ranges")
        self.hd_box.toggled.connect(
            lambda _: self.request_render(interactive=False))
        self.hd_clamp.toggled.connect(
            lambda _: self.request_render(interactive=False))
        for combo in (self.hd_y, self.hd_eotf, self.hd_mode,
                      self.hd_shape):
            combo.currentIndexChanged.connect(
                lambda _: self.request_render(interactive=False))
        for w in (self.hd_box, self.hd_mode, self.hd_y, self.hd_eotf,
                  self.hd_shape, self.hd_clamp):
            hd_row.addWidget(w)
        hd_row.addStretch(1)

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
        imp = QPushButton("Import…")
        imp.setToolTip("Load a whole node tree from JSON (params by "
                       "name; optionally restores B LUT, openDRT and "
                       "HD-curve settings)")
        imp.clicked.connect(lambda: self._import_chain())
        paste = QPushButton("Paste")
        paste.setToolTip("Import a chain JSON straight from the "
                         "clipboard (same format as Import/Export)")
        paste.clicked.connect(self._paste_chain)
        exp = QPushButton("Export…")
        exp.setToolTip("Save the current node tree (+ B LUT path, "
                       "openDRT and HD settings) as JSON")
        exp.clicked.connect(lambda: self._export_chain())
        actions.addWidget(reset)
        actions.addWidget(copy)
        actions.addWidget(imp)
        actions.addWidget(paste)
        actions.addWidget(exp)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(160)

        col.addWidget(self.banner)
        col.addLayout(add_row)
        col.addLayout(toggles)
        col.addLayout(lut_row)
        col.addLayout(hd_row)
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

    # ---------------- JSON chain import/export ----------------

    def _export_chain(self, path=None):
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export chain", "", "Chain JSON (*.json)")
            if not path:
                return
        data = {
            "version": 1,
            "chain": chain_to_export(self.chain(), self.specs),
            "drt": self.drt_box.isChecked(),
            "hd": {"on": self.hd_box.isChecked(),
                   "mode": self.hd_mode.currentIndex(),
                   "y": self.hd_y.currentIndex(),
                   "eotf": self.hd_eotf.currentIndex(),
                   "shape": self.hd_shape.currentIndex(),
                   "clamp": self.hd_clamp.isChecked()},
        }
        if self._lut_path:
            data["lut"] = self._lut_path
        Path(path).write_text(json.dumps(data, indent=1))
        self.statusBar().showMessage(f"exported {Path(path).name} ✓", 3000)

    def _import_chain(self, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Import chain", "", "Chain JSON (*.json)")
            if not path:
                return
        try:
            data = json.loads(Path(path).read_text())
        except Exception:
            self._show_error(traceback.format_exc())
            return
        self._apply_chain_data(data, Path(path).parent, Path(path).name)

    def _paste_chain(self):
        """Import a chain JSON straight from the clipboard."""
        text = QApplication.clipboard().text().strip()
        try:
            data = json.loads(text)
        except Exception:
            self._show_error("Clipboard is not valid chain JSON:\n"
                             + traceback.format_exc())
            return
        root = Path(__file__).resolve().parents[1]
        self._apply_chain_data(data, root, "clipboard")

    def _apply_chain_data(self, data, base_dir, label):
        chain, notes = chain_from_import(data.get("chain", []), self.specs)
        for p in self.panels():
            self.chain_col.removeWidget(p)
            p.deleteLater()
        for node in chain:
            self.add_node(node["stage"], node["params"], node["bypass"])

        if "drt" in data:
            self.drt_box.setChecked(bool(data["drt"]))
        hd = data.get("hd", {})
        if hd:
            self.hd_box.setChecked(bool(hd.get("on", False)))
            self.hd_mode.setCurrentIndex(int(hd.get("mode", 0)))
            self.hd_y.setCurrentIndex(int(hd.get("y", 0)))
            self.hd_eotf.setCurrentIndex(int(hd.get("eotf", 0)))
            self.hd_shape.setCurrentIndex(int(hd.get("shape", 0)))
            self.hd_clamp.setChecked(bool(hd.get("clamp", True)))

        lut_ref = data.get("lut", "")
        if lut_ref:
            resolved = self._resolve_lut_path(lut_ref, base_dir)
            if resolved:
                self._set_lut(resolved)
            else:
                notes.append(f"B LUT not found: {lut_ref}")
        if notes:
            self._show_error("Import notes:\n" + "\n".join(notes))
        else:
            self.banner.hide()
        self._need_original = True
        self.request_render()
        self.statusBar().showMessage(f"imported {label} ✓", 3000)

    @staticmethod
    def _resolve_lut_path(ref, json_dir):
        """Try the reference as-is, relative to the JSON file, then
        relative to the project root (colorchecker/)."""
        root = Path(__file__).resolve().parents[1]
        for cand in (Path(ref), json_dir / ref, root / ref):
            if cand.is_file():
                return str(cand)
        return ""

    # ---------------- B-side LUT ----------------

    def _pick_lut(self):
        start = str(Path(self._lut_path).parent) if self._lut_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose B-side LUT", start, "Cube LUT (*.cube)")
        if path:
            self._set_lut(path)

    def _set_lut(self, path):
        try:
            lut = parse_cube(path)
        except Exception:
            self._show_error(traceback.format_exc())
            return
        self._lut = lut
        self._lut_path = str(path)
        self.lut_label.setText(f"B: {Path(path).stem}")
        self._need_original = True
        self.request_render()

    def _clear_lut(self):
        if self._lut is None:
            return
        self._lut = None
        self._lut_path = ""
        self.lut_label.setText("B: original")
        self._need_original = True
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
            "lut": self._lut,
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
            self.viewer.set_hd(
                self._hd_data(chain) if self.hd_box.isChecked() else None)
            self.report.setPlainText(chain_report(chain, self.host.pool))
        except Exception:
            self._show_error(traceback.format_exc())

    def _hd_data(self, chain):
        """A and B HD curves through the same paths as the wipe
        (A = chain (+openDRT), B = LUT or untouched (+openDRT)),
        merged into ONE coordinate system: the stop range is the union
        of both active ranges, the density axis fits both curves."""
        pool = self.host.pool
        drt = self.host.drt if self.drt_box.isChecked() else None
        y_mode = "density" if self.hd_y.currentIndex() == 1 else "percent"
        eotf = ("none", "g24", "srgb")[self.hd_eotf.currentIndex()]
        clamp = self.hd_clamp.isChecked()

        def through(fn):
            return (lambda codes: drt(fn(codes))) if drt else fn

        lut = self._lut
        kwargs = dict(y_mode=y_mode, eotf=eotf, clamp=clamp)
        a = hd_side(through(lambda c: apply_chain(c, chain, pool)),
                    "A · chain", **kwargs)
        # The LUT stands on its own (no DRT on top); without a LUT the
        # B curve is the bare probe through the same DRT setting as A.
        if lut is not None:
            b = hd_side(lambda c: apply_lut(lut, c),
                        f"B · {Path(self._lut_path).stem}", **kwargs)
        else:
            b = hd_side(through(lambda c: c), "B · original", **kwargs)

        smin = min(a["stop_min"], b["stop_min"])
        smax = max(a["stop_max"], b["stop_max"])
        y_max = max(a["y_max"], b["y_max"])
        if y_mode == "density":
            # re-fit the density axis to both curves over the union
            sel = (a["stops"] >= smin) & (a["stops"] <= smax)
            dmax = max(a["y"][sel].max(), b["y"][sel].max())
            y_max = max(float(np.ceil((dmax + 0.05) * 2.0) / 2.0), 0.5)
        return {"y_mode": y_mode,
                "mode": "wipe" if self.hd_mode.currentIndex() == 0
                else "both",
                "shape": self.hd_shape.currentIndex(),
                "stop_min": smin, "stop_max": smax, "y_max": y_max,
                "a": a, "b": b}

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
        self._poll_self_restart()
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

    def _poll_self_restart(self):
        """Auto-restart when tools/dev_preview.py itself changes — the
        npm-run-dev feel for the UI shell too. State survives through
        QSettings (chain, sliders, LUT, toggles, window geometry). A
        file that does not compile shows as a banner instead; the next
        good save restarts."""
        try:
            mtime = TOOL_FILE.stat().st_mtime
        except OSError:
            return
        if mtime == self._self_mtime:
            return
        self._self_mtime = mtime
        try:
            py_compile.compile(str(TOOL_FILE), doraise=True)
        except py_compile.PyCompileError:
            self._show_error(traceback.format_exc())
            return
        if self._persist:
            self._save_state(self.chain())
            self._settings.setValue("geometry", self.saveGeometry())
            self._settings.sync()
        self.worker.stop()
        os.execv(sys.executable,
                 [sys.executable, "-m", "tools.dev_preview"]
                 + sys.argv[1:])

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
        self._settings.setValue("lut_path", self._lut_path)
        self._settings.setValue("hd", self.hd_box.isChecked())
        self._settings.setValue("hd_clamp", self.hd_clamp.isChecked())
        self._settings.setValue("hd_y", self.hd_y.currentIndex())
        self._settings.setValue("hd_eotf", self.hd_eotf.currentIndex())
        self._settings.setValue("hd_mode", self.hd_mode.currentIndex())
        self._settings.setValue("hd_shape", self.hd_shape.currentIndex())

    def _restore_state(self):
        raw = self._settings.value("chain", "")
        for box, key in ((self.drt_box, "drt"), (self.curve_box, "curve"),
                         (self.wipe_box, "wipe"), (self.hd_box, "hd"),
                         (self.hd_clamp, "hd_clamp")):
            v = self._settings.value(key)
            if v is not None:
                box.setChecked(v in (True, "true"))
        for combo, key in ((self.hd_y, "hd_y"), (self.hd_eotf, "hd_eotf"),
                           (self.hd_mode, "hd_mode"),
                           (self.hd_shape, "hd_shape")):
            v = self._settings.value(key)
            if v is not None:
                combo.setCurrentIndex(int(v))
        lut_path = self._settings.value("lut_path", "")
        if lut_path and Path(lut_path).is_file():
            self._set_lut(lut_path)
        geometry = self._settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
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
        if self._persist:
            self._settings.setValue("geometry", self.saveGeometry())
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
