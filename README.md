# color-science

Tools for film-emulation color work: raw conversions, DCTLs, and a
Python **film-look engineering toolkit** (`colorchecker/`).

## colorchecker/

A macOS (Apple Silicon), Python/PySide6 chart-readout tool grown into a
film-look engineering toolkit. It reads ColorChecker patches from
scene-referred footage, fits **interpretable parametric look chains** to
match a source to a target (or explain a LUT), and exports the result as
DCTL slider values, `.cube` files, and ready-to-import DaVinci Resolve
PowerGrades (`.drx`).

**The hard invariant:** zero color management in the measurement path —
input pixels equal output pixels. No ICC, gamma, clamp, or
normalization between the TIFF and the CSV. Emissive values above 1.0
pass through raw.

### What it does

- **Read** — load scene-referred TIFFs, auto-detect the chart, sample
  every patch; multi-exposure / Kelvin / hue sweeps; emissive-light
  overlays; batch import and CSV export.
- **Match** — fit source→target and export a `.cube`. Two solvers: an
  RBF scattered-data fit, and a **parametric chain** of interpretable
  film tools that a free-order search assembles automatically.
- **The parametric tools** operate in an Arri LogC3 / AWG3 pipeline and
  are each mirrored as a DCTL:
  - **Contrast Curve** — a bounded film S with independent toe/shoulder
    (Black/White Point levels, Toe/Shoulder Length + Strength), a
    movable mid, and a mid-grey-referenced achromatic exposure.
  - **Split Tone** — per-channel cubic-Bezier shadow/highlight split
    with an optional per-channel crossover (subtractive, in log RGB).
  - **Colour Saturation / Colour Crosstalk / Highlight Bleach /
    Brilliance Reduction** and a **Sector** family (Skew / Brightness /
    Saturation / Squash) — surgical single-hue moves.
  - All in the **Reuleaux** opponent color space where hue/chroma work
    is cleanest.
- **DRT sandwich** — fit *under* a display transform. Includes an
  analytic Python port of **OpenDRT** (Jed Smith) matching Marc's
  Resolve config, so the fit optimizes the display-domain result.
- **Explain a LUT** — decode a `.cube` into a stack of these tools with
  a per-stage error waterfall and paste-ready slider values.
- **Export a PowerGrade** — generate a complete `.drx` from scratch:
  clone exactly the fitted chain (+ the DRT), in order, each node
  labeled, sliders patched.

### Quick start

```
cd colorchecker
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 main.py                       # the app
QT_QPA_PLATFORM=offscreen python3 -m pytest tests/   # the tests (headless)
```

Explain a LUT and drop a PowerGrade next to it:

```
python3 -m tools.lut_match --lut somelook.cube --search \
    --target-is-display --drt-math --drx-out fitted.drx
```

See `colorchecker/HANDOFF.md` for the full project map and the current
open threads, and `colorchecker/ROADMAP.md` for planned work.
