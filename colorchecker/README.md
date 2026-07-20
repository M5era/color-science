# Color Checker

Chart readout tool for film-stock profiling: load scene-referred TIFFs,
auto-detect a ColorChecker chart, sample every patch, export the values.

Pixel values pass through untouched — no color management, no gamma, no
clamping. What is in the file is what lands in the export.

## Requirements

macOS (Apple Silicon), Python 3.11+.

## Setup

```
cd colorchecker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```
python3 main.py
```

## Tests

```
pip install pytest
python3 -m pytest tests/
```

Tests generate their own tiny synthetic TIFFs at runtime; no footage is
committed to the repo (`*.tif` is git-ignored — keep real frames in
`testdata/` locally if you want them handy).

## Status

All three tabs are built and working on real footage:

- **Processing** — load TIFFs, auto-detect the chart, sample every
  patch, multi-exposure sessions, emissive light overlays, batch
  import / Process All, CSV export.
- **Matching** — fit source→target and export a `.cube`, with two
  solvers: **RBF** (hierarchical, optional matrix pre-fit) and
  **Parametric** (an ordered chain of Matrix / Luma / RGB-curve /
  Reuleaux Broad (fixed 6 anchors) / Reuleaux Fine (free 360° hue zone
  with smooth sat + luma masks) stages, ML-ready, with a per-stage
  error waterfall and paste-ready Reuleaux slider values).
  Scene-referred or
  display-referred (sandwich fit under a fixed DRT).
- **LUT Inspector** — load a `.cube` and inspect it: image preview,
  RGB response curves, 3D lattice.

Plus `tools/reuleaux_bake.py`, a CLI that bakes reuleaux parameters
into a `.cube` for A/B against the real DCTL in Resolve.

See `HANDOFF.md` for the full project map and `ROADMAP.md` for the
planned zone-model / DCTL / PowerGrade work (Plan B).
