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

Building the Processing (readout) tab first. Matching (RBF) and
LUT Inspector tabs are placeholders wired into the tab router for later.
