# Color Checker — Project Handoff

Read this first in a new session. It is the map: what the tool is, what
is built, the non-negotiable decisions, the resources, and where we
left off. Deep design lives in `ROADMAP.md`; this is the orientation
layer.

---

## 1. What this is

A macOS (Apple Silicon), Python/PySide6 clone-and-extension of Nico
Fink's (Demystify-Color) chart-readout tool, built for **film-stock
profiling**. Marc shoots reference charts under controlled variations,
reads the patch values here, and fits transforms that emulate a film
stock (log → negative/Cineon, or log → display-referred reversal like
E100 slide film).

**Capture pack** Marc profiles against:
- Alexa Mini LF, LogC3 (AWG / EI800) footage, float/16-bit TIFF
- ColorChecker Digital SG charts (the 96-patch layout)
- EV sweeps (±5), Kelvin sweeps (2700K–6000K), full hue swings (Hue 0–300)
- Some frames contain an Aputure light **in frame** — an emissive
  source sampled alongside the reflective chart

**Dev model:** development happens in the cloud (Linux, headless Qt via
`QT_QPA_PLATFORM=offscreen`). Marc pulls the branch and runs it on his
Mac. So: everything must be testable offscreen, and the core is kept
UI-free so it can be exercised without a display.

---

## 2. THE hard invariant (never violate)

**Zero color management in the measurement path. Input = output.**
No ICC, no gamma, no clamp, no normalization anywhere between the TIFF
pixels and the CSV. What is in the file is what lands in the export.
Emissive values >1.0 pass through raw. This was stated by Marc in the
strongest possible terms ("do NOT change any of the displayed RGB
values, that is the whole point!!!!"). The on-screen preview is the
ONLY place pixels are transformed, it is strictly one-way and display-
only (`app/core/preview.py`), and it never touches measured values.

Bit-exactness of the TIFF decode was proved first, as Phase 0 risk
retirement (`test_image_io.py`).

---

## 3. Architecture

```
colorchecker/
  main.py                      launch
  app/core/                    UI-FREE logic (all headless-testable)
    image_io.py                bit-exact TIFF load; EV/Kelvin/Hue filename parsing
    project.py                 ProjectStore + ImageEntry (the "database"); schema_version=1, unknown keys preserved
    overlay.py                 Overlay model (uid-keyed!), presets incl. Light Source 1x1 emissive
    detect.py / refine.py      chart auto-detect (patch-mosaic blobs) + margin refinement
    homography.py / sampler.py DLT unit-square->quad; pixel-center-in-quad sampling (pure numpy)
    csv_export.py              deterministic export (auto-sorted); header below
    preview.py                 display-only one-way transform (NOT measurement)
    lut.py                     CubeLUT parse/apply (trilinear, domain clamp), gradients, lattice
    match.py                   HierarchicalRBF + solve_match; sandwich DRT inversion; CSV/session loaders
    reuleaux.py                1:1 port of the reuleaux DCTL math (validated in Resolve)
    stages.py                  parametric stages (Matrix/Luma/RGB/Reuleaux) — ML-ready contract
    parametric.py              solve_parametric: stage chain fit + waterfall + paste-ready reports
  app/ui/                      PySide6; tabs/ = the three tabs
  tools/reuleaux_bake.py       CLI: bake reuleaux (Broad) params into a .cube for Resolve A/B
  tools/reuleaux_fine_bake.py  CLI: bake a Fine zone into a .cube (units = DCTL sliders)
  dctl/ReuleauxFine.dctl       companion DCTL for the Fine stage — solver report pastes in 1:1
  tests/                       82 tests, all green, offscreen
  ROADMAP.md                   Plan B and future design (detailed)
  HANDOFF.md                   this file
```

**CSV header:** `label,ev,group,overlay,kind,patch_row,patch_col,R,G,B`
Export order is deterministic: session order (auto-sorted by EV then
label) → overlay order → row-major. Project JSON is the source of
truth; CSV is a view.

**Why this shape:** UI-free core means (a) headless cloud testing and
(b) the toolkit could be swapped without rewriting the logic. Tests
drive REAL interaction paths (canvas signals, mocked file dialogs), not
just direct field-setting — a lesson learned after early offscreen
tests missed real click-path bugs.

---

## 4. What is built (all three tabs + solvers)

### Processing (readout) tab — DONE
- Load float/16-bit TIFF (÷65535 for 16-bit), broad rubber-band rectangle
  → auto-detect chart → refine → sample every patch.
- Multi-exposure **session list**: many frames in one project,
  non-destructive. Auto-sorted always (`(ev is None, ev, label.lower())`).
- Filename parsing: EV (`+1_EV`, `EV-1`, etc.), Kelvin (`5600K`), Hue
  (`hue120`). Defaults for unmarked files: **EV 0**, **group 5600K**.
- **Batch import** (Load Folder), **Process All** (one click, all frames),
  preview updates on frame switch, **arrow-key** navigation.
- **Overlays**: multiple per frame. Presets incl. an **emissive Light
  Source (1×1)** for the Aputure panel (portrait 1×1, `kind="emissive"`,
  tagged in CSV). Every overlay carries a `uid`.
- **Per-frame overlay control (uid-keyed, v2):** enable/disable an
  overlay on a given frame ("Use on this frame"), and override its
  position for one frame only ("Position for this frame only") — for
  frames that shifted slightly. Keyed by overlay `uid`, NOT name (name
  collisions after add/remove were the v1 bug).
- **Multi-row batch edit** in the table (apply a change to all selected).
- Results panel shows **all active overlays side by side**.
- CSV export + preview; project save/load.

### Matching tab — DONE (two solvers)
Fit source patches (footage) to target patches (reference), export a
.cube. Both sides come from the current session or a loaded CSV; rows
pair up by shared ordering.

- **Match type:** Scene-referred (log→log) OR **Display-referred through
  a fixed DRT**. Display-referred = the "sandwich": invert the DRT
  numerically at the target patch values, solve underneath it, export a
  cube you stack BEFORE the DRT node. Patches clipped by the stock
  (D-max/D-min plateaus) are dropped + counted. Errors reported THROUGH
  the DRT (what the eye sees). Validated on Marc's real ODTs (openDRT:
  1435/1449 invertible; referent hard-toe: 1294/1449).

- **Solver = RBF:** vendored, improved HierarchicalRBF (from Marc's
  camera-match fork), ~60× faster than the fork (~0.7s vs 43s), with
  optional 3×3 matrix pre-fit, smoothness/detail-layer/strength knobs,
  out-of-domain extrapolation via direct model eval.

- **Solver = Parametric (newest):** an ordered chain of parametric
  stages — **Matrix (9), Luma Curve (monotone), RGB Curves (3×monotone,
  split-tone), Reuleaux Broad (20, the validated fixed-6-anchor port),
  Reuleaux Fine (12, one freely placed 360° hue zone with smooth hue
  window + sat mask + luma mask — chain several for several zones;
  masks are plateau windows with cos² shoulders, `app/core/windows.py`,
  C¹-smooth by construction, wide-open = off at identity)**. Chain
  presets: *Full (Luma→RGB→Reuleaux Broad)* / *Reuleaux Broad only* /
  *Matrix + Reuleaux Broad* / *Reuleaux Broad + Fine* /
  Custom, with add/remove/reorder and a curve-point count. Solve =
  stagewise coordinate descent → joint bounded least-squares with
  identity regularization. Output shows a **per-stage error waterfall**
  and paste-ready stage reports — including the **Reuleaux fitted slider
  values formatted for `ReuleauxUserStandalone.dctl`**, so the match can
  be rebuilt parametrically in Resolve. Every stage is a pure function
  over a flat param vector with box bounds + identity anchor — the
  **ML-ready contract**: swap numpy→torch / finite-diff→autograd later
  and the architecture holds (backprop is the planned next step).

- Export .cube with size + domain min/max controls (guard rejects
  domain width <0.05 — a zero-width domain once produced an all-white
  cube).

### LUT Inspector tab — DONE
Load a .cube and *see what it does* (explicitly NOT smoothing it yet):
image preview (default hue/value gradient or a loaded TIFF), per-channel
RGB response curves, and a rotatable/auto-fit 3D lattice view.

### Reuleaux port + bake CLI — DONE, validated in Resolve
`app/core/reuleaux.py` is a 1:1 port of the reuleaux DCTL math
(RGB→(hue,sat,val) spherical-ish model, 6 hue anchors × hue/sat/val +
overall sat/val). `tools/reuleaux_bake.py` bakes chosen params into a
.cube for A/B against the real DCTL. Marc confirmed: **"sick! it
works."**

---

## 5. Resources

**Reference repos (cloned in earlier sessions under `/workspace/`):**
- `camera-match` — Marc's fork; source of the RBF (now vendored/improved).
- `Demystify-Color-DCTLs` (M5era fork, MIT, Nico Fink) — **the Plan B
  blueprint.** Ships `OKLAB.dctl` (AWG↔Oklab with signed cbrt + gamut
  guard), `LCHab.dctl` (MISNAMED — it's OkLCh), Log-C↔Linear (Arri LogC3
  EI800 constants), `DMC_3x3Matrix` v1–v3 (**v3 "Preserve Neutrals" is
  sequential, not a true matrix — porting hazard**). Encrypted .dctle
  (PiecewisePower, etc.) = black-box only.
- `reuleaux` (hotgluebanjo) — **NO LICENSE.** The port is for Marc's
  private evaluation only and must NOT be redistributed. `resolve/
  Reuleaux.dctl`, `extra/ReuleauxUserStandalone.dctl`.

**Marc's uploads** (`/root/.claude/uploads/<session>/`, may not persist):
- `all_EV0.csv` — 1449 rows, 15 frames (measured reference set).
- `referent_LOGC3_to_sRGB.cube` — a DRT (hard toe).
- `openDRT_LogC3_srgb_3...cube` — a more neutral DRT.
- `K64_1.0_1.5.2.drx` — a PowerGrade; DRX format reverse-engineered from it.

**Licensing rules:** Demystify DCTLs MIT (credit). reuleaux no-license
(private eval, never redistribute). Commercial .dctle encrypted
(black-box, never decrypt).

---

## 6. Future plans (see ROADMAP.md for full design)

**Plan B — parametric zone model (planned, not started; do after the
current toolchain is proven on real footage):**
- Up to ~20 SIMPLE single-zone nodes in **OkLCh** (chosen for hue
  linearity; blueprint = Nico's OKLAB.dctl, adopt constants verbatim
  with credit). Each node ~7 sliders (hue anchor+width, lum center+width,
  Δhue/Δchroma/Δlightness, Gaussian falloff). Solver grows the chain
  greedily and **auto-names** nodes by their job ("light green desat",
  "bleach highlights"). Fixed-6 "reuleaux mode" (cos² partition-of-unity
  weights) + free-anchor solver mode.
- **Prismatic saturation** (Hart 2015 prismatic color space) — the open
  equivalent of Nico's Advanced Natural Saturation (subtractive,
  film-like, no garish high-sat highlights). Plus an **OkLab saturation**
  alternative (6 fixed hue-vector sliders). Same fitter slot; keep the
  lower-error one.
- **Own companion DCTLs with parameter parity** — fitted numbers paste
  straight into Resolve; mandatory Resolve pixel-match verification gate.

**PowerGrade (.drx) generation — feasibility PROVEN & verified in
Resolve:** XML wrapper + zstd-compressed protobuf body; slider doubles
are fixed-width so **template patching** (clone a template .drx, write
fitted values, recompress) changes no lengths. Open question: native
custom-curve encoding in DRX (K64 file has no curve data) — experiment
defined (diff a default vs single-known-point drx).

**Backup plan only** (Marc's call): black-box fitting of encrypted
commercial .dctle via Resolve scripting API (patch drx → grab still →
measure → optimize). Only if the own-DCTL path proves insufficient.

**Chart-prep idea:** datasheet D-logE alignment tool (Nico's Film
Profile Journey #22 automated) — align measured grayscale to the
stock's published sensitometric curves before profiling. Works for
negative and reversal.

**Backprop is BUILT** (this session): `backend="torch"` on
solve_parametric / "Backprop refine (PyTorch)" checkbox in the UI
(disabled with an install hint if torch is missing — torch is an
OPTIONAL dep, deliberately NOT in requirements.txt). Torch mirrors of
all five stages (`app/core/torch_stages.py`, parity-tested to 1e-9
against the numpy stages), Adam over a sigmoid box-bounds
reparametrization, **multi-restart hue placement for Fine zones**
(`app/core/backprop.py`), then the existing scipy joint refine
polishes the winner — so torch can only improve on scipy. Proven in
tests: a Fine zone hidden in the greens that scipy provably cannot
find (zero finite-diff gradient, no window overlap from the red start)
is found and fit by the torch backend.

**Next up:** the separate **HueSquash node** (compress nearby hues
toward a chosen target; sat-gated; foldover-proof monotone
parametrization — design agreed with Marc, in chat). OkLab/OkLCh is
explicitly OFF the table for now — everything stays in reuleaux space
until Marc says otherwise.

---

## 7. Dev workflow / gotchas

- **Branch:** PRs #1/#2 (`claude/color-checker-reader-tool-z9q2ts`)
  are MERGED into `main`. Current work happens on
  `claude/color-checker-handoff-91whob` (branched from main).
- **Run on Mac:** `python3 main.py` from `colorchecker/`. Install deps
  with `python3 -m pip install -r requirements.txt` — **use
  `python3 -m pip`, NOT `pip3`** (repo path has spaces, which breaks the
  pip script shim). `scipy` IS required (a missing entry once crashed
  the app on import — it's in requirements now).
- **Tests:** `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/` — 104
  green, ~22–35s (torch tests auto-skip if torch is not installed).
  Synthetic TIFFs generated at runtime; no footage
  committed (`*.tif` git-ignored). UI tests mock the file dialogs and
  fail fast on any unexpected modal (an unmocked modal hangs headless).
- **Detection:** the current detect flow is the one Marc blessed
  ("perfect") — detected corners + `refine_margins`. `align_grid` exists
  in refine.py but is UNUSED; earlier attempts to change detection "made
  things worse" and were reverted. Don't touch detection without cause.
- **Commit trailers used this project:**
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and a
  `Claude-Session:` line. (Do not put the model identifier in commits.)

---

## 8. Where we left off (2026-07-20, second session)

**Reuleaux Broad / Fine split just landed** (95 tests green): the
validated fixed-6 port is now the "Reuleaux Broad" stage (math
untouched, renamed only), and the new "Reuleaux Fine" stage is one
freely placed 360° hue zone — smooth plateau hue window plus **sat
mask and luma mask** (`app/core/windows.py`), gating DCTL-style
hue/sat/val moves, neutral-axis protected, chainable for several
zones. New preset "Reuleaux Broad + Fine". This was Marc's explicit
design: keep broad as-is, add fine — NOT masks bolted onto the fixed-6.

**Fine now has its companion DCTL** (`dctl/ReuleauxFine.dctl`, 12
sliders, hue values in degrees — the solver's stage report prints in
exactly these units, paste 1:1; chain several nodes for several
zones). `tools/reuleaux_fine_bake.py` bakes the same slider values
into a .cube for the Resolve A/B — same validation flow that proved
Broad. NOT yet pixel-verified in Resolve (needs Marc). The DCTL embeds
the unlicensed reuleaux conversions: private use only, never
redistribute.

**Awaiting Marc:** real-footage validation of the full parametric
pipeline (incl. Broad + Fine); report which stage earns its keep from
the waterfall; try pasting the fitted Reuleaux sliders into the DCTL
in Resolve.

**Open threads:** an earlier "also, none of thre…" message was never
completed; the curves-in-DRX experiment is designed but not run.
