# Color Checker — Project Handoff

Read this first in a new session. It is the map: what the tool is, what
is built, the non-negotiable decisions, the resources, and where we
left off. Deep design lives in `ROADMAP.md`; this is the orientation
layer. (Refreshed 2026-07-20 end-of-session; supersedes prior handoff.)

---

## 1. What this is

A macOS (Apple Silicon), Python/PySide6 clone-and-extension of Nico
Fink's (Demystify-Color) chart-readout tool, grown into a **film-look
engineering toolkit**: read chart patches from footage, fit
interpretable parametric look chains (Chromogen-style tools, Reuleaux),
match LUTs, and export the result as DCTL slider values, .cube files,
and ready-to-import Resolve PowerGrades (.drx).

**Marc's pipeline:** Alexa Mini LF, LogC3 (AWG3 / EI800), float/16-bit
TIFF; ColorChecker Digital SG; EV/Kelvin/Hue sweeps; sometimes an
emissive Aputure panel in frame. Monitoring through openDRT.

**Dev model:** development happens in the cloud (Linux, headless Qt via
`QT_QPA_PLATFORM=offscreen`). Marc pulls the branch and runs on his
Mac. Everything must be testable offscreen; core stays UI-free.

---

## 2. THE hard invariant (never violate)

**Zero color management in the measurement path. Input = output.**
No ICC, no gamma, no clamp, no normalization anywhere between the TIFF
pixels and the CSV. Emissive values >1.0 pass through raw. Marc:
"do NOT change any of the displayed RGB values, that is the whole
point!!!!". The on-screen preview (`app/core/preview.py`) is the ONLY
place pixels are transformed — display-only, one-way. Bit-exact TIFF
decode proved in `test_image_io.py`.

---

## 3. Architecture

```
colorchecker/
  main.py                      launch
  app/core/                    UI-FREE logic (all headless-testable)
    image_io.py                bit-exact TIFF load; EV/Kelvin/Hue filename parsing
    project.py                 ProjectStore + ImageEntry; schema_version=1
    overlay.py                 Overlay model (uid-keyed!), incl. emissive Light Source
    detect.py / refine.py      chart auto-detect + margin refinement (Marc-blessed, don't touch)
    homography.py / sampler.py DLT unit-square->quad; pixel-center sampling
    csv_export.py              deterministic export
    preview.py                 display-only transform (NOT measurement)
    lut.py                     CubeLUT parse/apply (trilinear), gradients, lattice
    match.py                   HierarchicalRBF solve_match; invert_lut_at (DRT sandwich);
                               write_cube; load_patch_csv
    reuleaux.py                1:1 reuleaux port (validated in Resolve)
    stage_base.py              Stage ABC + reg_scale + label()/short_label()
    stages.py                  Matrix/LumaCurve/RGBCurves/ReuleauxBroad/ReuleauxFine/
                               LiftGammaGain + STAGE_POOL + CHAIN_PRESETS
    chromogen.py               the 9 Chromogen-style stages + modulation block + hue_word
    windows.py                 plateau/wrapped windows + signed ramp_window
    parametric.py              solve_parametric: stagewise init (prep stages LAST) ->
                               optional torch refine -> joint least-squares; waterfall,
                               noise-gain KPI, labels, reports
    torch_stages.py            differentiable mirrors of ALL stages (parity 1e-9)
    backprop.py                Adam + sigmoid box bounds + Fine-zone restarts
    diagnostics.py             noise_gain artifact KPI
    lut_match.py               fit chain to a LUT; --drt sandwich; sample_lut_domain
    drx.py                     .drx parse/patch: zstd bodies, DCTL nodes, slider doubles
                               (fixed-width patch), combo/checkbox READ
  app/ui/                      PySide6; three tabs (Processing/Matching/LUT Inspector)
  dctl/                        11 companion DCTLs (paste-parity with solver reports)
  tools/                       reuleaux_bake, reuleaux_fine_bake, stage_bake,
                               lut_match (CLI), drx_export
  templates/                   Marc's powergrades: example_powergrade_1.6.1.T.drx
                               (8 Chromogen DCTL nodes + genesis cube node),
                               openDRT_powergrade_1.6.2.T.drx (same + openDRT node),
                               contrast_boost_1.6.4.T.drx (adds ContrastBoost),
                               liftgammagain_1.2.1.T.drx (FULL STACK: all 9
                               Chromogen tools + LiftGammaGain + openDRT +
                               MONO cube + 2 DMC nodes — the DEFAULT
                               --drx-template; default chain maps with zero
                               unmatched stages)
  reference/OpenDRT.dctl       openDRT source (Jed Smith, GPLv3) for the port
  tests/                       135 green offscreen (torch tests auto-skip w/o torch)
```

Tests drive REAL interaction paths (canvas signals, mocked dialogs).
STAGE_POOL-looping tests give every registered stage identity +
torch-parity coverage automatically.

---

## 4. What is built

### Processing tab — DONE (unchanged this session)
TIFF load -> auto-detect -> refine -> sample; multi-exposure sessions,
batch import/Process All, uid-keyed per-frame overlay overrides,
emissive overlays, CSV export, project save/load.

### Stage system + solvers — DONE
- **Stage contract:** flat param vector, box bounds, identity anchor,
  pure vectorized apply, `param_names` (DCTL slider order),
  `label(params)` (grading note: "skew dark greens toward cyan",
  "cool lows", "(idle)"), `short_label` (<=9 chars for Resolve node
  labels: "SkwDkGrn", "CoolLo"), `reg_scale` (identity-anchoring
  multiplier; prep stages high).
- **Stages:** Matrix, Luma Curve, RGB Curves (monotone), Reuleaux
  Broad (validated fixed-6 port, untouched), Reuleaux Fine (free
  360-degree zone + sat/luma masks), Lift Gamma Gain (prep: master
  lift+gamma, per-channel gain; reg_scale=25, fitted LAST in stagewise
  init — only moves if it makes the fit a LOT easier; verified both
  ways), and the **Chromogen family**: Colour Saturation (R/G+Y/B
  opponent axes; Y/B is a native reuleaux axis), Colour Crosstalk
  (inherent luminance-weighted tilt), Contrast Boost (grey/highlight
  pivots + chroma mix 0=val-only..1=per-RGB), Highlight Bleach (RYGB
  sectors x highlight ramp), Neutral Tint (signed amount +-highs/lows,
  val-preserving, x0.25 internal scale), Sector Skew/Brightness/
  Saturation/Squash (single picked hue; squash signed, foldover-proof;
  sector SATURATION IS LINEAR — the power law amplified noise).
- **Modulation block everywhere:** Zone (signed, middle=all), Pivot
  (IN STOPS from mid-grey; LogC3 calibration MID_GREY=0.391,
  STOP=0.0741), Chroma (signed: right=saturated, left=neutrals).
  Falloff (stops) only where Chromogen exposes it.
- **Solvers:** RBF (unchanged) and Parametric — stagewise coordinate
  descent (prep stages last, per-stage identity reg) -> optional
  **backprop** (torch optional dep; Adam over sigmoid-bounded params;
  multi-restart hue placement for Fine zones) -> scipy joint refine.
  Output: error waterfall + **noise-gain KPI** per stage/chain +
  labels + paste-ready reports.
- **Chain presets** incl. "Chromogen match (LGG prep -> Chromogen
  chain)" and "Chromogen film look (full stack)" (Marc's canonical
  order: sectors BEFORE Highlight Bleach; duplicates allowed).

### DCTLs — 11 files, sliders EXACTLY = solver report units
ReuleauxFine + 9 Chromogen tools + LiftGammaGain. Resolve quirk:
transform() signature must be ONE LINE. Chromogen-family DCTLs carry
the reuleaux no-license warning (private use only). Marc: "it fucking
works". NOT yet formally A/B-verified vs Python (tools/stage_bake
bakes any stage by slider name for that).

### LUT matching (Plan C) — WORKS
`python3 -m tools.lut_match --lut look.cube [--backend torch]
[--drt drt.cube] [--target-is-display] [--out fitted.cube]
[--drx-out fitted.drx] [--source-csv patches.csv]`
- --drt = display-referred sandwich (fit in log under the DRT, errors
  through it, unreachable targets dropped).
- --target-is-display = the look LUT already renders to display
  (genesis!): solve DRT(chain(x)) ~= lut(x).
- Real-world result (genesis e100 under openDRT): display error
  0.224 -> 0.109 mean; residual is mostly TONE (two different
  renderings); 484/1395 unreachable (cube inversion + gamut). See
  ROADMAP "first real run" section. Bound-pinned params = wrong
  composition smell.

### PowerGrade (.drx) — GENERATION WORKS, VERIFIED IN MARC'S RESOLVE
`app/core/drx.py`: XML wrapper -> prefix byte + zstd protobuf bodies;
DCTL nodes found by path; sliders = fixed-width doubles after
`sliderFloatParamN\x12\x09\x11` (patch never shifts bytes); combos/
checkboxes/int-sliders now READ (varints — do NOT patch, would shift).
`tools/drx_export.py --list / --set "Node:Slider=v" / --out`;
`lut_match --drx-out` patches a fitted chain straight into the
template (stage<->node by DCTL filename, k-th of a type; sliders by
param_names order — proven correct by Marc's import). Template gaps
are reported for manual pasting. sliderFloatParam6..11 on 6-slider
nodes are leftover pool junk — ignore.
- **2026-07-20: full-stack template landed.** Marc uploaded
  contrast_boost_1.6.4.T.drx and liftgammagain_1.2.1.T.drx; the
  latter (all 9 Chromogen tools incl. ContrastBoost + LiftGammaGain +
  openDRT) is now the DEFAULT --drx-template — the default
  Chromogen-match chain maps with zero unmatched stages (tested
  end-to-end).
- **Limits (answered for Marc 2026-07-20):** patching FILLS existing
  nodes only — it cannot add/duplicate/reorder nodes (variable-length
  protobuf; needs the generic re-serializer, see ROADMAP "TEMPLATE
  LIMITS"). Different orders/stacks = more templates in the library.
  Node byte order in the body is storage order, NOT graph order.

---

## 5. IMMEDIATE NEXT TASK: the openDRT port

Goal: replace the baked openDRT cube with exact math
(`app/core/opendrt.py` + torch mirror) -> exact/cheap inversion (fewer
dropped patches) + display-domain backprop loss. Full plan in ROADMAP
("openDRT analytic port"). Source: `reference/OpenDRT.dctl` (GPLv3 —
port module must carry the license; fine for private use).

**openDRT settings: CONFIRMED by Marc (screenshot, 2026-07-20).**
- Input Gamut **Arri Wide Gamut 3**, Input Transfer **Arri LogC3**,
  Look Preset **Standard**, Tonescale Preset **Low Contrast**,
  Creative White **USE LOOK PRESET**, Display Encoding **sRGB Display
  (2.2 power / Rec.709)**. All float sliders at defaults (Lp 100,
  grey boost 0.13, HDR purity 0.5, Lg 10, cwp limit 0.25), overlay off.
- VERSION MISMATCH ESTABLISHED: the node's stored combo indices
  [12, 8, 0, 2, 0, 2] put AWG3 at 12 and LogC3 at 8, but the uploaded
  reference/OpenDRT.dctl has them at 6 and 4 — Marc's INSTALLED
  OpenDRT.dctl (at ______DCTL______/DRTs/opendrt/OpenDRT.dctl) is a
  different, likely newer version than the uploaded file. **ASK MARC
  to upload that exact installed file** before transcribing; if only
  the v1.1.0 file is available, port it and check the validation gate
  against the baked cube first — if it passes within tolerance the
  version difference doesn't matter for this config.
- Also re-upload the baked openDRT_LogC3_srgb cube next session (the
  validation target; uploads don't persist).
- Port discipline = reuleaux port: 1:1 transcription, float64,
  vectorized, tests first, then the validation gate vs the baked cube,
  then wire as --drt-math in lut_match + Matching tab (analytic
  inversion replacing invert_lut_at).

---

## 6. Resources & licensing

- `reference/OpenDRT.dctl` — Jed Smith, **GPLv3** (port carries license).
- reuleaux (hotgluebanjo) — **NO LICENSE**: port + derived DCTLs are
  for Marc's private use only, never redistribute.
- Demystify-Color-DCTLs (M5era fork) — MIT (credit Nico Fink); the
  OkLab/LogC constants blueprint for Plan B (OkLab currently OFF the
  table per Marc — everything stays in reuleaux space).
- Marc's uploads do NOT persist between sessions. Currently in-repo:
  the two powergrade templates. NOT in repo (re-upload when needed):
  genesis_e100_base.cube, openDRT_LogC3_srgb cube, all_EV0.csv.
- Marc's Resolve DCTL folder is `0_MS` (paths inside the powergrades).

## 7. Dev workflow / gotchas

- **Branch:** `claude/color-checker-lut-matcher-z7a1qm` (the
  Chromogen/backprop/drx era branch was merged via PRs #3/#4;
  main carries everything through the 2026-07-20 handoff).
- **Run on Mac:** `python3 main.py` from `colorchecker/`; deps
  `python3 -m pip install -r requirements.txt` (NOT pip3 — path has
  spaces). torch is OPTIONAL (backprop); zstandard required (drx).
- **Tests:** `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/` —
  135 green, ~2-3 min. Cloud container may need
  `apt-get install libegl1 libgl1 libxkbcommon0` for Qt.
- **Detection code is Marc-blessed** — don't touch without cause.
- **Commit trailers:** Co-Authored-By + Claude-Session lines; never
  put the model identifier in commits.
- Marc communicates mid-turn; slider/design decisions come from HIS
  Chromogen screenshots — copy tool designs literally when in doubt.

## 8. Open threads

1. **openDRT port** (section 5) — the next session's task; blocked
   only on the settings question above.
2. Marc's real-footage validation: Fine-zone DCTL pixel A/B, the
   fitted genesis .drx on footage, stops-calibrated pivots feel.
3. Order-search option for chain order (roadmap, soft preference).
4. Sector Saturation linear range 0-2: Marc may want it tighter.
5. Tone pre-curve option for LUT matching (bridging different
   renderings — roadmap).
6. drx generic protobuf re-serializer: would unlock node LABEL
   patching (short_label() names exist already) AND node add/remove/
   reorder (Marc asked 2026-07-20; currently template-library only —
   see ROADMAP "TEMPLATE LIMITS").
7. Transfer-function dropdown for stops calibration (LogC3-only now).
8. Curves-in-DRX experiment (old thread, still unrun).
