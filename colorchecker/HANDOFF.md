# Color Checker — Project Handoff

Read this first in a new session. It is the map: what the tool is, what
is built, the non-negotiable decisions, the resources, and where we
left off. Deep design lives in `ROADMAP.md`; this is the orientation
layer. (Refreshed 2026-07-21 end-of-session; supersedes prior handoff.)

---

## 0. Latest session (2026-07-22)

**Housekeeping (later same day): test suite fast/slow split.** The
suite had grown to ~5 min; ~95% was 21 end-to-end solver tests.
`pytest` is now the ~10 s dev loop (slow tests skipped, visibly);
`pytest --full -n auto` is the pre-push gate (~3.5 min). No test
content changed — only markers + conftest + pytest.ini +
requirements-dev.txt. Details in section 7.

Curve + split-tone overhaul, driven by Marc grading in Resolve:

- **Contrast Curve exposure** is now achromatic AND applied AFTER the
  curve, mid-grey referenced: dialling ±N stops moves mid-grey by exactly
  ±N stops, preserves hue/chroma (~1e-13 drift). It no longer recolours.
  (`_expose` in chromogen.py; mirrored in torch + DCTL.)
- **Contrast Curve toe/shoulder rebuilt on Contour's model**: Black/White
  Point (asymptote LEVELS, capped to the visible range — black -6..-1.5
  stops ≈ code 0..0.28, baseline ~code 0.08 at -4.2), Toe/Shoulder Length
  (roll EXTENT: 0=knee at point/identity, 1=knee at pivot), Toe/Shoulder
  Strength (knee sharpness n 1..9), Preserve Color (= old Luma Blend),
  Mid Push + Mid Compensate (Compensate ON = the mid-S). Solver `init()`
  hook seeds the toe/shoulder engaged (identity is a dead-gradient
  region). All exact-identity + numpy/torch parity.
- **Split Tone stage NEW** (`SplitToneStage`, chromogen.py + torch mirror
  + dctl/SplitTone.dctl): per-channel cubic-Bezier shadow/highlight
  (Black/Shadow/Highlight/White ×RGB), ported from Marc's Bezier split
  tone. RGB/log per-channel (the subtractive split we agreed on). Added
  **per-channel Crossover offset**: 0 pins the channel to the pivot
  (neutral mid, as the stock tool forced all 3), non-zero floats its
  crossover LEVEL so channels need NOT converge at one point (mid tint).
  Uses exact 1/3 framing (stock DCTL's 0.333 left ~1e-4 non-identity).
- **Neutral Tint is OUT of the ML search pool** (Marc): Split Tone
  replaces it for fitting. It stays in STAGE_POOL (presets / manual /
  .drx mapping). See `chain_search.default_pool()`. local_search's
  tone-unfreeze now triggers on Split Tone OR Neutral Tint.
- **DCTLs renamed to theme**: Bezier_Split_Tone_V2 → SplitTone.dctl,
  ME_Filmic_Contrast_v1.3 → FilmicContrast.dctl. Both default to LogC3.
- **New kitchen-sink template** `1.5.3.T.drx` (scratchpad) has
  FilmicContrast + SplitTone + sectors + OpenDRT — but NO ContrastCurve /
  ColourSaturation. NOT yet wired as the export default because the fit
  still uses ContrastCurve (see next).

**2026-07-22 second block — FilmicContrast port + full ML->PowerGrade
pipeline (Marc: "get the whole ML + powergrade pipeline working"):**

- **ME_Filmic_Contrast V1.3 PORTED** (`app/core/filmic.py`, 1:1 float64
  like reuleaux/openDRT; torch mirror; `dctl/FilmicContrast.dctl`
  committed — Marc must install THIS copy). LogC3 / Preserve Mid-gray
  ON / end-exposure config. Its Exposure is a linear gain = mid-grey
  balanced + achromatic, exactly the required tone-node behaviour.
  Documented deviations (all exact mathematical limits): WP raw 1.02 /
  BP raw 0.0 = section exactly OFF (stage identity), 0-valued
  exposure/pop/flare skip their log<->lin round-trips, shoulder base
  floored 1e-9 (the stock tool can NaN there), and _mix_sat at
  contrast==1 skips the CHEN round-trip (its truncated rotation
  constants carry ~4e-9). **Black Point range extended to 1.5** (Marc:
  "extend the range to 1, maybe 1.5"; stock 0.5) — required lowering
  the stock sanitize floor 0.69 -> 0.4 which silently dead-zoned the
  slider past ~0.775; mirrored in the committed DCTL.
- **ML pool swap**: Filmic Contrast IS the tone tool now — it takes the
  grey-locked-tone freeze slot and Contrast Curve left `default_pool()`
  (still in STAGE_POOL for presets/manual, like Neutral Tint).
  `_fit_candidate` now starts stages at `init()` (Filmic seeds its
  white/black points engaged; identity is dead-gradient).
- **PowerGrade export: NODE SYNTHESIS** (`drx_build.SYNTH_SPECS`). Node
  types missing from the template are now synthesized: clone any DCTL
  node, rewrite its protobuf param map (path + sliderFloatParamN doubles
  + checkBox/comboBox entries, name-sorted like Resolve writes them).
  Proven: Resolve honors entries BY NAME (Marc's old templates carry
  slider indices 10/11). FilmicContrast (14 sliders) + SplitTone (16)
  export with EVERY slider stored — no "wiggle it in Resolve" gaps, no
  template dependency. Round-trip verified in tests; **Marc still needs
  one Resolve import to bless a synthesized grade on his machine.**
- Full gate green: 177 tests (14 new filmic + synthesis coverage).

**2026-07-22 evening — Filmic tuning round (Marc grading on footage;
ALL of these are toolkit deviations from stock, mirrored in port +
torch + dctl/FilmicContrast.dctl — reinstall the DCTL again):**

- **NaN guards shipped**: Black Point 0.0 / White Point 1.02 blacked
  the whole image (stock formula divides by zero at the section-off
  limit; guards return input). Roll ratio floored at 1e-3 (float32-safe)
  — WP near top + high Shoulder took pow(negative) = NaN in stock.
- **White Point extended DOWN** (slider min -0.15; the 0.5 and 0.7
  sanitize floors relaxed to pivot*1.1): stock dead-zoned the slider
  below ~0.42; the white ceiling now fades to just above mid-grey.
- **Shoulder** slider step 0.1 -> 0.001 (stock had 8 coarse detents —
  the "shoulder isn't doing anything" feel; it also only shapes when
  WP is engaged). **Shoulder Falloff** max 9 -> 9.7 (softness floor).
- **Toe Falloff remapped** in preserve-midgray: stock squashed the
  whole slider into internal strength 1.48..2.95; now 0.25..3.35 with
  the default (falloff 2 -> 2.65) EXACTLY unchanged (Marc: toe works
  fine, wants more shape range).
- **Pop Mids is mid-grey COMPENSATED** (Marc: "compensate exposure to
  keep mid grey intact"): a global counter-gain re-anchors a mid-grey
  pixel exactly where the tone block put it, for any Pop Mids value —
  Pop Mids and Exposure are now fully DECOUPLED, so the ML needs no
  special ordering for the pair (Marc floated "tweak all other params
  first"; with the compensation that's unnecessary).
- Gate: 180 green.

**2026-07-22 late — Neutral Tint falloff fix + first 250D match:**

- **Neutral Tint floors raised** (Marc found the bug on footage: a
  near-step transition on the scope): Falloff floor 0.1 -> 1.0 stop,
  Pivot floor -6 -> -4 stops (below ~-4.2 is sub-black code) — DCTL
  sliders + stage bounds. **Neutral Tint is BACK in default_pool()**
  alongside Split Tone (Marc: "add it back in the ML") — 11 tools.
- **genesis250D_2383.cube matched** (`--drt-math --target-is-display
  --search --max-nodes 14`, cloud, ~35 min, run BEFORE the NT
  re-admission — the "pipeline as is" run Marc asked for): display
  error 0.172 -> 0.044 (p95 0.096, worst patch 0.234), 0 dropped.
  Chain: Filmic tone freeze (punch contrast, roll highs, lift blacks,
  -1.4 stop) + 5x Split Tone + Highlight Bleach + 4x Sector Brightness
  + 2x Sector Skew + Crosstalk. Delivered to Marc as .cube/.drx/
  chain.json (scratchpad files don't persist — chain.json is the
  refit-warm-start insurance). CAVEATS for the next iteration: still
  gaining ~1%/node at the max_nodes stop (try 18-20 or Marc's Mac),
  several bound-pinned params (Skew -60 x2, G->Y/B -1.0, Split Tone
  Blacks at -1), noise gain max x59.8 on a Split Tone (median x0.62;
  local_search prune was OFF — try --local-search next), and 5 "cool
  lows" Split Tones smell like one tool doing gradual descent — the
  re-admitted Neutral Tint may consolidate those in the next run.

**Open / next (in priority order):**

1. **Marc verifies in Resolve**: (a) install the committed
   dctl/FilmicContrast.dctl (extended BP range); (b) import one
   search-delivered .drx containing synthesized FilmicContrast +
   SplitTone nodes and confirm sliders arrive; (c) eyeball a real
   genesis search with the new tone node.
2. **Contrast curve: one extra FREE control point** (Marc, 2026-07-22),
   for model flexibility. Params: an **x coordinate** (where on the curve)
   and a **y ± offset** from where the curve currently is at that x —
   maybe a spline/tension param if needed, but PREFER leaving it out and
   finding a smooth way (fewer knobs for the same functionality is the
   explicit goal). A single movable point that bumps the curve locally
   and smoothly. Lands on FilmicContrast now (it won the tone slot).
   DEFERRED by Marc same day ("lets do this later").
3. Split Tone crossover disable-toggle question — DEFERRED by Marc
   ("lets do this later too, i want to see how it works atm").
4. ~~Kitchen-sink template needed~~ RESOLVED via node synthesis (any
   template with one DCTL node suffices; all_nodes stays the default).

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
    filmic.py                  ME_Filmic_Contrast port (THE tone tool) + FilmicContrastStage
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
  dctl/                        12 companion DCTLs (paste-parity with solver reports)
  tools/                       reuleaux_bake, reuleaux_fine_bake, stage_bake,
                               lut_match (CLI), drx_export
  templates/                   Marc's powergrades. DEFAULT (2026-07-21):
                               brilliance_red_1.4.1.T.drx — 12 look nodes incl.
                               BrillianceReduction, LGG, 2x NeutralTint, all 4
                               Sectors + 2x OpenDRT (which one is live?
                               unverified). Its BrillianceReduction node has NO
                               stored double for Pivot (slider 2) — Marc must
                               wiggle it once + re-save to make it patchable.
                               Older: contrast_boost_1.6.4.T.drx + 1.6.1/1.6.2
  reference/OpenDRT.dctl       openDRT source (Jed Smith, GPLv3) for the port
  tests/                       177 offscreen incl. torch (auto-skip w/o);
                               fast/slow split — see section 7 Tests
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
  ways), and the **Chromogen family** (now 10): Colour Saturation
  (R/G+Y/B opponent axes; Y/B is a native reuleaux axis), Colour
  Crosstalk (inherent luminance-weighted tilt; the inherent val
  weighting FADES OUT as |Zone| rises so full-zone throw still has an
  effect — Marc 2026-07-21), **Contrast Curve v2** (NEW 2026-07-21,
  replaces the old soft-S "Contrast Boost": a real toe+shoulder film S
  modelled on Diachromie's [1D] CONTRAST panel — Contrast (linear
  slope), White/Black Offset (highlight/shadow slope, the smooth
  shoulder/toe), Mid Push + Mid Compensate (midtone bump that lifts the
  pivot, or holds it and adds local S), Shoulder/Toe Rolloff (knee
  sharpness, -1≈linear, inert while its offset=1), Luma Blend
  (0=per-RGB/raises sat, 1=luma only), Blend, plus pre-curve Flare
  (milky shadows) and Exposure; all in LogC3 stops, pivot fixed at
  mid-grey, identity=do-nothing; the DCTL has a Draw Curve checkbox that
  scopes the live curve on screen), Highlight Bleach (RYGB
  sectors x highlight ramp), **Neutral Tint v3** (Baselight-style,
  2026-07-21: SUM-PRESERVING offset in LOG RGB — not a reuleaux chroma
  push; signed amount 0-centred, right=highs/left=lows; Chroma slider
  is Baselight's 0..2 sat mask, 1=all/0=saturated/2=neutrals;
  TINT_SCALE=0.15), **Brilliance Reduction** (NEW 2026-07-21, the last
  missing Chromogen tool: luminance scale weighted by a SAT-domain
  ramp; Amount 1.0=identity at right end, pull DOWN to reduce;
  Chroma/Pivot/Falloff all in sat units, defaults 0.6/0.35/0.5;
  CORRECTED same day: identity is Amount 0 at the LEFT end, raise to
  reduce — the first screenshot's Amount 1.0 was a graded value, and
  the identity-at-1 first version read as a dead panel),
  Sector Skew/Brightness/Saturation/Squash (single picked hue; squash
  signed, foldover-proof; sector SATURATION IS LINEAR — the power law
  amplified noise).
- **Modulation block everywhere:** Zone (signed, middle=all), Pivot
  (IN STOPS from mid-grey; LogC3 calibration MID_GREY=0.391,
  STOP=0.0741), Chroma (signed: right=saturated, left=neutrals).
  Falloff (stops) only where Chromogen exposes it. EXCEPTIONS (copied
  from Baselight's own panels): Neutral Tint's Chroma is 0..2 with
  1=everything; Brilliance Reduction's Chroma/Pivot/Falloff are in the
  sat domain.
- **Panel calibration source:** `reference/chromogen_panels.md` (repo
  root) — full transcription of Marc's Baselight panel screenshots
  (ALL 10 tools covered incl. the four Sector panels) with per-slider
  defaults/ranges/bar graphics, the pivot=stops evidence, Marc's
  answers (Baselight hue 0 = yellow, ours stays red-at-0; Extended
  Ranges ~doubles effect; keep our Chroma sign), and the remaining
  open questions (tooltip texts "later", falloff units). 2026-07-21
  recalibration from it: Colour Saturation R/G+Y/B range 0..2
  (identity centred), Contrast Boost floor 0.0 (no negative), Bleach
  falloff default 0.5 stops, Tint falloff default 1.0 stops (falloff
  lower bounds now 0.1). Baselight panel default pivots (Bleach -2.00,
  Tint -0.70~mid-grey) CONFIRM our stops convention.
- **Solvers:** RBF (unchanged) and Parametric — stagewise coordinate
  descent (prep stages last, per-stage identity reg) -> optional
  **backprop** (torch optional dep; Adam over sigmoid-bounded params;
  multi-restart hue placement for Fine zones) -> scipy joint refine.
  Output: error waterfall + **noise-gain KPI** per stage/chain +
  labels + paste-ready reports. solve_parametric now takes
  `init_params` (warm start).
- **FREE-ORDER CHAIN SEARCH** (Marc's 2026-07-21 pipeline rework,
  `app/core/chain_search.py`): no preset, no LGG, no prescribed
  order — greedy forward construction auditions every Chromogen tool
  each round (hue multi-seeded), appends the winner, jointly refines
  the whole chain, stops at --max-nodes or when the best candidate
  gains < min_gain (0.5% default); final polish+report via
  solve_parametric warm start. Validated: recovers a hidden 4-tool
  chain exactly, in order, error 0.179 -> 0.002 after 4 nodes; ~40s
  for 8 nodes / 1500 samples / scipy on the cloud box.
  - **broad_bias** (default 0.15, CLI --broad-bias): single-hue tools
    (Sector family, Fine — `Stage.local_tool=True`) get their audition
    gain discounted so BROAD tools win ties (Marc: "so much sector
    stuff"). Acceptance/min_gain always use the real undiscounted gain,
    and a stalled biased winner falls back to the raw best so the bias
    can never prematurely stop the search.
  - **neutral_tone / grey-locked tone** (default ON, CLI --free-tone
    to disable; Marc: "contrast adjusted based on grey scale only"):
    before the free search, ONE Contrast Boost is fitted on the
    NEUTRAL samples only and FROZEN as node 1; Contrast Boost then
    leaves the audition pool. Every other tool is neutral-safe by
    construction, so the grey match is exact and can't be disturbed.
    Implemented via solve_parametric's new `frozen=N` arg (first N
    stages applied but never optimized).
  - **crash insurance**: search mode writes a `<out>.chain.json`
    (stages + params) as soon as the search finishes, and the drx
    template is now opened BEFORE the solve (fail-fast on missing
    zstandard) — a failed export can never cost the search again.
- **Chain presets** incl. "Chromogen match (LGG prep -> Chromogen
  chain)" and "Chromogen film look (full stack)" (Marc's canonical
  order: sectors BEFORE Highlight Bleach; duplicates allowed).

### DCTLs — 14 files, sliders EXACTLY = solver report units
FilmicContrast.dctl is the MODIFIED toolkit copy (BP range 1.5 / floor
0.4, WP max 1.02) — Marc must install it over the stock ME version.
ReuleauxFine + 10 Chromogen tools + LiftGammaGain (ContrastBoost.dctl
renamed → ContrastCurve.dctl: 10 float sliders + TWO checkboxes — Mid
Push Compensate (right after Mid Push, like Diachromie) and Draw Curve
(scopes the live curve on screen). Because a checkbox is NOT a
sliderFloatParam, "Mid Compensate" is the LAST entry in the stage's
param_names so the 10 float sliders stay contiguous (sliderFloatParam
0..9) for the .drx patch; its fitted 0/1 value is exported as a template
gap, not auto-patched. DCTL overlay math is per-component only — this
compiler rejects float3+scalar). NOTE: the
NeutralTint.dctl sliders CHANGED 2026-07-21 (Chroma now 0..2 default
1; log-RGB math) — Marc must reinstall it, and NeutralTint nodes saved
in the powergrade templates carry old-convention values. Resolve quirk:
transform() signature must be ONE LINE. Chromogen-family DCTLs carry
the reuleaux no-license warning (private use only). Marc: "it fucking
works". NOT yet formally A/B-verified vs Python (tools/stage_bake
bakes any stage by slider name for that).

### LUT matching (Plan C) — WORKS
`python3 -m tools.lut_match --lut look.cube [--backend torch]
[--drt drt.cube] [--target-is-display] [--out fitted.cube]
[--drx-out fitted.drx] [--source-csv patches.csv]`
- **--search --max-nodes N [--min-gain 0.005] [--broad-bias 0.15]
  [--free-tone] --deliver**: the free-order search mode (see above);
  --deliver drops the fitted .cube AND .drx into ~/Downloads (for
  local runs on the Mac). Search+drx: the .drx can only run the
  template's node order, so if the discovered order differs the CLI
  REFITS the found stage set in template order (warm start) and
  reports both errors; unused template look-nodes are reset to
  identity so the exported grade is exactly the fitted chain.
- **--drt-math** = the ANALYTIC openDRT (app/core/opendrt.py, Marc's
  exact config) instead of a baked --drt cube: display-domain loss,
  NO inversion, NO unreachable-dropping. This is what surfaced
  the tone tool in the genesis match (the cube sandwich had been
  deleting the tone evidence). scipy backend only (no torch mirror
  yet). Recommended genesis command:
  `python3 -m tools.lut_match --lut test_luts/genesis_e100_base.cube
  --drt-math --target-is-display --search --max-nodes 20 --deliver`
- --drt = display-referred sandwich (fit in log under the DRT, errors
  through it, unreachable targets dropped).
- --target-is-display = the look LUT already renders to display
  (genesis!): solve DRT(chain(x)) ~= lut(x).
- Real-world result (genesis e100 under openDRT): OLD cube sandwich
  0.224 -> 0.109 mean, 484/1395 unreachable. NEW analytic --drt-math
  + free search (20 nodes, Marc's local run): 0 dropped, display error
  0.199 -> 0.060, and Contrast Boost is now picked (node 2, +0.837).
  Still bound-pinned sector params + a noise-gain spike (max ×457) =
  next-session cleanup (noise-gain-aware search).

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

---

## 5. openDRT port: DONE (2026-07-21) — and it fixed genesis

`app/core/opendrt.py`: 1:1 float64 port of Marc's EXACT installed
OpenDRT v1.1.0b50 (`reference/OpenDRT_installed.dctl`, GPLv3), preset
tables resolved for his confirmed config (AWG3 / LogC3 / Standard /
Low Contrast / sRGB 2.2, cwp D65, Lp 100). **VALIDATION GATE PASSED
first try**: mean abs err 7e-6 / max 6e-5 vs the Resolve-baked 65^3
cube (`test_luts/`, committed) — float32 quantization territory.

**Display-domain loss shipped**: `display_transform=` in
solve_parametric / search_chain / lut_match (`--drt-math` CLI flag) —
residual = openDRT(chain(x)) - display_target, computed analytically.
No cube inversion, no unreachable-dropping. That dropping was the bug
that hid Contrast Boost from the genesis search (the contrasty
shadow/highlight pairs were exactly the ones deleted; a synthetic
sandwich with few drops picked Contrast Boost immediately).

**Genesis verification (cloud, 12 nodes, 1200 samples, scipy):**
Contrast Boost appears as node 2 at +0.809; display error 0.064
(old cube-sandwich baseline: 0.109). Caveats for next session:
chain noise gain max x4188 (a squash/skew pinned at bounds — consider
a noise-gain penalty or cap in the search), several bound-pinned
sector params, worst patch 0.335.

### Next steps (new priority order)
1. **CONTRAST CURVE v2 — DONE (Marc, 2026-07-21).** Replaced the old
   soft-S "Contrast Boost" with `ContrastCurveStage`, a full film S
   modelled behaviorally on Diachromie's [1D] CONTRAST panel (Marc's
   six curve screenshots pinned every slider). Sliders: Contrast
   (linear log slope), White/Black Offset (highlight/shadow slope = the
   smooth shoulder/toe; 1=neutral), Mid Push + Mid Compensate (midtone
   bump: off lifts the pivot, on holds it and adds local S),
   Shoulder/Toe Rolloff (knee sharpness; -1≈linear, inert while the
   matching offset=1), Luma Blend (0=per-RGB raises sat, 1=luma-only
   chroma-preserving), Blend, Flare (milky shadows), Exposure (stops).
   Identity = do-nothing; pivot fixed at mid-grey; validated to machine
   epsilon against all six screenshots (slopes, flare, mid-push pivot
   hold). numpy + torch mirror (parity 1e-9) + ContrastCurve.dctl
   (with a Draw Curve on-screen scope, à la Film_Curve_1.dctl) + the
   neutral-tone freeze path all updated together; presets renamed.
   NOTE the deliberate overlap: per-RGB contrast also moves saturation,
   so in a pool WITH neutrals the neutral_tone freeze pulls one Contrast
   Curve out and removes it from the audition (no double-dip); the two
   synthetic search tests that build sat looks now exclude it / cap the
   contrast to stay unambiguous. STILL TO EYEBALL ON FOOTAGE (Marc): the
   guessed shape constants — rolloff knee width/range (_K0=1.2,
   _K_RANGE=6), mid-push width/scale (_MID_W=2, _MID_SCALE=1), flare
   scale/width (0.045/3 stops) — and whether Contrast should cap at 2.0.
2. Torch mirror of the openDRT port -> backprop with display loss
   (currently --drt-math is scipy-only; --backend torch raises early).
3. Noise-gain-aware search (penalize auditions that amplify noise) —
   the genesis run still pins some sector params at bounds + spikes
   noise gain.
4. Matching-tab UI hookup for the chain search + drt-math + the new
   toggles (broad-bias, grey-locked tone).

## 5b. (historical) the original port plan

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

- **Branch:** `claude/color-tools-crosstalk-tint-d6pbl7` (current;
  crosstalk-zone fix + Neutral Tint v3 + Brilliance Reduction). The
  Chromogen/backprop/drx era lived on `claude/color-checker-handoff-
  91whob`, merged via PRs #3-#5.
- **Run on Mac:** `python3 main.py` from `colorchecker/`; deps
  `python3 -m pip install -r requirements.txt` (NOT pip3 — path has
  spaces). torch is OPTIONAL (backprop); zstandard required (drx).
- **Tests (fast/slow split, 2026-07-22):**
  `QT_QPA_PLATFORM=offscreen python3 -m pytest` — the DEV LOOP, ~10 s:
  skips the 21 tests marked `slow` (end-to-end solver/search runs; the
  skip count is shown, never silent). The GATE before every commit/push:
  `python3 -m pytest --full -n auto` (~3.5 min on the 4-core cloud box,
  vs ~5 min serial; needs pytest-xdist — `pip install -r
  requirements-dev.txt`). conftest.py pins BLAS to 1 thread per xdist
  worker (without it, parallel = serial from core thrashing). 155 green
  + 3 torch-skips on the cloud box (torch optional). Slow-marking
  policy: any test ≥2 s (real scipy/Adam optimization) gets
  `@pytest.mark.slow`; keep at least an import-level smoke test of each
  module in the fast set. Cloud container may need `apt-get update &&
  apt-get install libegl1 libgl1 libxkbcommon0` for Qt.
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
4. ~~Sector Saturation linear range 0-2~~ RESOLVED 2026-07-21: the
   Baselight panel shows 1.00 dead-centre of 0..2 — our range is right.
5. Tone pre-curve option for LUT matching (bridging different
   renderings — roadmap).
6. drx node LABEL patching (variable-length strings -> needs generic
   protobuf re-serialize; short_label() names exist already).
7. Transfer-function dropdown for stops calibration (LogC3-only now).
8. Curves-in-DRX experiment (old thread, still unrun).
9. Brilliance Reduction is still absent from the "Chromogen film look
   (full stack)" preset (search mode makes order moot, but preset
   mode users should know). The NEW template has its node, but with
   Pivot unpatchable (see templates/ note) and saved with old-DCTL
   values (identity reset on export handles that). Matching-tab UI
   does not expose the chain search yet — CLI only.
10. Marc to eyeball the 2026-07-21 changes on footage: crosstalk
    shadow-zone strength, Neutral Tint v3 feel (TINT_SCALE=0.15 max
    throw), Brilliance Reduction slider ranges (all guessed 0..1 from
    the Baselight screenshot knob positions).
