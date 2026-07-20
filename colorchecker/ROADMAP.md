# Roadmap

## Matching: sandwich fit under a fixed DRT (designed 2026-07-19)

For display-referred targets (E100 reversal scans, print stocks) where
a direct log->display RBF is hazardous (shoulder extrapolation,
ringing on steep slopes, no monotonicity guarantee — see chat): let
the user supply a FIXED DRT .cube with roughly the right contrast;
fit the RBF underneath it.

Method: invert the DRT numerically AT THE TARGET PATCH VALUES only
(per-patch root-find through our cube interpolation — no global LUT
inversion needed), then solve the standard log-domain fit
log -> DRT^-1(target). Ship as RBF cube + DRT stacked in Resolve.

Handling: patches clipped by the target stock (D-max/D-min plateaus)
have undefined inverses -> drop + count, like NaN patches (they carry
no information about what's under the DRT). Ambiguous inverses at
gamut edges -> nearest least-squares + residual report. Error metric
reported THROUGH the DRT (display-referred, what the eye sees).

UI: Matching tab gains "Output transform (fixed): load .cube";
export = RBF cube alone (DRT stays its own node) or baked combo.

## Matching: Algorithm B — parametric zone model (refined 2026-07)

Status: planned, NOT started. Do after the current toolchain is proven on
real footage.

REFINED DESIGN (supersedes plain reuleaux port): custom hue-anchored
zone model + companion DCTL that we author ourselves.
- Space (decided 2026-07-18): OkLCh — cylindrical form of Oklab.
  Chosen for hue LINEARITY (broad adjustments don't bend blue->purple
  etc.) and perceptual smoothness, i.e. the Chromogen-like "doesn't
  break anything" character; Chromogen's own space is proprietary
  (Truelight/TCAM) and not copyable. Oklab math is two 3x3 matrices +
  a cube root — trivial to mirror bit-exact in DCTL. Alternatives
  considered: IPT (fine, clunkier constants), ICtCp (only if PQ/HDR
  pipelines), CIELAB LCh (rejected: blue-hue bend), naive HSV
  cylindrical (rejected: perceptually crooked, the breakage source).
  NOTE: Oklab expects ~linear input; bracket the zone math with a
  fixed documented log<->linear transform, identical in Python and
  DCTL, so parameter parity holds end to end.
  BLUEPRINT FOUND (2026-07-18): Demystify-Color/DCTLs (MIT, Nico Fink)
  ships OKLAB.dctl — AWG<->OKLAB with signed cbrt AND an inset/outset
  gamut guard for extreme blues; AWG matches Marc's Alexa footage.
  Plus open Linear<->LogC DCTLs = the exact log/linear bracket in Arri
  math. Plan: adopt these constants verbatim (with credit) in BOTH our
  zone DCTL and the Python fitter — shared published reference instead
  of two homegrown implementations; Python can be sanity-checked
  against Nico's node in Resolve before our DCTL exists. Also there:
  DMC_3x3Matrix v1-v3 (open, slider coefficients) — Stage 1 matrix
  export can target these directly, no own matrix DCTL needed.
  Encrypted (.dctle, black-box only): DMC_PiecewisePower (contrast),
  AWG3_To_CamNative.
  Fallback if ever needed: Ottosson's reference math is public
  (two 3x3 matrices + cbrt each way, ~15 DCTL lines) and we author
  both sides.

  Source verified 2026-07-18 from Marc's fork
  (M5era/Demystify-Color-DCTLs, cloned in-session):
  - OKLAB.dctl: inset -> AWG->XYZ->LMS -> signed cbrt -> OKLAB
    (+ inverse with outset). Expects LINEAR AWG input -> bracket is
    LogC->Linear, inset+oklab, zone math, inverse, Linear->LogC.
  - Log-C_To_Linear.dctl / inverse: Arri LogC3 EI800 constants
    (cut 0.1496582, a 5.555556, b 0.052272, c 0.2471896, d 0.385537,
    e 5.367655, f 0.092809).
  - LCHab.dctl is MISNAMED: it is LCh on OKLAB (OkLCh!) — AWG->Oklab->
    polar (L, C, h degrees, hue normalized 0-1). The full conversion
    shell for the zone model exists verbatim in the fork.
  - Zone math spec (2026-07-18): in OkLCh per zone k:
    w = cos^2(pi*d/120deg) for wrapped hue distance d<60deg (fixed
    6-anchor "reuleaux mode": weights partition unity, sliders never
    fight) OR Gaussian at fitted anchor (solver mode) — same code,
    swapped weight fn. Apply h+=sum(w*hueRot), C*=sum(w*sat),
    L+=sum(w*lum)*C (chroma-scaled: neutrals untouchable). Bounded
    hue rotation (~±30deg) keeps mapping monotonic/invertible.
  - Saturation stage decision (Marc, 2026-07-18): model it on the
    PRISMATIC COLOR SPACE — Hart, "The Prismatic Color Space for RGB
    Computations" (2015, PDF public; implemented in colour-science as
    colour.models.rgb.prismatic and in ColorAide). Barycentric Maxwell
    triangle chromaticity + rho=max(R,G,B) light/dark axis; saturating
    pins the max channel while purity rises -> subtractive film-like
    character, no garish high-sat+high-lum. This is (per its
    description) the model behind Nico's Advanced Natural Saturation
    DCTLE; we implement openly from the paper, not from his product.
    Division of labor: saturation in prismatic space, hue rotation +
    per-zone luminance in OkLCh zones. Nico's RGBCMY OKLAB Sat Shaper
    (6 fixed vectors in Oklab, Bogdanowicz LAB-sat rationale) equals
    our fixed-anchor mode — external validation of the design.
  - CAUTION DMC_3x3Matrix_v3 "Preserve Neutrals" mode is SEQUENTIAL,
    not a true matrix: red is overwritten first, green computed from
    the MODIFIED red, blue from both. A fitted true 3x3 cannot be
    pasted into those sliders naively — map to the sequential form or
    use the standard mode (verify rest of file first). Exactly the
    failure class the Resolve pixel-match gate exists for. Known porting risks + fixes: signed cbrt for
  negative scene values (pick convention once, mirror exactly);
  guard hue at chroma~0 so neutrals pass through; float32 GPU vs
  float64 numpy parity ~1e-6 (invisible; covered by the mandatory
  Resolve pixel-match gate).
  Reuleaux remains design reference only, not a dependency.
- Model (clarified 2026-07-18): up to ~20 SIMPLE NODES, each one zone.
  One tiny single-zone DCTL (~7 sliders: hue anchor + width, luminance
  center + width, delta hue/chroma/lightness, Gaussian falloff),
  instantiated as separate stacked Resolve nodes — bypass/tweak each in
  isolation. Solver grows the chain greedily: fit a zone, measure the
  residual, add the next node where error is largest, stop at target
  error or the node cap. Auto-name each node by its job.
- Solver: scipy least_squares over unit params against patch pairs
  (same data path as RBF).
- Auto-naming of fitted units: hue center -> color word, value center
  -> shadows/mids/highlights, dominant delta -> verb ("light green
  desat", "bleach highlights").
- Companion DCTL written by us with IDENTICAL parameters — parity by
  construction; fitted numbers paste straight into Resolve.
- Existing alternatives evaluated (thatcherfreeman/utility-dctls):
  Smooth Tetra (C1-smooth Yedlin tetra, fixed anchors), Hue Curve DCTL
  (free-anchor single-hue primitive — closest existing tool), Matrix
  Manipulator, Cylindrical/Spherical space conversions.

Original reference (kept for the zone-control scheme):

Reference: https://github.com/hotgluebanjo/reuleaux (HSV-like spherical
color model for film characterisation, Yedlin-inspired; implementations
in C / Nuke / DaFX / Resolve — the math to transcribe lives in that
repo's source).

Why: unlike the RBF (Algorithm A, arbitrary smooth warp), a reuleaux-
style model is PARAMETRIC and BOUNDED — a small set of interpretable
zone controls (hue/chroma/value adjustments about the neutral axis).
That means:
- the fitted parameters can be typed straight into the matching
  reuleaux/tetra-style DCTL inside Resolve (no LUT needed)
- parametric control after fitting: nudge individual zones by hand
- inherently less accurate than RBF (bounded model) — that's the
  trade, interpretability + Resolve portability vs raw fit quality

Feasibility: moderate. Forward/inverse transform is closed-form (small
port job from the C/DCTL source). Fitting = small nonlinear least
squares over the zone parameters (scipy.optimize.least_squares) against
the same patch pairs the RBF uses. Architecture already fits: MatchModel
is a stage chain; this slots in as an alternative stage.

## Target pipeline (Marc's sketch, order still open)

1. Linear 3x3 matrix (exposure invariant, gross channel mix)
2. EITHER reuleaux-style gamut adjustments (parametric, Resolve-
   portable) OR RBF (unbounded, best fit; hard to separate RGB curves
   out of it afterwards — if curves matter, fit them BEFORE the RBF)
3. Separate per-channel RGB curves (split toning)
4. Contrast curve — order unresolved: correcting it early gives the
   later stages cleaner input, but last-in-chain is nicer to have.
   Current lean: fit early, place late, re-fit once to confirm.

Plan B tool deliverables (each = Python fitter mirror + open DCTL
with identical parameters):
1. Single-zone DCTL (OkLCh): hue anchor + width, lum center + width,
   delta hue/chroma/lightness — stacked <=20x; fixed-6 "reuleaux
   mode" + free-anchor solver mode
2. Prismatic Saturation DCTL (Hart's prismatic space): global amount
   + optional per-hue-zone amounts; the open Advanced-Natural-Sat-
   character tool; fittable stage AND standalone hand tool
2b. OkLab Saturation DCTL (alternative sat model): chroma scaling in
   OkLCh — global amount + 6 fixed hue-vector sliders (RGBCMY, cos^2
   partition weights), the open equivalent of Nico's OKLAB Sat
   Shaper; perceptually uniform character vs prismatic's subtractive
   film character. Same fitter slot as 2 — solve with either, or
   compare both against the target and keep the lower-error one
3. Matrix stage: fitted 3x3 targeting Nico's DMC_3x3Matrix (mind the
   sequential quirk) or plain matrix DCTL
4. Curves stage: 1D .cube or curve control points

Per-stage export (the point of the whole design):
- Stage 1: the 9 matrix numbers (RGB Mixer / matrix DCTL)
- Stage 2a zones: the fitted PARAMETER SET, portable 1:1 into our
  zone DCTL in Resolve — fully parametric, no LUT
- Stage 2b RBF: 3D .cube (only stage that inherently needs a LUT)
- Stage 2c saturation: prismatic saturation parameters -> our DCTL
- Stages 3/4: 1D .cube or curve control points
Each stage individually toggleable, solvable, exportable, and hand-
adjustable without touching the others.

Parameter parity requirement (2a): our port must be bit-faithful to
the reuleaux DCTL — identical math, parameter names, ranges, defaults,
op order — so fitted numbers transfer verbatim. MANDATORY verification:
render a test frame through the real DCTL in Resolve with known
settings, run the same frame through our port, assert pixel match.

Notes:
- The more that lands in interpretable stages (1, 2a, 3, 4), the less
  any residual RBF has to explain — better extrapolation, smaller warp.

## PowerGrade (.drx) generation — feasibility PROVEN 2026-07-18

DRX format (verified on Marc's K64_1.0_1.5.2.drx): XML wrapper; grade
node graph in <Body> = 1 prefix byte + zstd-compressed protobuf.
DCTL effect nodes carry the .dctl path plus named params —
sliderFloatParamN as 8-byte LE doubles after b"<name>\x12\x09\x11",
checkboxes as varint bools, combos as indices. All readable.

Generation strategy: TEMPLATE PATCHING, not authoring from scratch.
Slider doubles are fixed-width -> patching changes no lengths, no
protobuf surgery. Proven round-trip: decompress -> patch value ->
zstd recompress -> splice hex -> reparse OK. VERIFIED IN RESOLVE
2026-07-18: patched K64 test file imported cleanly and showed the
changed slider value. The full app -> .drx pipeline is validated;
only the zone DCTL itself remains to be built.

Flow once Algorithm B exists: Marc saves a one-time template .drx
containing our zone DCTL node(s) with defaults; the app clones it,
writes fitted values into the sliders, outputs a ready PowerGrade.

NODE-GRAPH SURGERY — LANDED 2026-07-20 (app/core/drx_graph.py; was
"TEMPLATE LIMITS"). The grade protobuf is now fully decoded and
re-serializable (byte-identical round-trip gated on every template
body in tests — Resolve writes minimal varints, so a faithful
re-encode reproduces the input exactly):
- grade msg: .7 repeated NODE {1 id, 2 serial badge, 4/5 x/y,
  6 LABEL string, 8 kind (44 corrector / 90 layer mixer), 9 builtin
  params raw, 10 OFX/DCTL payload with the fixed-width sliders},
  .8 repeated EDGE {1 from, 3 to, [4 dest port], 7 link id},
  .9 ENTRY (repeated connection — Marc's full template feeds two
  branches into a layer mixer), .10 EXIT.
- Capabilities: duplicate_node, serial_rebuild (rewire the whole
  grade as one serial chain, drop mixers/orphans, renumber badges,
  grid-layout x/y), node LABEL patching (old open thread — labels are
  plain strings in the node record), slider patching inside the node's
  own OFX bytes.
- rebuild_as_chain materializes a fitted chain: assign nodes by DCTL
  type, DUPLICATE when the chain wants more instances than the
  template has, keep head prep + display tail (openDRT/3DCube) in
  place, reset unfitted stage nodes to identity (kept for
  hand-tweaking), drop layer mixers (Marc: generated grades are pure
  serial). lut_match --drx-out uses this; the full-stack preset (3x
  ColourSaturation, 2x NeutralTint) now materializes completely.
- The DMC/SLog3 mini-grade in body 0 of liftgammagain_1.2.1.T.drx is
  untouched — surgery targets only the body holding our stage nodes.
- VERIFICATION GATE OPEN: Resolve import of a generated
  (duplicated/reordered) .drx not yet confirmed by Marc — that is the
  remaining proof, same as the original slider-patch feasibility run.

OPEN QUESTION — native custom curves in DRX (would let Stages 3/4
ship as Resolve's own curve UI instead of 1D LUT/DCTL): Marc's K64
drx contains NO curve data (untouched tools are omitted), so encoding
unknown. Experiment defined: save curves_default.drx + one drx with a
single known curve point (0.25->0.40 master), diff decompressed blobs
to reveal the encoding. If readable: template must carry curves with
the SAME point count as the fitter outputs (variable-length protobuf
arrays — patch values only, fix point count, e.g. always 8 points).
Commercial DCTLs (PrimeGrade / MONONODES etc.): these are ENCRYPTED
.dctle — no source to read, so "map parameter semantics" is off the
table. Black-box paths instead:
- Tier 1 (easy, frozen): bake a node's transform at fixed settings to
  a LUT via Resolve's Generate 3D LUT / identity-lattice render.
- Tier 2 (experimental, keeps sliders live): automated black-box
  fitting loop — app patches candidate params into .drx (proven),
  Resolve scripting API applies grade + exports a probe still, app
  measures error vs target patches, optimizer iterates. Resolve
  executes the encrypted math; we only steer sliders. Needs Resolve
  Studio scripting. STATUS: backup plan only (Marc, 2026-07-18) —
  revisit if the primary path (own zone DCTL) proves insufficient.

  Mechanics (designed 2026-07-18):
  - One-time setup: solver project with a tiny synthetic lattice TIFF
    probe clip (all LUT lattice points encoded in one small frame) on
    a dedicated timeline. Probe imported ONCE, never re-imported.
  - Per iteration: ApplyGradeFromDRX(patched.drx) -> GrabStill() ->
    ExportStills(16-bit tif) -> read with our loader -> error ->
    optimizer step. ~1s/iteration; no render queue involved.
  - Batching: N probe clips on the timeline, N candidate .drx per
    pass. Plus surrogate modeling (sample slider responses with
    ~100-200 evals, optimize the surrogate natively in Python, verify
    + refine with a few real evals) -> expect 10-20 min per one-time
    solve per stock.
  - MANDATORY gate before any fitting: identity-grade round trip —
    exported probe must equal the input bit-for-bit (16-bit), proving
    the still-export path is color-management-clean.

## Chart-prep: datasheet alignment tool (idea, 2026-07-19)

Nico's Film Profile Journey #22 workflow, automated: align measured
grayscale patches to the stock's published D-logE sensitometric curves
(digitized from the whitepaper) instead of just balancing mid-grey —
corrects aged/faded stocks back to spec before profiling.

Works for NEGATIVE and POSITIVE/REVERSAL stocks alike: the method
lives in DENSITY (Cineon = 95 + 500*D), and reversal datasheets
publish the same D-logE curves. Differences for reversal: curve is
mirrored (D-min anchor at the high-exposure end), grayscale samples
the steeper curve more sparsely (mitigate with gray patches across
the EV sweep — data we already capture), and display-referred scans
must be converted to density first (or rescanned in density mode).

Tool shape: inputs = measured grayscale CSV (ours, across EVs) +
digitized datasheet curve points CSV; anchor at base (Cineon 95),
look up expected density per patch via known scene log exposures
(BabelColor Y -> logE), fit per-channel correction curves, export as
1D LUT/curve points. All measurement machinery exists in the app.

## Matching: Parametric solver (planned 2026-07-19, Marc-approved design)

Alternative to RBF in the Matching tab: chain of parametric stages,
each a PURE FUNCTION + FLAT PARAM VECTOR + BOUNDS (the ML-ready
contract: swap numpy->torch and finite-diff->autograd later without
touching the architecture; reuleaux's max/abs kinks are subgradient-OK).

Stages v1: LumaCurve (monotone-by-construction: positive-increment
parameterization over fixed x grid, ~6-8 params), RGBCurves (3x same,
identity-initialized: residual split-tone), Reuleaux (validated port,
20 params, DCTL slider bounds), LinearMatrix (9, pool only).
DEFAULT CHAIN: Luma -> RGB curves -> Reuleaux. User can add/remove/
reorder/disable stages (ordered list UI).

Solve: stagewise init (coordinate descent vs residual in chain order)
then joint least_squares over concatenated params (~60) with identity
regularization against stage overlap. Shares the match-type prep
(sandwich/DRT inversion, NaN/clip dropping, through-DRT errors) with
the RBF path — refactor solve_match prep to be solver-agnostic.

Outputs: per-stage error waterfall; exports per stage (matrix text,
curves 1D cube + points, REULEAUX FITTED SLIDER VALUES to paste into
ReuleauxUserStandalone.dctl — parity proven), plus combined 3D cube.

UI: Solver selector RBF | Parametric in the Model box; Parametric
panel = stage list w/ reorder + minimal per-stage settings.

Parametric solver addendum (Marc, 2026-07-19): subset solving is a
first-class requirement — chain presets in the UI: Full (Luma -> RGB
-> Reuleaux) / Reuleaux only / Matrix + Reuleaux / Custom. Rationale:
contrast/split may be hand-built in Resolve beforehand. Workflow note:
in that case the SOURCE patches must be measured through the manual
prep (re-render charts through the prep nodes), OR use the planned
"fixed input transform" option (load the prep as a LUT, applied to
source patches before fitting — mirror of the DRT sandwich on the
input side; cheap addition, not in v1 unless requested).

## Chromogen-style stage family (evaluated 2026-07-20, Marc-directed)

Goal: replicate the BEHAVIOR of FilmLight's Chromogen (Baselight
look-dev tool) with simple math in our reuleaux-space stage
architecture. Sources: Marc's screenshots + the EnergaCAMERIMAGE 2023
demo transcript (FilmLight). Chromogen internally = "Eab" opponent
space, scene-referred, pure formulas, no LUTs, every stage smooth —
"smooth onto smooth stays smooth; one non-smooth op breaks the look."
That is exactly our stage contract, so the whole family fits the
existing solver/backprop/DCTL pipeline. We stay in reuleaux space
(Marc's rule: OkLab off the table); this is behavior replication, not
a clone.

VERDICT: all tools replicable with simple math. Two shared primitives
to build first, then each tool is small.

### Shared primitive 1: signed ramp masks ("pivot for our luma masks")
windows.py gains a one-sided smooth ramp: 0 below pivot, 1 above,
cos^2 transition over falloff width, signable (negative = ramp toward
shadows). Chromogen's standard Modulation block on EVERY tool is:
  Zone (signed, 0 = everywhere; + = highlights, - = shadows)
  Pivot (where, relative to mid-gray) / Falloff (how smooth)
  Chroma (SIGNED sat gate: + = only already-saturated colors,
          - = only desaturated colors, 0 = uniform. The transcript's
          "add saturation only to colors that aren't saturated".)
Weight form: m = 1 - |zone| + |zone| * ramp(sign(zone)*(L - pivot),
falloff), same shape on the sat axis for chroma. Identity at 0 by
construction, smooth, differentiable, ~4 params. This block is
appended to every Chromogen-style stage below.

### Shared primitive 2: opponent-axis view of the reuleaux chroma plane
Lucky geometry: in reuleaux, Yellow (60 deg) and Blue (240 deg) are
exactly antipodal -> Y/B is a true axis. The orthogonal axis
(330/150 deg, magenta-red vs green-cyan) plays Chromogen's
"green-magenta" axis. Rotate chroma plane by -60 deg (+ optional user
Rotate param like Chromogen's), operate on the two components,
rotate back, reconstruct at chosen val.

### The stages (order = suggested build order)
1. COLOUR SATURATION (~7p): scale the two opponent axes independently
   (unganged R-G vs Y-B sliders, ganged by default) + Rotate + the
   modulation block. Reconstruct at constant val. Anisotropic scaling
   bends hues toward the stronger axis = the observed behavior.
   Typical uses from the demo: top-of-stack desat of only-saturated
   colors ("sand off the spikes"), then add sat to shadows/desaturated.
2. CONTRAST BOOST (~5p): analytic smooth contrast, grey pivot +
   highlight pivot (soft shoulder = HDR-safe rolloff, midtone
   steepening). Chroma slider 0..1 mixes two applications:
   0 = val-only at constant sat ("base grade", chromaticity untouched)
   1 = per-RGB-channel ("film grade", sat rises with contrast);
   default 0.5 ("splitting the difference is the best solution").
3. HIGHLIGHT BLEACH (~8p): 4 sector amounts (R/Y/G/B wrapped windows,
   ganged by default; demo: relax blues to save skies, relax yellows
   to save skin) x highlight ramp (pivot ~2 stops below mid-gray,
   smooth early kick-in) x chroma gate; sat -> down at constant val.
4. NEUTRAL TINT (~5p): target hue + SIGNED amount (+ = tint
   highlights, - = tint shadows), pivot/falloff, chroma gate
   protecting saturated colors from gamut overshoot. Add chroma
   vector toward target hue, reconstruct at unchanged val (tint
   without contrast change by construction). Two instances = the demo
   "Warm Highs" + "Cold Lows" (overlapping pivots recommended).
5. COLOUR CROSSTALK (~6p): the "twister". Per opponent direction, hue
   displacement along the orthogonal axis, with LUMINANCE AS THE
   WEIGHT (Marc + transcript: "stronger with the brighter signals"):
   a smooth monotone luma ramp (pivot/falloff) scales the
   displacement, so shadows barely move and highlights move fully =
   the tilt/shift of the whole space. A signed tilt option displaces
   shadows opposite to highlights — that full twist is where the
   blue-goes-red-in-shadows/green-in-highlights S shape comes from.
   4 amounts (R->Y/B, Y->R/G, G->Y/B, B->R/G), ganged by default,
   sat-weighted (stronger where saturated). Honest approximation of
   the most abstract tool; solver fits our version to targets.
6. SECTOR family — single PICKED hue (0-360 + falloff), NOT fixed-6
   (transcript + screenshots corrected the earlier fixed-6 guess).
   Each ~4p + modulation block; all are thin variants of the
   Reuleaux Fine machinery:
   - SECTOR SKEW: delta-hue only (Fine minus sat/val)
   - SECTOR BRIGHTNESS: val only
   - SECTOR SATURATION: sat only
   - SECTOR SQUASH: the HueSquash design (h' = T + delta*(1 - s*w)),
     foldover-proof normalization; strength is SIGNED — negative =
     SPREAD hues apart (demo: amplify makeup nuances). Squash toward
     picked hue; chroma gate excludes near-neutrals (reuleaux hue
     noise) AND can protect saturated objects (demo: red clothes).
   Demo placement wisdom: sector tools upstream of bleach (while the
   full scene-referred volume is available); bleach-then-tint shifts
   bleached highlights globally, tint-then-bleach re-neutralizes.
7. BRILLIANCE REDUCTION: SKIPPED for now (Marc). Darkens colors too
   bright for their saturation; inactive when no such colors. Add
   later if wanted.

### Why it composes
Every stage: pure function, flat bounded params, identity anchor ->
scipy + torch backprop solvers work as-is, waterfall attribution per
stage, paste-ready reports, companion DCTLs trivial (analytic, no
LUTs). A "Chromogen-ish" chain preset in the demo's typical order:
Sat(desat extremes) -> Sat(add) -> Crosstalk -> Contrast -> Sector* ->
Bleach -> Tint(warm highs) -> Tint(cold lows) -> Sat(final shaping).
Stage naming by job ("skin squash", "green-to-yellow") aligns with
the auto-naming plan already in this roadmap.

## PLAN C (Marc, 2026-07-20, for safekeeping)

1. LUT MATCHING: upload a .cube and fit the parametric stage chain to
   the LUT itself instead of measured patches. This works without any
   footage: a LUT is a function, so sample source points, apply the
   LUT (app.core.lut.apply_lut), and the input->output pairs ARE the
   patch pairs — solve_parametric doesn't care where targets came
   from. Design choices when built: sampling distribution (uniform
   lattice vs footage-realistic distribution vs Marc's existing
   1449-row LogC3 patch dataset — the dataset is attractive because it
   weights the fit toward colors that actually occur on real charts),
   domain coverage/weighting, and optionally reporting error through a
   DRT. Output as usual: per-stage waterfall + paste-ready DCTL
   sliders — i.e. "explain this LUT as Chromogen-style moves".
2. TRANSFER-FUNCTION DROPDOWN: the stops calibration (MID_GREY 0.391,
   STOP 0.0741 in app/core/chromogen.py) is hardcoded Arri LogC3
   EI800 — Marc only shoots LogC3 today. Later: a dropdown selecting
   the working transfer function (LogC3 / LogC4 / linear / ...) that
   sets MID_GREY + STOP per curve, in the DCTLs likely as a
   DCTLUI_COMBO_BOX. Keep slider VALUES in stops so looks stay
   portable across transfer functions.

## Chromogen solve modes + order preference (Marc, 2026-07-20)

Solve modes for the parametric solver (chain presets double as modes):
- "Reuleaux" mode = the existing Luma/RGB curves + Reuleaux Broad +
  Fine presets.
- "Chromogen match" mode (BUILT): Lift Gamma Gain prep -> Chromogen
  chain. LGG = master lift + master gamma + PER-CHANNEL gain (ganged =
  exposure, unganged = white balance) — smooth/monotone, cannot break
  the image. reg_scale=25 + fitted LAST in the stagewise init: the
  model assumes exposure/WB are fine and only moves prep when it makes
  the fit a LOT easier (verified by tests both ways). Rejected as prep:
  matrix (crosstalk is the look's job), free 1D curves (can kink/band).
- "Chromogen film look (full stack)" preset (BUILT) = Marc's real
  stack order: LGG -> Sat x2 -> Crosstalk -> Contrast -> sector tools
  (Squash/Sat/Brightness/Skew) -> Highlight Bleach -> Tint x2 -> final
  Sat. KEY ORDERING RULE from Marc + the demo: sector tools BEFORE
  Highlight Bleach (they want the full scene-referred volume); tints
  after bleach shift the bleached result. Contrast position is
  flexible (Andy: before/after sectors makes little difference).
ORDER PREFERENCE is soft: the presets encode the canonical order, but
Marc explicitly does not want to over-constrain the model — FUTURE:
an order-search option (solve a few candidate permutations, e.g.
bleach-before/after-sectors, contrast late, keep the best error) so
the solver can discover better orders than the prior.

Auto-naming (BUILT): every stage has label(params) -> short grading
note ("skew dark greens toward cyan", "cool lows", "bleach highlights
(spare blues)", "white balance + exposure trim", "(idle)"), shown in
the waterfall and CLI so a fitted chain reads top-level without
opening sliders.

Artifact KPI (BUILT): noise gain (app/core/diagnostics.py) — empirical
amplification of a small perturbation, per stage at its real input
distribution + whole chain, median/p95/max, reported next to the
residual everywhere. ~1 = transparent, >>1 = noise amplifier (caught
the sector-sat power-law bug class).

## LUT matching under a DRT — first real run (2026-07-20)

Genesis e100_base + openDRT test taught three things:
1. genesis_e100_base is DISPLAY-referred (header + measured S-curve) —
   the right composition is --target-is-display: solve
   DRT(chain(x)) ~= lut(x), "rebuild the look as chain under openDRT".
   Wrong composition (DRT after an already-rendered LUT) shows up as
   stages pinned at their bounds — a useful smell.
2. Result: display error 0.224 -> 0.109 mean (worst 0.53), 484/1395
   samples unreachable through the DRT (mostly gamut-edge colors the
   print rendering reaches but openDRT does not). The Chromogen chain
   alone cannot fully bridge two DIFFERENT renderings (2383 print vs
   openDRT) — the missing flexibility is mostly TONE. Option when
   wanted: a monotone Luma Curve pre-stage variant of the match mode
   (fit-only; no DCTL/drx node yet).
3. DRT MATH > DRT CUBE (Marc asked): a .cube DRT costs us trilinear
   plateaus + noisy numeric inversion (dropped patches) and blocks
   display-domain torch losses. openDRT is open source (Jed Smith
   DCTL) — porting Marc's exact config would give exact+cheap
   inversion, differentiable display-domain optimization, fewer
   drops. Candidate next step; cube path stays as the generic
   fallback for arbitrary DRTs.
