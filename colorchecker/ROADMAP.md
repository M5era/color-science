# Roadmap

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
