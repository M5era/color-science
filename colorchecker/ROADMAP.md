# Roadmap

## Matching: Algorithm B — parametric zone model (refined 2026-07)

Status: planned, NOT started. Do after the current toolchain is proven on
real footage.

REFINED DESIGN (supersedes plain reuleaux port): custom hue-anchored
zone model + companion DCTL that we author ourselves.
- Space: cylindrical hue/chroma/value about the neutral axis on log
  input (cf. thatcherfreeman Cylindrical DCTL); reuleaux is design
  reference only, not a dependency.
- Model: N adjustment units (start N=3), each with FREE hue anchor +
  width, value (luminance) center + width, and delta hue/chroma/value
  with smooth Gaussian falloff — not tied to fixed R/G/B/C/M/Y sliders.
  ~6-7 params/unit, cap ~20 total.
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

Per-stage export (the point of the whole design):
- Stage 1: the 9 matrix numbers (RGB Mixer / matrix DCTL)
- Stage 2a reuleaux: the fitted PARAMETER SET, portable 1:1 into the
  reuleaux DCTL in Resolve — fully parametric, no LUT
- Stage 2b RBF: 3D .cube (only stage that inherently needs a LUT)
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
