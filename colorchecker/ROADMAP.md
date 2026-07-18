# Roadmap

## Matching: Algorithm B — parametric spherical model (reuleaux-style)

Status: planned, NOT started. Do after the current toolchain is proven on
real footage.

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

Notes:
- Each stage should be individually toggleable + exportable (matrix
  values, curve points, zone params, cube for whatever remains).
- The more that lands in interpretable stages (1, 3, 4), the less the
  RBF has to explain — better extrapolation, smaller residual warp.
