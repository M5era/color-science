
## openDRT analytic port (staged 2026-07-20, Marc-approved)

Marc supplied the openDRT DCTL source (reference/OpenDRT.dctl, Jed
Smith, GPLv3 — port module must carry the license; private use
unrestricted, only distribution triggers GPL obligations). Header says
v1.1.0; the file has the look/tonescale preset system (Standard/
Arriba/Sylvan/... x tonescale presets x creative white).

WHY: replace the baked openDRT cube in the DRT sandwich with exact
math — exact/cheap inversion (no trilinear plateaus -> far fewer
dropped patches), and a DIFFERENTIABLE torch mirror so backprop can
optimize display-domain loss directly.

PLAN (own session, reuleaux-port discipline — 1:1 transcription,
float64, tests first):
1. Port to app/core/opendrt.py: gamut matrices, input OETFs (LogC3
   path matters), tonescale, purity/render modules, display encodings,
   preset tables. Vectorized numpy; torch mirror after parity.
2. SETTINGS: Marc will supply a powergrade containing his openDRT
   node -> our drx parser reads the comboBox/slider params = exact
   settings. Fallback: grid-search the preset combos (in_gamut=AWG3,
   in_oetf=LogC3, display=sRGB, 7 look presets x tonescale presets)
   against his baked cube and keep the pixel-match.
3. VALIDATION GATE: port(settings) vs openDRT_LogC3_srgb cube on a
   dense lattice, tolerance ~cube quantization. Then wire as
   output_transform option in lut_match (--drt-math) and the Matching
   tab, incl. analytic inversion (per-channel monotone tonescale ->
   fast newton) replacing invert_lut_at where available.
