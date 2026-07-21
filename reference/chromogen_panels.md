# Chromogen panel reference (Baselight screenshots, transcribed)

Slider-by-slider record of Marc's Baselight/Chromogen panel
screenshots (sent 2026-07-20 and 2026-07-21), the evidence they give
about ranges/defaults/units, and how our stages + DCTLs map onto them.
This is the calibration source for "defaults do nothing, ranges feel
like the original".

NOTE ON THE IMAGES: the actual screenshot files could not be saved
from the chat — this file is a detailed transcription instead. If the
originals matter, drop the PNGs into `reference/screenshots/` and link
them from here.

---

## Global observations (all panels)

- Every tool panel: tool sliders on top, then a **Modulation**
  section, then an **Extended Ranges** checkbox (which presumably
  widens the slider ranges — exact extended values unknown).
- **Standard Modulation defaults are 0.000 / 0.000 / 0.000**
  (Zone / Pivot / Chroma), every knob dead-centre — seen identically
  on Colour Saturation, Sector Skew, Colour Crosstalk. Matches our
  signed Zone/Pivot/Chroma with identity 0.
- **Slider bars are semantic graphics**, not decoration:
  - Zone: black -> white gradient (left = shadows, right = highlights).
    Matches our signed Zone (left = shadows only, right = highlights).
  - Pivot: a STEPPED grey scale showing the luma level the pivot sits
    on at each position. This is the key unit evidence (below).
  - Chroma: colourful on the LEFT half fading to grey on the RIGHT.
    So in Baselight, LEFT = saturated colors, RIGHT = neutrals —
    the **opposite direction of our signed Chroma** (ours: right =
    saturated only; Marc's chosen convention, kept deliberately).
- **Pivot units are stops-compatible.** Two independent checks put the
  Baselight pivot slider on our −6..+8 stop scale:
  - Highlight Bleach default Pivot −2.00 sits at ~28% of its bar;
    (−2 + 6) / 14 = 28.6%.
  - Neutral Tint default Pivot −0.70 sits at ~38-43% and Marc reads
    the grey under it as "basically middle grey"; (−0.7 + 6) / 14 =
    37.9%, and −0.7 stops is indeed barely under mid-grey.
  Working hypothesis: Baselight pivots ≈ stops from mid-grey with a
  −6..+8 visible range — exactly our convention. Falloffs are then
  most likely widths in stops too (adopted for our defaults).

---

## Per-tool panels

### Colour Saturation (screenshot 2026-07-21)
| Slider | Default | Knob | Bar | Ours |
|---|---|---|---|---|
| R/G | 1.00 | centre | green<->magenta colours | 1.0, range 0..2 (was 0..3, tightened to centre the identity) |
| Y/B | 1.00 | centre | yellow<->blue colours | 1.0, range 0..2 |
| Zone | 0.000 | centre | black->white | 0, −1..1 ✓ |
| Pivot | 0.000 | centre | stepped greys | 0 stops, −6..8 ✓ |
| Chroma | 0.000 | centre | colours->grey | 0, −1..1 ✓ (sign flipped vs Baselight, see above) |

R/G and Y/B have gang-link icons in Baselight (move together unless
unlinked); our two sliders are always independent — set both for the
ganged move.

### Contrast Boost (screenshot 2026-07-21)
| Slider | Default | Knob | Bar | Ours |
|---|---|---|---|---|
| Contrast Boost | 0.00 | FAR LEFT | grey->white | 0, range now 0..2 (dropped the −0.9 flattening range — panel has none) |
| Grey Pivot | 0.00 | centre | stepped greys | 0 stops, −4..4 ✓ |
| Highlight Pivot | 6.00 | ~55-60% | stepped greys, bright | 6 stops, range 0.5..14 (our 6 sits at 41% — Baselight's range is likely different, maybe −4..14 or 0..10; value matches, range unresolved) |
| Chroma | 0.50 | centre | greys->colours | 0.5, 0..1 ✓ (val-only vs per-RGB mix) |

### Colour Crosstalk (screenshot 2026-07-21, Marc's grade — not defaults)
Shown values R->Y/B 0.00, Y->R/G −0.07, G->Y/B 0.00, B->R/G −0.07
(knobs a hair left of centre), Modulation 0/0/0. Marc confirms the
DEFAULT for all four is 0.00. Bars are two-colour gradients per slider
(the two directions the sector can tilt). Ours: 0 default, −1..1 ✓.
Marc's actual look: a gentle Y and B pull of −0.07 each.

### Highlight Bleach (screenshot 2026-07-21)
| Slider | Default | Knob | Bar | Ours |
|---|---|---|---|---|
| R | 0.00 | FAR LEFT | pink->white | 0, 0..1 ✓ |
| Y | 0.00 | FAR LEFT | orange->white | 0, 0..1 ✓ |
| G | 0.00 | FAR LEFT | green->white | 0, 0..1 ✓ |
| B | 0.00 | FAR LEFT | blue->white | 0, 0..1 ✓ |
| Pivot | −2.00 | ~28% | stepped greys | −2 stops ✓ (exact match; also the stops-unit evidence) |
| Falloff | 0.500 | centre | grey gradient | default now 0.5 stops (was 4.0); range 0.1..16 — reading Baselight's 0.5 as stops: a soft-kneed threshold ~2 stops under mid-grey |
| Chroma | 0.00 | centre | colours->grey | 0 ✓ |

The colour->white amount bars are the tool itself: raise a sector's
slider and that sector's highlights bleach to white.

### Neutral Tint (screenshot 2026-07-20, re-analysed)
| Slider | Default | Knob | Bar | Ours |
|---|---|---|---|---|
| Hue | 0.0 | CENTRE | full spectrum | ours 0..360° with 0 at the left (reuleaux red). Baselight's is a signed slider centred at 0 — what colour their 0.0 means is UNKNOWN (open question) |
| Amount | 0.000 | centre | black->white | 0, −1..1 ✓ — bar confirms left = tint shadows, right = tint highlights |
| Pivot | −0.70 | ~38-43% | stepped greys | 0 stops (= mid grey, per Marc's reading of the bar) |
| Falloff | 1.0000 | centre | grey gradient | default now 1.0 stop (was 4.0); range 0.1..16 — near-clean shadow/highlight split at mid-grey, falloff widens it |
| Chroma | 1.0000 | centre | colours->grey | 1.0, range 0..2 ✓ (1 = everything, 0 = only saturated, 2 = only neutrals) |

v3 of our stage applies the tint in LOG RGB (sum-preserving offset);
TINT_SCALE 0.15 at full throw.

### Brilliance Reduction (screenshot 2026-07-20)
| Slider | Default | Knob | Bar | Ours |
|---|---|---|---|---|
| Amount | 1.000000 | FAR RIGHT | rainbow, darkening left | 1.0, 0..1 ✓ (1 = identity, pull DOWN to reduce) |
| Chroma | 0.6000 | ~60% | saturated colours | 0.6, 0..1 ✓ (mask strength) |
| Pivot | 0.35000 | ~35% | greys | 0.35, 0..1 ✓ (SAT units — where the ramp starts biting) |
| Falloff | 0.500 | ~50% | greys | 0.5, 0..1 ✓ (SAT units — ramp width) |

All three mask sliders live in the saturation domain (Marc). All four
ranges inferred 0..1 from knob positions — the one panel where every
value/percentage lines up self-consistently.

### The four Sector tools (screenshots 2026-07-21, second batch)

All four panels share the layout Hue / <amount> / Falloff +
Modulation, and every default matches our stages — no code changes
were needed from this batch.

| Slider | Baselight default | Knob | Ours |
|---|---|---|---|
| Hue (all four) | 0.0 | CENTRE of spectrum bar | 0..360°, 0 = red. Baselight's centred 0.0 = **yellow** in their representation (Marc) — deliberately NOT copied, ours stays reuleaux-red-at-0 ("not important, keep as is") |
| Skew | 0.00 | centre | 0 in ±60° ✓ (bar: the two skew directions, green<->red at yellow) |
| Brightness | 0.00 | centre | 0 in ±3 ✓ (bar: dark -> bright orange) |
| Saturation | 1.00 | centre | 1.0 in 0..2 ✓ — confirms our linear 0..2 range with the identity dead-centre (bar: grey -> saturated orange) |
| Squash | 0.00 | centre | 0 in ±1 ✓ (bar: spread <-> squash, green<->orange) |
| Falloff (all four) | 0.500 / 0.50 | centre | ours 60° of hue, range 5..180. Baselight's hue-domain falloff is normalized (0..1, mid default); unit unresolved — if 0..1 maps to 0..120° their default equals our 60°, but that's a guess |
| Zone/Pivot/Chroma | 0/0/0 | centre | ✓ all match |

(The Sector Saturation screenshot came from Marc's grade with the node
named "Dark Red Boost", sliders at defaults.)

---

## Answered (Marc, 2026-07-21)

- **Hue 0.0 in Baselight = yellow**, sliders signed around it. Not
  important — our 0..360°/red-at-0 convention stays.
- **Extended Ranges**: the sliders stay the same, the effect roughly
  DOUBLES. So our wider-than-panel solver bounds (e.g. pivot stops,
  falloff stops up to 16) are the moral equivalent — fine as-is.
- **Chroma bar direction** (Baselight: left = saturated; ours: right =
  saturated): keep ours.

## Open questions / missing screenshots

1. **The "?" tooltips** — every slider has one; their text would
   settle units definitively (Pivot, Falloff, Chroma). Marc: "later".
2. **Falloff units**: hue-domain (Sector tools' 0.500 normalized —
   what width in degrees?) and confirmation that luma-ramp falloffs
   are stops (Bleach 0.5, Tint 1.0 — currently adopted as stops on
   the strength of the pivot-bar evidence).
3. **Contrast Boost Highlight Pivot range** (its 6.00 sits at ~55-60%
   of the bar; ours sits at 41%).
