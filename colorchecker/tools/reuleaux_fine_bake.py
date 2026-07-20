"""Bake a Reuleaux Fine zone into a .cube for evaluation.

Set the same slider values as in dctl/ReuleauxFine.dctl (hue arguments
in DEGREES, exactly like the DCTL sliders and the solver's stage
report), bake, then A/B in Resolve: [ReuleauxFine.dctl node with those
sliders] vs [this cube].

Usage (from the colorchecker/ directory):

  python3 -m tools.reuleaux_fine_bake --out fine.cube \
      --hue-center 130 --hue-core 15 --hue-soft 25 \
      --hue-shift 8 --sat 1.3 --val 0.2 \
      --luma-mask 0.2 0.15 0.2

Masks default to wide open (off), matching the DCTL defaults.
"""

import argparse

from app.core.match import write_cube
from app.core.stages import ReuleauxFineStage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output .cube path")
    parser.add_argument("--size", type=int, default=33)
    parser.add_argument("--domain-min", type=float, default=0.0)
    parser.add_argument("--domain-max", type=float, default=1.0)
    parser.add_argument("--hue-center", type=float, default=0.0,
                        help="zone center in degrees, 0..360")
    parser.add_argument("--hue-core", type=float, default=14.4,
                        help="full-strength half-width, degrees")
    parser.add_argument("--hue-soft", type=float, default=28.8,
                        help="falloff width, degrees")
    parser.add_argument("--hue-shift", type=float, default=0.0,
                        help="gated hue shift, degrees")
    parser.add_argument("--sat", type=float, default=1.0,
                        help="gated sat factor (1 = neutral)")
    parser.add_argument("--val", type=float, default=0.0,
                        help="gated val slider (0 = neutral)")
    parser.add_argument("--luma-mask", nargs=3, type=float,
                        default=list(ReuleauxFineStage._LUMA_OPEN),
                        metavar=("CENTER", "CORE", "SOFT"))
    parser.add_argument("--sat-mask", nargs=3, type=float,
                        default=list(ReuleauxFineStage._SAT_OPEN),
                        metavar=("CENTER", "CORE", "SOFT"))
    args = parser.parse_args()

    stage = ReuleauxFineStage()
    params = stage.identity()
    params[0] = args.hue_center / 360.0
    params[1] = args.hue_core / 360.0
    params[2] = args.hue_soft / 360.0
    params[3] = args.hue_shift / 360.0
    params[4] = args.sat
    params[5] = args.val
    params[6:9] = args.luma_mask
    params[9:12] = args.sat_mask

    write_cube(
        lambda rgb: stage.apply(rgb, params),
        args.out,
        size=args.size,
        domain_min=args.domain_min,
        domain_max=args.domain_max,
        title="Reuleaux Fine zone bake",
    )
    print(stage.describe(params))
    print(f"wrote {args.out} ({args.size}^3, domain {args.domain_min}..{args.domain_max})")


if __name__ == "__main__":
    main()
