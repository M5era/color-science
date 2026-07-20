"""Bake the Reuleaux-port transform into a .cube for evaluation.

Set the same slider values as in ReuleauxUserStandalone.dctl, bake,
then A/B in Resolve: [DCTL node with those sliders] vs [this cube].

Usage (from the colorchecker/ directory):

  python3 -m tools.reuleaux_bake --out test.cube \
      --red 0.05 1.2 0.0 --blue -0.02 0.9 0.3 --overall-sat 1.1

Each color takes three numbers: HUE SAT VAL (DCTL slider values,
defaults 0 1 0). --size / --domain-min / --domain-max as in the
Matching tab exports; --invert mirrors the DCTL checkbox.
"""

import argparse

from app.core.match import write_cube
from app.core.reuleaux import ReuleauxUserParams, reuleaux_user


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output .cube path")
    parser.add_argument("--size", type=int, default=33)
    parser.add_argument("--domain-min", type=float, default=0.0)
    parser.add_argument("--domain-max", type=float, default=1.0)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--overall-sat", type=float, default=1.0)
    parser.add_argument("--overall-val", type=float, default=0.0)
    for color in ("red", "yellow", "green", "cyan", "blue", "magenta"):
        parser.add_argument(
            f"--{color}", nargs=3, type=float, default=[0.0, 1.0, 0.0],
            metavar=("HUE", "SAT", "VAL"),
        )
    args = parser.parse_args()

    params = ReuleauxUserParams(
        overall_sat=args.overall_sat,
        overall_val=args.overall_val,
        red=tuple(args.red), yellow=tuple(args.yellow), green=tuple(args.green),
        cyan=tuple(args.cyan), blue=tuple(args.blue), magenta=tuple(args.magenta),
    )
    write_cube(
        lambda rgb: reuleaux_user(rgb, params, invert=args.invert),
        args.out,
        size=args.size,
        domain_min=args.domain_min,
        domain_max=args.domain_max,
        title="Reuleaux port bake",
    )
    print(f"wrote {args.out} ({args.size}^3, domain {args.domain_min}..{args.domain_max})")


if __name__ == "__main__":
    main()
