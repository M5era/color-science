"""Bake any named parametric stage into a .cube for Resolve A/B.

Works for every stage that declares `param_names` (the Chromogen-style
family). Values are given in DCTL SLIDER UNITS via --set, using the
exact slider names; unset params stay at identity. Example:

  python3 -m tools.stage_bake --stage "Colour Saturation" \
      --set "RG Saturation=1.4" --set "YB Saturation=1.8" \
      --set "Chroma=-0.5" --out sat.cube

Then A/B in Resolve: [the stage's DCTL with those sliders] vs [cube].
List stages and their sliders with --list. For Reuleaux Broad/Fine use
tools/reuleaux_bake.py / tools/reuleaux_fine_bake.py.
"""

import argparse
import sys

from app.core.match import write_cube
from app.core.stages import STAGE_POOL


def _named_stages():
    return {
        name: cls for name, cls in STAGE_POOL.items()
        if hasattr(cls, "param_names")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", help="stage name as shown in --list")
    parser.add_argument("--set", action="append", default=[],
                        metavar="NAME=VALUE",
                        help="slider value in DCTL units (repeatable)")
    parser.add_argument("--out", help="output .cube path")
    parser.add_argument("--size", type=int, default=33)
    parser.add_argument("--domain-min", type=float, default=0.0)
    parser.add_argument("--domain-max", type=float, default=1.0)
    parser.add_argument("--list", action="store_true",
                        help="list bakeable stages and their sliders")
    args = parser.parse_args()

    stages = _named_stages()
    if args.list or not args.stage:
        for name, cls in stages.items():
            print(f"{name}: {', '.join(cls.param_names)}")
        if not args.stage:
            return

    if args.stage not in stages:
        sys.exit(f"Unknown stage {args.stage!r} — choose from: "
                 + ", ".join(stages))
    if not args.out:
        sys.exit("--out is required to bake")

    stage = stages[args.stage]()
    params = stage.identity()
    names = stage.param_names
    lo, hi = stage.bounds()
    for item in args.set:
        name, _, raw = item.partition("=")
        name = name.strip()
        if name not in names:
            sys.exit(f"{args.stage} has no slider {name!r} — sliders: "
                     + ", ".join(names))
        i = names.index(name)
        value = float(raw)
        if not (lo[i] <= value <= hi[i]):
            sys.exit(f"{name}={value:g} outside bounds [{lo[i]:g}, {hi[i]:g}]")
        params[i] = value

    write_cube(
        lambda rgb: stage.apply(rgb, params),
        args.out,
        size=args.size,
        domain_min=args.domain_min,
        domain_max=args.domain_max,
        title=f"{args.stage} bake",
    )
    print(stage.describe(params))
    print(f"wrote {args.out} ({args.size}^3, domain "
          f"{args.domain_min}..{args.domain_max})")


if __name__ == "__main__":
    main()
