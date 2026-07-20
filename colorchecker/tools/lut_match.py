"""Explain a LUT as parametric stage moves (Plan C item 1).

Loads a .cube, samples its domain (or a patch CSV), fits the chosen
stage chain to it, and prints the waterfall, the noise-gain artifact
KPI, and paste-ready DCTL slider values. Optionally exports the fitted
chain as a .cube for A/B against the original.

Usage (from the colorchecker/ directory):

  python3 -m tools.lut_match --lut somelook.cube
  python3 -m tools.lut_match --lut somelook.cube --backend torch \
      --source-csv all_EV0.csv --out fitted.cube

Default chain is the "Chromogen match" mode: Lift Gamma Gain prep
(strongly anchored at identity — it only moves if it makes the fit a
LOT easier) followed by the Chromogen look chain. --preset picks any
chain preset; --list-presets shows them.
"""

import argparse

from app.core.lut import parse_cube
from app.core.lut_match import solve_lut_match
from app.core.match import load_patch_csv, write_cube
from app.core.stages import CHAIN_PRESETS, STAGE_POOL

DEFAULT_PRESET = "Chromogen match (LGG prep → Chromogen chain)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lut", help=".cube to explain")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--backend", choices=["scipy", "torch"],
                        default="scipy")
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--source-csv",
                        help="patch CSV to use as source points instead "
                             "of the lattice sample")
    parser.add_argument("--out", help="export the fitted chain as .cube")
    parser.add_argument("--size", type=int, default=33)
    parser.add_argument("--drx-out",
                        help="also write a PowerGrade with the fitted "
                             "values patched into the template's DCTL nodes")
    parser.add_argument("--drx-template",
                        default="templates/example_powergrade_1.6.1.T.drx")
    args = parser.parse_args()

    if args.list_presets or not args.lut:
        for name, chain in CHAIN_PRESETS.items():
            print(f"{name}: {' -> '.join(chain)}")
        if not args.lut:
            return

    lut = parse_cube(args.lut)
    stages = [STAGE_POOL[name]() for name in CHAIN_PRESETS[args.preset]]

    source_points = None
    if args.source_csv:
        source_points, _ = load_patch_csv(args.source_csv)

    result = solve_lut_match(
        lut, stages,
        source_points=source_points,
        n_samples=args.samples,
        backend=args.backend,
    )

    print(f"Backend: {result.backend}   pairs: {result.pairs_used}")
    print(f"Error before: {result.error_before:.5f}")
    for (name, err), (_, gain), label in zip(result.waterfall,
                                             result.stage_noise_gain,
                                             result.stage_labels):
        shown = name if label == name else f"{name} — {label}"
        print(f"  after {shown}: {err:.5f}   "
              f"[noise gain ×{gain['median']:.2f}, max ×{gain['max']:.2f}]")
    print(f"After match: {result.error_after:.5f} "
          f"(worst {result.error_after_max:.5f})")
    g = result.chain_noise_gain
    print(f"Chain noise gain: ×{g['median']:.2f} median, ×{g['max']:.2f} max")
    print()
    print("Node names (<=9 chars for Resolve):",
          " | ".join(s.short_label(p) for s, p in
                     zip(result.model.stages, result.model.params)))
    print()
    for report in result.stage_reports:
        print(report)
        print()

    if args.out:
        write_cube(
            result.model, args.out, size=args.size,
            domain_min=float(lut.domain_min[0]),
            domain_max=float(lut.domain_max[0]),
            title="Parametric LUT match",
        )
        print(f"wrote {args.out} ({args.size}^3) — A/B against {args.lut}")

    if args.drx_out:
        from app.core.drx import DrxTemplate

        drx = DrxTemplate(args.drx_template)
        counters: dict = {}
        unmatched = []
        for stage, params, label in zip(result.model.stages,
                                        result.model.params,
                                        result.stage_labels):
            name = stage.name.replace(" ", "")
            matches = [n for n in drx.nodes if n.dctl_name == name]
            k = counters.get(name, 0)
            counters[name] = k + 1
            if k >= len(matches):
                unmatched.append(f"{stage.name} — {label}")
                continue
            node = matches[k]
            for i, value in enumerate(params):
                if i in node._offsets:
                    drx.set_slider(node, i, float(value))
            print(f"drx node {name}#{k} <- {label}  "
                  f"[node name: {stage.short_label(params)}]")
        drx.write(args.drx_out)
        print(f"wrote {args.drx_out} (template: {args.drx_template})")
        if unmatched:
            print("NO NODE IN TEMPLATE for these fitted stages — add the "
                  "DCTL node to your powergrade and re-export it as the "
                  "template, or paste the sliders by hand:")
            for u in unmatched:
                print(f"  {u}")


if __name__ == "__main__":
    main()
