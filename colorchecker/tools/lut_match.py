"""Explain a LUT as parametric stage moves (Plan C item 1).

Loads a .cube, samples its domain (or a patch CSV), fits the chosen
stage chain to it, and prints the waterfall, the noise-gain artifact
KPI, and paste-ready DCTL slider values. Optionally exports the fitted
chain as a .cube for A/B against the original.

Usage (from the colorchecker/ directory):

  python3 -m tools.lut_match --lut somelook.cube
  python3 -m tools.lut_match --lut somelook.cube --backend torch \
      --source-csv all_EV0.csv --out fitted.cube

FREE-ORDER SEARCH MODE (Marc's 2026-07-21 pipeline): pass --search and
no chain is prescribed at all — the solver auditions every Chromogen
tool (no Lift Gamma Gain), reuses types as often as they help, and
discovers its own order; --max-nodes is the only structural limit.

  python3 -m tools.lut_match --lut somelook.cube --search \
      --max-nodes 10 --backend torch --deliver

--deliver drops the fitted .cube AND the patched .drx straight into
~/Downloads (for running the script locally on the Mac).

Preset mode is still available: default chain is the "Chromogen match"
mode; --preset picks any chain preset; --list-presets shows them.
"""

import argparse
from pathlib import Path

from app.core.lut import parse_cube
from app.core.lut_match import search_lut_match, solve_lut_match
from app.core.match import load_patch_csv, write_cube
from app.core.stages import CHAIN_PRESETS, STAGE_POOL

DEFAULT_PRESET = "Chromogen match (LGG prep → Chromogen chain)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lut", help=".cube to explain")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--search", action="store_true",
                        help="free-order chain search: no preset, no "
                             "LGG — every Chromogen tool auditioned, "
                             "reuse allowed, order discovered")
    parser.add_argument("--max-nodes", type=int, default=10,
                        help="search mode: maximum nodes in the chain")
    parser.add_argument("--min-gain", type=float, default=0.005,
                        help="search mode: minimum relative fit "
                             "improvement a new node must deliver "
                             "(0.005 = 0.5%%)")
    parser.add_argument("--broad-bias", type=float, default=0.15,
                        help="search mode: discount applied to "
                             "single-hue (Sector) tool auditions so "
                             "broad tools get a slight preference; "
                             "0 disables (default 0.15)")
    parser.add_argument("--free-tone", action="store_true",
                        help="search mode: DISABLE the grey-scale-"
                             "locked tone (by default one Contrast "
                             "Curve is fitted on the neutral ramp "
                             "only, frozen as node 1, and removed "
                             "from the audition pool)")
    parser.add_argument("--local-search", action="store_true",
                        help="search mode: light local search — un-"
                             "freeze the tone node to co-adapt with a "
                             "Neutral Tint (tinted neutrals / crossover), "
                             "and prune redundant nodes after building "
                             "(noise-gain-aware). Off by default")
    parser.add_argument("--prune-full", action="store_true",
                        help="local search: re-refine EVERY node each "
                             "prune round (thorough, slow) instead of the "
                             "default fast screen of the 4 cheapest-to-"
                             "drop candidates")
    parser.add_argument("--deliver", action="store_true",
                        help="write the fitted .cube and .drx into "
                             "~/Downloads (implies --out/--drx-out)")
    parser.add_argument("--backend", choices=["scipy", "torch"],
                        default="scipy")
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--drt",
                        help="DRT .cube to work under (display-referred "
                             "sandwich: fit in log, errors through the DRT; "
                             "stack the result BEFORE the DRT node)")
    parser.add_argument("--drt-math", action="store_true",
                        help="use the ANALYTIC openDRT (Marc's exact "
                             "config, app/core/opendrt.py) instead of a "
                             "baked --drt cube: display-domain loss, no "
                             "inversion, no unreachable-dropping. scipy "
                             "backend only.")
    parser.add_argument("--target-is-display", action="store_true",
                        help="the LUT already outputs display (print "
                             "emulation etc.) — rebuild it as chain+DRT: "
                             "solve DRT(chain(x)) ~= lut(x)")
    parser.add_argument("--source-csv",
                        help="patch CSV to use as source points instead "
                             "of the lattice sample")
    parser.add_argument("--out", help="export the fitted chain as .cube")
    parser.add_argument("--size", type=int, default=33)
    parser.add_argument("--drx-out",
                        help="also write a PowerGrade with the fitted "
                             "values patched into the template's DCTL nodes")
    parser.add_argument("--drx-template",
                        default="templates/brilliance_red_1.4.1.T.drx")
    args = parser.parse_args()

    if args.list_presets or not args.lut:
        for name, chain in CHAIN_PRESETS.items():
            print(f"{name}: {' -> '.join(chain)}")
        if not args.lut:
            return

    lut = parse_cube(args.lut)

    if args.deliver:
        downloads = Path.home() / "Downloads"
        base = Path(args.lut).stem + "_fit"
        if not args.out:
            args.out = str(downloads / f"{base}.cube")
        if not args.drx_out:
            args.drx_out = str(downloads / f"{base}.drx")

    source_points = None
    if args.source_csv:
        source_points, _ = load_patch_csv(args.source_csv)

    # fail fast on the export prerequisites BEFORE the expensive solve:
    # a 20-node search once completed and then died at the drx step on
    # a missing zstandard module
    drx_template_obj = None
    if args.drx_out:
        from app.core.drx import DrxTemplate  # needs zstandard

        drx_template_obj = DrxTemplate(args.drx_template)

    drt_math = None
    if args.drt_math:
        from app.core.opendrt import OpenDRTModel

        if args.backend == "torch":
            raise SystemExit(
                "--drt-math has no torch mirror yet — drop --backend torch"
            )
        drt_math = OpenDRTModel()
        drt = None
    else:
        drt = parse_cube(args.drt) if args.drt else None
    if args.search:
        result = search_lut_match(
            lut,
            max_nodes=args.max_nodes,
            min_gain=args.min_gain,
            broad_bias=args.broad_bias,
            neutral_tone=not args.free_tone,
            local_search=args.local_search,
            prune_screen_k=0 if args.prune_full else 4,
            source_points=source_points,
            n_samples=args.samples,
            backend=args.backend,
            drt=drt,
            target_is_display=args.target_is_display,
            verbose=True,
            drt_math=drt_math,
        )
        # insurance: persist the found chain immediately, so a failed
        # export step can never cost the search
        if args.out or args.drx_out:
            import json

            chain_path = Path(args.out or args.drx_out).with_suffix(
                ".chain.json")
            chain_path.write_text(json.dumps({
                "lut": args.lut,
                "stages": [s.name for s in result.model.stages],
                "params": [p.tolist() for p in result.model.params],
            }, indent=2))
            print(f"wrote {chain_path} (chain spec)")
        print()
        print("Search log:")
        for entry in result.search_log:
            if isinstance(entry, str):
                print(f"  {entry}")
            else:
                n, name, err = entry
                print(f"  node {n}: {name}  fit error -> {err:.5f}")
        print()
    else:
        stages = [STAGE_POOL[name]() for name in CHAIN_PRESETS[args.preset]]
        result = solve_lut_match(
            lut, stages,
            source_points=source_points,
            n_samples=args.samples,
            backend=args.backend,
            drt=drt,
            target_is_display=args.target_is_display,
            drt_math=drt_math,
        )

    print(f"Backend: {result.backend}   pairs: {result.pairs_used}"
          + (f"   (dropped {result.pairs_unreachable} unreachable through DRT)"
             if result.display_referred else ""))
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
        from app.core.stages import STAGE_POOL as _POOL

        drx = drx_template_obj

        export_result = result
        if args.search and len(result.model.stages) > 1:
            # A .drx can only run the template's own node order. Map
            # each found stage to its k-th node of that type; if the
            # discovered order differs, REFIT the same stage set in
            # template order (warm-started) so the exported grade is
            # optimal for the order it will actually run in.
            counters: dict = {}
            keyed = []
            for pos, (stage, params) in enumerate(
                    zip(result.model.stages, result.model.params)):
                name = stage.name.replace(" ", "")
                node_ids = [i for i, n in enumerate(drx.nodes)
                            if n.dctl_name == name]
                k = counters.get(name, 0)
                counters[name] = k + 1
                idx = node_ids[k] if k < len(node_ids) else 10_000 + pos
                keyed.append((idx, stage, params))
            order = sorted(range(len(keyed)), key=lambda j: keyed[j][0])
            if order != list(range(len(keyed))):
                print("search order != template node order — refitting "
                      "in template order for the .drx export (assumes "
                      "the template's stored node order is its graph "
                      "order — verify the stack in Resolve)")
                export_result = solve_lut_match(
                    lut, [keyed[j][1] for j in order],
                    source_points=source_points,
                    n_samples=args.samples,
                    backend=args.backend,
                    drt=drt,
                    target_is_display=args.target_is_display,
                    init_params=[keyed[j][2] for j in order],
                    drt_math=drt_math,
                )
                print(f".drx (template-order) error: "
                      f"{export_result.error_after:.5f}   "
                      f"free-order error: {result.error_after:.5f}")

        counters = {}
        unmatched = []
        patched_ids = set()
        for stage, params, label in zip(export_result.model.stages,
                                        export_result.model.params,
                                        export_result.stage_labels):
            name = stage.name.replace(" ", "")
            matches = [n for n in drx.nodes if n.dctl_name == name]
            k = counters.get(name, 0)
            counters[name] = k + 1
            if k >= len(matches):
                unmatched.append(f"{stage.name} — {label}")
                continue
            node = matches[k]
            missing = []
            for i, value in enumerate(params):
                if i in node._offsets:
                    drx.set_slider(node, i, float(value))
                elif abs(value - stage.identity()[i]) > 1e-9:
                    missing.append(f"{stage.param_names[i]}={value:.4f}")
            patched_ids.add(id(node))
            print(f"drx node {name}#{k} <- {label}  "
                  f"[node name: {stage.short_label(params)}]")
            if missing:
                print(f"  WARNING: {name}#{k} sliders not stored in the "
                      f"template, set by hand: {', '.join(missing)} "
                      "(wiggle them once in Resolve and re-save the "
                      "template to fix)")

        # template nodes carry the grade they were saved with — reset
        # every look node the fit did NOT use to its stage identity so
        # the exported powergrade is EXACTLY the fitted chain (+ DRT)
        by_dctl = {cls.name.replace(" ", ""): cls for cls in _POOL.values()}
        resets = 0
        for node in drx.nodes:
            if id(node) in patched_ids or node.dctl_name not in by_dctl:
                continue
            identity = by_dctl[node.dctl_name]().identity()
            for i, value in enumerate(identity):
                if i in node._offsets:
                    drx.set_slider(node, i, float(value))
            resets += 1
        if resets:
            print(f"reset {resets} unused look nodes to identity")

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
