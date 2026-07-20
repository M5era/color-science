"""Generate a Resolve PowerGrade (.drx) with fitted stage parameters.

Clones a template .drx whose node stack already contains our DCTL
nodes (e.g. templates/contrast_boost_1.6.4.T.drx, built by Marc)
and writes slider values into them — nodes are matched to stages by
DCTL filename, sliders by position (our param_names order == the
DCTL's DEFINE_UI_PARAMS order). Patching is fixed-width, so nothing
shifts; bodies are re-zstd-compressed on save.

Usage (from the colorchecker/ directory):

  python3 -m tools.drx_export --template templates/contrast_boost_1.6.4.T.drx --list
  python3 -m tools.drx_export --template ... --out fitted.drx \
      --set "ColourSaturation:R/G=1.15" --set "ColourSaturation:Y/B=1.45" \
      --set "NeutralTint:Amount=0.3" --set "NeutralTint:Hue=40"

Node selectors: DCTL name, optionally with #k for the k-th node of
that type (0-based), e.g. "SectorSkew#1:Skew=12".
"""

import argparse

from app.core.drx import DrxTemplate
from app.core.stages import STAGE_POOL

# DCTL filename -> stage (for slider names); extend as templates grow
_DCTL_TO_STAGE = {
    cls().name.replace(" ", ""): cls for cls in
    (STAGE_POOL[n] for n in STAGE_POOL)
    if hasattr(cls, "param_names")
}


def _slider_names(node):
    cls = _DCTL_TO_STAGE.get(node.dctl_name)
    return list(cls.param_names) if cls is not None else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--set", action="append", default=[],
                        metavar="NODE:SLIDER=VALUE")
    parser.add_argument("--out")
    args = parser.parse_args()

    drx = DrxTemplate(args.template)

    if args.list or not args.set:
        for i, node in enumerate(drx.nodes):
            names = _slider_names(node)
            print(f"[{i}] {node.dctl_name}  ({node.dctl_path})")
            for idx in sorted(node.sliders):
                label = names[idx] if idx < len(names) else f"param{idx}"
                print(f"      {idx:2d} {label:16s} = {node.sliders[idx]:g}")
        if not args.set:
            return

    counters: dict = {}
    for item in args.set:
        selector, _, assignment = item.partition(":")
        slider, _, raw = assignment.partition("=")
        name, _, k = selector.partition("#")
        k = int(k) if k else 0

        matches = [n for n in drx.nodes if n.dctl_name == name]
        if not matches:
            raise SystemExit(f"No node for DCTL {name!r} in the template — "
                             f"nodes: {[n.dctl_name for n in drx.nodes]}")
        if k >= len(matches):
            raise SystemExit(f"Only {len(matches)} {name!r} node(s); "
                             f"#{k} does not exist")
        node = matches[k]
        names = _slider_names(node)
        if slider.strip() in names:
            idx = names.index(slider.strip())
        else:
            try:
                idx = int(slider)
            except ValueError:
                raise SystemExit(
                    f"{name} has no slider {slider!r} — sliders: {names}"
                ) from None
        drx.set_slider(node, idx, float(raw))
        print(f"set {name}#{k} {slider.strip()} = {float(raw):g}")

    if not args.out:
        raise SystemExit("--out is required when patching")
    drx.write(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
