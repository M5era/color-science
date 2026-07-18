"""CSV export: a view over the project's stored patch results.

Combined long format, one row per patch per included exposure:

    label,ev,group,overlay,kind,patch_row,patch_col,R,G,B

Exposures appear in session-list order; within an exposure, overlays in
stored order (chart grids, light-source squares), patches row-major —
the deterministic ordering that lines up row-for-row against a
companion capture exported with the same conventions. `kind` separates
reflective chart patches from emissive light-source samples. Values are
written with repr-level precision (no rounding).
"""

import io

from app.core.project import ImageEntry

HEADER = "label,ev,group,overlay,kind,patch_row,patch_col,R,G,B"


def _escape(text: str) -> str:
    if any(ch in text for ch in ',"\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


def combined_csv(entries: list[ImageEntry]) -> str:
    """CSV text for all included entries that have patch results."""
    out = io.StringIO()
    out.write(HEADER + "\n")
    for entry in entries:
        if not entry.include or not entry.patch_results:
            continue
        ev_text = "" if entry.ev is None else f"{entry.ev:g}"
        prefix = f"{_escape(entry.label)},{ev_text},{_escape(entry.group)}"
        for result in entry.patch_results:
            r, g, b = result["rgb"]
            overlay = _escape(result.get("overlay", "Overlay 1"))
            kind = result.get("kind", "reflective")
            out.write(
                f"{prefix},{overlay},{kind},"
                f"{result['row']},{result['col']},{r!r},{g!r},{b!r}\n"
            )
    return out.getvalue()


def exportable_count(entries: list[ImageEntry]) -> tuple[int, int]:
    """(number of exportable entries, number skipped for missing results)."""
    included = [e for e in entries if e.include]
    with_results = [e for e in included if e.patch_results]
    return len(with_results), len(included) - len(with_results)
