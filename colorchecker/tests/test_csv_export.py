"""CSV export ordering, filtering, and full-precision values."""

from app.core.csv_export import combined_csv, exportable_count
from app.core.project import ImageEntry


def _entry(label, ev, include=True, results=None, group=""):
    return ImageEntry(
        source_path=f"/x/{label}.tif",
        label=label,
        ev=ev,
        group=group,
        include=include,
        patch_results=results
        if results is not None
        else [
            {"row": 1, "col": 1, "rgb": [0.1, 0.2, 0.3], "pixel_count": 10},
            {"row": 1, "col": 2, "rgb": [0.4, 0.5, 0.6], "pixel_count": 10},
        ],
    )


def test_entries_in_list_order_patches_row_major():
    text = combined_csv([_entry("plus3", 3.0), _entry("minus3", -3.0)])
    lines = text.strip().split("\n")
    assert lines[0] == "label,ev,group,patch_row,patch_col,R,G,B"
    assert [ln.split(",")[0] for ln in lines[1:]] == ["plus3", "plus3", "minus3", "minus3"]
    assert [ln.split(",")[4] for ln in lines[1:]] == ["1", "2", "1", "2"]


def test_unchecked_and_unprocessed_entries_skipped():
    entries = [
        _entry("in", 0.0),
        _entry("unchecked", 1.0, include=False),
        _entry("unprocessed", 2.0, results=[]),
    ]
    text = combined_csv(entries)
    assert "unchecked" not in text
    assert "unprocessed" not in text
    assert exportable_count(entries) == (1, 1)  # one exported, one skipped


def test_full_float_precision_survives():
    value = 0.123456789012345678
    entries = [_entry("x", None, results=[{"row": 1, "col": 1, "rgb": [value, 1.5, -0.25]}])]
    text = combined_csv(entries)
    row = text.strip().split("\n")[1].split(",")
    assert float(row[5]) == value  # repr round-trips exactly
    assert row[6] == "1.5" and row[7] == "-0.25"
    assert row[1] == ""  # ev=None -> empty field


def test_labels_with_commas_are_quoted():
    entries = [_entry("800T, tungsten", 0.0, group='hue "swing" A')]
    text = combined_csv(entries)
    assert '"800T, tungsten"' in text
    assert '"hue ""swing"" A"' in text
