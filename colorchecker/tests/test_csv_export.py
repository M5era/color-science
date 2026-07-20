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
    assert lines[0] == "label,ev,group,overlay,kind,patch_row,patch_col,R,G,B"
    assert [ln.split(",")[0] for ln in lines[1:]] == ["plus3", "plus3", "minus3", "minus3"]
    assert [ln.split(",")[6] for ln in lines[1:]] == ["1", "2", "1", "2"]
    # Results without overlay/kind tags (pre-emissive projects) export with defaults.
    assert [ln.split(",")[3] for ln in lines[1:]] == ["Overlay 1"] * 4
    assert [ln.split(",")[4] for ln in lines[1:]] == ["reflective"] * 4


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
    assert float(row[7]) == value  # repr round-trips exactly
    assert row[8] == "1.5" and row[9] == "-0.25"
    assert row[1] == ""  # ev=None -> empty field


def test_emissive_rows_appear_in_stream_tagged():
    chart_rows = [
        {"row": 1, "col": 1, "rgb": [0.1, 0.2, 0.3], "overlay": "Overlay 1", "kind": "reflective"},
        {"row": 1, "col": 2, "rgb": [0.2, 0.3, 0.4], "overlay": "Overlay 1", "kind": "reflective"},
    ]
    light_row = [
        {"row": 1, "col": 1, "rgb": [1.9, 0.7, 0.6], "overlay": "Overlay 2", "kind": "emissive"},
    ]
    entries = [
        _entry("with_light", 0.0, results=chart_rows + light_row),
        _entry("chart_only", 1.0, results=chart_rows),
    ]
    lines = combined_csv(entries).strip().split("\n")[1:]
    kinds = [ln.split(",")[4] for ln in lines]
    assert kinds == ["reflective", "reflective", "emissive", "reflective", "reflective"]
    emissive_line = lines[2].split(",")
    assert emissive_line[3] == "Overlay 2"
    assert emissive_line[7] == "1.9"  # emissive value >1 exported raw


def test_labels_with_commas_are_quoted():
    entries = [_entry("800T, tungsten", 0.0, group='hue "swing" A')]
    text = combined_csv(entries)
    assert '"800T, tungsten"' in text
    assert '"hue ""swing"" A"' in text
