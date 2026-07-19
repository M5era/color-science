"""Project store round-trip: versioned schema, unknown keys survive."""

import json

import pytest

from app.core.project import SCHEMA_VERSION, ImageEntry, ProjectStore


def _sample_store() -> ProjectStore:
    return ProjectStore(
        images=[
            ImageEntry(
                source_path="/footage/800T_-3EV.tif",
                label="800T -3EV",
                ev=-3.0,
                group="3200K",
                overlays=[{"preset": "sg", "corners": [[0, 0], [1, 0], [1, 1], [0, 1]]}],
                patch_results=[{"row": 1, "col": 1, "rgb": [0.1, 0.2, 0.3]}],
            ),
            ImageEntry(source_path="/footage/800T_0EV.tif", label="800T 0EV", include=False),
        ]
    )


def test_roundtrip_preserves_everything(tmp_path):
    store = _sample_store()
    path = tmp_path / "session.ccproj.json"
    store.save(path)
    loaded = ProjectStore.load(path)

    assert loaded.to_dict() == store.to_dict()
    assert loaded.images[0].ev == -3.0
    assert loaded.images[0].group == "3200K"
    assert loaded.images[1].include is False
    assert loaded.images[0].patch_results[0]["rgb"] == [0.1, 0.2, 0.3]


def test_export_order_is_list_order(tmp_path):
    store = _sample_store()
    store.images.reverse()
    path = tmp_path / "p.json"
    store.save(path)
    loaded = ProjectStore.load(path)
    assert [e.label for e in loaded.images] == ["800T 0EV", "800T -3EV"]


def test_unknown_tab_keys_survive_roundtrip(tmp_path):
    # A file touched by a future version with Matching/RBF state present.
    data = _sample_store().to_dict()
    data["matching"] = {"rbf": {"kernel": "thin_plate", "pairs": [[0, 1]]}}
    data["lut_inspector"] = {"lut_path": "/luts/foo.cube"}
    path = tmp_path / "future.json"
    path.write_text(json.dumps(data))

    loaded = ProjectStore.load(path)
    saved = loaded.to_dict()
    assert saved["matching"] == data["matching"]
    assert saved["lut_inspector"] == data["lut_inspector"]


def test_newer_schema_rejected_cleanly():
    with pytest.raises(ValueError, match="newer app version"):
        ProjectStore.from_dict({"schema_version": SCHEMA_VERSION + 1})


def test_label_defaults_to_filename():
    entry = ImageEntry.from_dict({"source_path": "/x/0_EV_v1-800T_MatRem.tif"})
    assert entry.label == "0_EV_v1-800T_MatRem.tif"


def test_overlay_overrides_roundtrip(tmp_path):
    store = _sample_store()
    store.images[0].overlay_overrides = {
        "Overlay 1": [[10.0, 10.0], [110.0, 10.0], [110.0, 80.0], [10.0, 80.0]]
    }
    path = tmp_path / "p.json"
    store.save(path)
    loaded = ProjectStore.load(path)
    assert loaded.images[0].overlay_overrides["Overlay 1"][2] == [110.0, 80.0]
    assert loaded.images[1].overlay_overrides == {}
