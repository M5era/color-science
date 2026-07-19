"""Project store: the document all tabs read and write.

The project JSON is the database; CSV exports are views of it. Schema is
versioned and each tab keeps its state under its own top-level key, so
files stay loadable as Matching / LUT Inspector appear later — unknown
keys are preserved on load and written back on save, never a parse error.

Image entries carry the multi-exposure session metadata:
  label   free-form, defaults to filename ("800T +3EV")
  ev      optional float, auto-parsed from filename, editable; drives Sort by EV
  group   free-form tag for lighting setups ("3200K", "hue swing A")
  include whether the entry participates in a combined export
Entry order in `images` IS the export order.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class ImageEntry:
    source_path: str
    label: str
    ev: float | None = None
    group: str = ""
    include: bool = True
    overlays: list[dict[str, Any]] = field(default_factory=list)
    patch_results: list[dict[str, Any]] = field(default_factory=list)
    # Overlay names switched off for THIS frame (e.g. the light-source
    # square on frames without the light). Overlays are shared across
    # frames; this records the per-frame exceptions.
    disabled_overlays: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "label": self.label,
            "ev": self.ev,
            "group": self.group,
            "include": self.include,
            "overlays": self.overlays,
            "patch_results": self.patch_results,
            "disabled_overlays": self.disabled_overlays,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageEntry":
        return cls(
            source_path=data["source_path"],
            label=data.get("label") or Path(data["source_path"]).name,
            ev=data.get("ev"),
            group=data.get("group", ""),
            include=data.get("include", True),
            overlays=data.get("overlays", []),
            patch_results=data.get("patch_results", []),
            disabled_overlays=data.get("disabled_overlays", []),
        )


@dataclass
class ProjectStore:
    images: list[ImageEntry] = field(default_factory=list)
    # State belonging to tabs this build doesn't implement yet (and any
    # future keys) survives a load/save round-trip untouched.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.extra)
        data["schema_version"] = SCHEMA_VERSION
        data["processing"] = {"images": [entry.to_dict() for entry in self.images]}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectStore":
        version = data.get("schema_version", 1)
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"Project was saved by a newer app version (schema {version})"
            )
        processing = data.get("processing", {})
        images = [ImageEntry.from_dict(d) for d in processing.get("images", [])]
        extra = {k: v for k, v in data.items() if k not in ("schema_version", "processing")}
        return cls(images=images, extra=extra)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ProjectStore":
        return cls.from_dict(json.loads(Path(path).read_text()))
