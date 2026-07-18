"""Overlay model: a patch grid pinned to an image by four corner points.

Geometry lives in a unit chart space (u, v in [0, 1]) that a homography
maps onto the image through the corners (TL, TR, BR, BL, image pixels).
Margins, patch size and offset are percentages, matching the reference
tool's sidebar fields.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Preset:
    name: str
    rows: int
    cols: int
    margin_x: float  # % of chart width consumed by the border on each side
    margin_y: float  # % of chart height
    patch_size: float  # % of a grid cell covered by the sample area


PRESETS: list[Preset] = [
    Preset("ColorChecker Digital SG (8 × 12)", rows=8, cols=12,
           margin_x=1.88, margin_y=2.82, patch_size=77.4),
    Preset("ColorChecker Classic (4 × 6)", rows=4, cols=6,
           margin_x=2.0, margin_y=3.0, patch_size=60.0),
    Preset("Custom", rows=8, cols=12, margin_x=2.0, margin_y=2.0, patch_size=75.0),
]


@dataclass
class Overlay:
    name: str = "Overlay 1"
    preset_name: str = PRESETS[0].name
    rows: int = 8
    cols: int = 12
    margin_x: float = 1.88
    margin_y: float = 2.82
    patch_size: float = 77.4
    patch_offset: float = 0.0
    # TL, TR, BR, BL in image pixel coordinates
    corners: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
    )

    @classmethod
    def from_preset(cls, preset: Preset, **kwargs) -> "Overlay":
        return cls(
            preset_name=preset.name,
            rows=preset.rows,
            cols=preset.cols,
            margin_x=preset.margin_x,
            margin_y=preset.margin_y,
            patch_size=preset.patch_size,
            **kwargs,
        )

    def apply_preset(self, preset: Preset) -> None:
        self.preset_name = preset.name
        self.rows = preset.rows
        self.cols = preset.cols
        self.margin_x = preset.margin_x
        self.margin_y = preset.margin_y
        self.patch_size = preset.patch_size

    # ------------------------------------------------- chart-space grid

    def patch_quads_unit(self) -> list[tuple[int, int, list[tuple[float, float]]]]:
        """Sample squares in unit chart space.

        Returns (row, col, [4 corner (u, v) points]) per patch, row-major,
        1-indexed to match the reference tool's table.
        """
        mx = self.margin_x / 100.0
        my = self.margin_y / 100.0
        cell_w = (1.0 - 2.0 * mx) / self.cols
        cell_h = (1.0 - 2.0 * my) / self.rows
        half_w = (self.patch_size / 100.0) * cell_w / 2.0
        half_h = (self.patch_size / 100.0) * cell_h / 2.0
        # patch_offset shifts the sample square within its cell, as a
        # percentage of the cell size (positive = right/down).
        off_x = (self.patch_offset / 100.0) * cell_w
        off_y = (self.patch_offset / 100.0) * cell_h

        quads = []
        for r in range(self.rows):
            cy = my + (r + 0.5) * cell_h + off_y
            for c in range(self.cols):
                cx = mx + (c + 0.5) * cell_w + off_x
                quads.append(
                    (
                        r + 1,
                        c + 1,
                        [
                            (cx - half_w, cy - half_h),
                            (cx + half_w, cy - half_h),
                            (cx + half_w, cy + half_h),
                            (cx - half_w, cy + half_h),
                        ],
                    )
                )
        return quads

    # ---------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "preset_name": self.preset_name,
            "rows": self.rows,
            "cols": self.cols,
            "margin_x": self.margin_x,
            "margin_y": self.margin_y,
            "patch_size": self.patch_size,
            "patch_offset": self.patch_offset,
            "corners": [list(map(float, pt)) for pt in self.corners],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Overlay":
        return cls(
            name=data.get("name", "Overlay 1"),
            preset_name=data.get("preset_name", PRESETS[0].name),
            rows=data.get("rows", 8),
            cols=data.get("cols", 12),
            margin_x=data.get("margin_x", 1.88),
            margin_y=data.get("margin_y", 2.82),
            patch_size=data.get("patch_size", 77.4),
            patch_offset=data.get("patch_offset", 0.0),
            corners=[list(map(float, pt)) for pt in data["corners"]],
        )
