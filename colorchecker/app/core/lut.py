"""3D LUT (.cube) loading, application, and inspection sampling.

Applying a LUT here is for INSPECTION ONLY — previews, curves, lattice
views. The measurement pipeline never routes pixel data through this.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CubeLUT:
    title: str
    size: int
    domain_min: np.ndarray  # (3,)
    domain_max: np.ndarray  # (3,)
    table: np.ndarray  # (size, size, size, 3), indexed [b, g, r]

    @property
    def name(self) -> str:
        return self.title or "untitled"


def parse_cube(path: str | Path) -> CubeLUT:
    """Read a 3D .cube file (R varies fastest, per the Resolve spec)."""
    title = ""
    size = None
    domain_min = np.zeros(3)
    domain_max = np.ones(3)
    values: list[list[float]] = []

    for n, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper.startswith("TITLE"):
            title = line[5:].strip().strip('"')
        elif upper.startswith("LUT_3D_SIZE"):
            size = int(line.split()[1])
        elif upper.startswith("LUT_1D_SIZE"):
            raise ValueError("1D LUTs are not supported yet — load a 3D .cube")
        elif upper.startswith("DOMAIN_MIN"):
            domain_min = np.array([float(v) for v in line.split()[1:4]])
        elif upper.startswith("DOMAIN_MAX"):
            domain_max = np.array([float(v) for v in line.split()[1:4]])
        elif line[0] in "+-.0123456789":
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Line {n}: expected 3 values, got {len(parts)}")
            values.append([float(v) for v in parts])

    if size is None:
        raise ValueError("Not a 3D cube file (no LUT_3D_SIZE)")
    if len(values) != size**3:
        raise ValueError(f"Expected {size ** 3} entries for size {size}, got {len(values)}")

    table = np.asarray(values, dtype=np.float64).reshape(size, size, size, 3)
    return CubeLUT(title=title, size=size, domain_min=domain_min,
                   domain_max=domain_max, table=table)


def apply_lut(lut: CubeLUT, rgb: np.ndarray) -> np.ndarray:
    """Trilinear LUT application to an (..., 3) array (inspection quality).

    Inputs outside the LUT domain are clamped to the domain edge —
    exactly what a LUT box in a grading app would do.
    """
    shape = rgb.shape
    x = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    # Normalize into lattice coordinates.
    span = lut.domain_max - lut.domain_min
    t = (x - lut.domain_min) / np.where(span == 0, 1, span)
    t = np.clip(t, 0.0, 1.0) * (lut.size - 1)

    i0 = np.floor(t).astype(int)
    i0 = np.clip(i0, 0, lut.size - 2)
    frac = t - i0
    i1 = i0 + 1

    def corner(ir, ig, ib):
        return lut.table[ib, ig, ir]

    r0, g0, b0 = i0[:, 0], i0[:, 1], i0[:, 2]
    r1, g1, b1 = i1[:, 0], i1[:, 1], i1[:, 2]
    fr, fg, fb = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]

    out = (
        corner(r0, g0, b0) * (1 - fr) * (1 - fg) * (1 - fb)
        + corner(r1, g0, b0) * fr * (1 - fg) * (1 - fb)
        + corner(r0, g1, b0) * (1 - fr) * fg * (1 - fb)
        + corner(r0, g0, b1) * (1 - fr) * (1 - fg) * fb
        + corner(r1, g1, b0) * fr * fg * (1 - fb)
        + corner(r1, g0, b1) * fr * (1 - fg) * fb
        + corner(r0, g1, b1) * (1 - fr) * fg * fb
        + corner(r1, g1, b1) * fr * fg * fb
    )
    return out.reshape(shape)


def neutral_curves(lut: CubeLUT, samples: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Sample the LUT along the neutral axis: returns (inputs (N,), outputs (N, 3))."""
    t = np.linspace(0.0, 1.0, samples)
    grays = lut.domain_min + t[:, None] * (lut.domain_max - lut.domain_min)
    return t, apply_lut(lut, grays)


def lattice_points(lut: CubeLUT, resolution: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """Subsampled lattice for the 3D view: (inputs (M, 3) in [0,1], outputs (M, 3))."""
    idx = np.linspace(0, lut.size - 1, min(resolution, lut.size)).round().astype(int)
    b, g, r = np.meshgrid(idx, idx, idx, indexing="ij")
    outputs = lut.table[b.ravel(), g.ravel(), r.ravel()]
    inputs = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1) / (lut.size - 1)
    return inputs, outputs


def reference_gradient(width: int = 900, height: int = 520) -> np.ndarray:
    """Default preview: hue sweep across X, white -> pure hue -> black down Y."""
    hue = np.linspace(0.0, 1.0, width)
    h6 = hue * 6.0
    c = np.clip(np.stack([
        np.abs(h6 - 3.0) - 1.0,
        2.0 - np.abs(h6 - 2.0),
        2.0 - np.abs(h6 - 4.0),
    ], axis=1), 0.0, 1.0)  # (W, 3) pure hues

    y = np.linspace(0.0, 1.0, height)[:, None, None]
    top = 1.0 - 2.0 * np.minimum(y, 0.5)  # 1 -> 0 over the top half
    bottom = np.clip(2.0 - 2.0 * y, 0.0, 1.0)  # 1 until midway, then -> 0
    img = c[None, :, :] * bottom + top
    return np.clip(img, 0.0, 1.0).astype(np.float32)
