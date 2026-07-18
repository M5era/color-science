"""Image loading with a hard invariant: pixel values pass through untouched.

No ICC / ColorSync, no gamma, no clamping, no tone mapping. tifffile decodes
the raw sample values exactly as stored; everything downstream (sampling,
CSV export) reads only this buffer. Display previews are derived elsewhere
and are strictly one-way.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

SUPPORTED_SUFFIXES = {".tif", ".tiff"}

# EV markers, number-first: "+3EV", "-5EV", "0_EV", "3.5EV", "+2 EV"
_EV_PATTERN = re.compile(r"([+-]?\d+(?:[.,]\d+)?)\s*_?\s*EV", re.IGNORECASE)
# EV markers, EV-first: "EV+1", "EV-1", "EV1", "EV_2.5". Only whitespace
# and underscores may separate — a hyphen is always the minus sign.
_EV_PATTERN_PREFIX = re.compile(r"EV[\s_]*([+-]?\d+(?:[.,]\d+)?)", re.IGNORECASE)

# Lighting-setup markers: "5600K" / "2700k" (4 digits + K), "Hue120" / "hue_60".
# Lookarounds instead of \b: underscores are word chars, so \b fails on "_5600K_".
_KELVIN_PATTERN = re.compile(r"(?<![0-9A-Za-z])(\d{4})\s*K(?![0-9A-Za-z])", re.IGNORECASE)
_HUE_PATTERN = re.compile(r"hue[_\- ]?(\d{1,3})", re.IGNORECASE)


@dataclass
class LoadedImage:
    path: Path
    pixels: np.ndarray  # float32, shape (H, W, 3), raw file values

    @property
    def height(self) -> int:
        return self.pixels.shape[0]

    @property
    def width(self) -> int:
        return self.pixels.shape[1]


def load_image(path: str | Path) -> LoadedImage:
    """Read a TIFF into a float32 (H, W, 3) array of raw sample values.

    Integer TIFFs are normalized to [0, 1] by their type range (the only
    value mapping in the app, and it is exact and invertible). Float TIFFs
    are passed through bit-exact. An alpha channel, if present, is dropped.
    """
    path = Path(path)
    data = tifffile.imread(path)

    if data.ndim == 2:
        data = np.stack([data] * 3, axis=-1)
    if data.ndim != 3:
        raise ValueError(f"Unsupported TIFF layout {data.shape} in {path.name}")
    if data.shape[2] > 3:
        data = data[:, :, :3]
    if data.shape[2] != 3:
        raise ValueError(f"Expected 1, 3 or 4 channels, got {data.shape[2]} in {path.name}")

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        pixels = data.astype(np.float32) / float(info.max)
    elif data.dtype == np.float32:
        pixels = data
    elif np.issubdtype(data.dtype, np.floating):
        pixels = data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported sample type {data.dtype} in {path.name}")

    return LoadedImage(path=path, pixels=pixels)


def sibling_images(path: str | Path) -> list[Path]:
    """All supported images in the same folder, sorted by name (for prev/next)."""
    path = Path(path)
    return sorted(
        p for p in path.parent.iterdir()
        if p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.startswith(".")
    )


def neighbor_image(path: str | Path, step: int) -> Path | None:
    """The image `step` positions away in folder order, or None at the ends."""
    path = Path(path)
    siblings = sibling_images(path)
    try:
        idx = siblings.index(path)
    except ValueError:
        return None
    target = idx + step
    if 0 <= target < len(siblings):
        return siblings[target]
    return None


def parse_ev_from_filename(name: str) -> float | None:
    """Best-effort EV extraction: '0_EV_v1.tif' -> 0.0, 'EV+1.tif' -> 1.0."""
    match = _EV_PATTERN.search(name) or _EV_PATTERN_PREFIX.search(name)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def parse_group_from_filename(name: str) -> str:
    """Best-effort lighting-setup tag: '..._5600K_Hue120.tif' -> '5600K Hue120'."""
    parts = []
    kelvin = _KELVIN_PATTERN.search(name)
    if kelvin:
        parts.append(f"{kelvin.group(1)}K")
    hue = _HUE_PATTERN.search(name)
    if hue:
        parts.append(f"Hue{int(hue.group(1))}")
    return " ".join(parts)
