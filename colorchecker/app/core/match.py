"""Color matching: optional 3x3 matrix stage + hierarchical RBF.

The RBF core is vendored and reworked from the camera-match fork
(MIT, Ethan Ou / M5era) — a pure numpy/scipy port of ALGLIB's
hierarchical RBF. Changes vs the fork:

- vectorized layer assembly and evaluation (~30x faster)
- the model is evaluated DIRECTLY (no baked-LUT clipping): inputs
  outside [0, 1] — emissive samples, scene-linear data — extrapolate
  smoothly instead of hitting a cube wall
- input validation: length mismatch raises, NaN pairs are dropped and
  counted
- error metrics are plain working-space RGB distances, no sRGB
  assumptions
- optional 3x3 matrix pre-fit (least squares, no offset, exposure
  invariant); the RBF then only fits the residual
- global strength: output = input + strength * (match(input) - input)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

from app.core.lut import CubeLUT, apply_lut, lattice_points

# Compactly-supported Gaussian-like basis, phi(u) with u = distance/radius.
# Constants reproduce ALGLIB's V2 basis to < 6e-5 over its support.
_RBF_A = 0.93750975
_RBF_D = 2.60513816
_RBF_P = 1.17659512


def _rbf_phi(u: np.ndarray) -> np.ndarray:
    return np.exp(-_RBF_A * u**2) * np.clip(1.0 - (u / _RBF_D) ** 2, 0.0, None) ** _RBF_P


def _thin(points: np.ndarray, h: float) -> np.ndarray:
    """Greedy poisson-disk decimation: maximal subset with pairwise
    distance >= h (ALGLIB's per-layer center coarsening)."""
    if h <= 0 or points.shape[0] == 0:
        return np.arange(points.shape[0])
    tree = cKDTree(points)
    kept_mask = np.zeros(points.shape[0], dtype=bool)
    removed = np.zeros(points.shape[0], dtype=bool)
    kept = []
    for i in range(points.shape[0]):
        if removed[i]:
            continue
        kept.append(i)
        kept_mask[i] = True
        for j in tree.query_ball_point(points[i], h):
            if not kept_mask[j]:
                removed[j] = True
    return np.asarray(kept, dtype=int)


def _layer_matrix(X: np.ndarray, centers: np.ndarray, radius: float) -> csr_matrix:
    """Sparse (len(X), len(centers)) basis matrix, fully vectorized."""
    n_x, n_c = X.shape[0], centers.shape[0]
    if n_c == 0:
        return csr_matrix((n_x, 0))
    cutoff = _RBF_D * radius
    pairs = cKDTree(X).query_ball_tree(cKDTree(centers), r=cutoff)
    lengths = np.fromiter((len(p) for p in pairs), dtype=int, count=n_x)
    if lengths.sum() == 0:
        return csr_matrix((n_x, n_c))
    rows = np.repeat(np.arange(n_x), lengths)
    cols = np.concatenate([np.asarray(p, dtype=int) for p in pairs if p])
    d = np.linalg.norm(X[rows] - centers[cols], axis=1) / radius
    return csr_matrix((_rbf_phi(d), (rows, cols)), shape=(n_x, n_c))


class HierarchicalRBF:
    """Affine term + multi-scale residual layers with radius halving."""

    def __init__(self, radius: float = 5.0, layers: int = 10, smoothing: float = 0.001):
        self.radius = radius
        self.layers = layers
        self.smoothing = smoothing

    def fit(self, source: np.ndarray, target: np.ndarray) -> None:
        X = np.asarray(source, dtype=np.float64)
        Y = np.asarray(target, dtype=np.float64)

        P = np.hstack([X, np.ones((X.shape[0], 1))])
        coef, *_ = np.linalg.lstsq(P, Y, rcond=None)
        self._coef = coef[:-1].T
        self._bias = coef[-1]
        residual = Y - P @ coef

        self._source = X
        self._weights, self._radii, self._center_idx = [], [], []
        radius = self.radius
        damp = self.smoothing * np.sqrt(X.shape[0])
        for _ in range(self.layers):
            idx = _thin(X, 0.1 * radius)
            centers = X[idx]
            A = _layer_matrix(X, centers, radius)
            w = np.empty((centers.shape[0], Y.shape[1]))
            for c in range(Y.shape[1]):
                w[:, c] = lsqr(A, residual[:, c], damp=damp, atol=1e-10, btol=1e-10)[0]
            residual = residual - A @ w
            self._weights.append(w)
            self._radii.append(radius)
            self._center_idx.append(idx)
            radius *= 0.5

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        X = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
        out = X @ self._coef.T + self._bias
        for w, r, idx in zip(self._weights, self._radii, self._center_idx):
            out = out + _layer_matrix(X, self._source[idx], r) @ w
        return out.reshape(np.asarray(rgb).shape)


@dataclass
class MatchModel:
    """Optional 3x3 matrix stage followed by an RBF on the residual."""

    matrix: np.ndarray | None  # (3, 3) or None
    rbf: HierarchicalRBF | None  # None when layers == 0
    strength: float = 1.0

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        x = np.asarray(rgb, dtype=np.float64)
        out = x
        if self.matrix is not None:
            out = out @ self.matrix.T
        if self.rbf is not None:
            out = self.rbf(out)
        if self.strength != 1.0:
            out = x + self.strength * (out - x)
        return out


@dataclass
class MatchResult:
    model: MatchModel
    pairs_used: int
    pairs_dropped: int  # NaN rows removed
    error_before: float  # mean RGB distance source vs target
    error_matrix: float | None  # after the matrix stage only (if used)
    error_after: float  # after the full model
    error_after_max: float
    per_patch_error: np.ndarray  # (N,) distances after the full model
    # Sandwich fit only: patches whose target the fixed output transform
    # cannot reach or resolve (clipped plateaus) — dropped from the fit.
    pairs_unreachable: int = 0
    display_referred: bool = False  # True -> error metrics are through the DRT


def invert_lut_at(
    lut: CubeLUT, targets: np.ndarray,
    tol: float = 0.004, min_slope: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each target value, find the LUT input that produces it.

    Returns (inputs, reachable_mask, residuals). A target is unreachable
    when no input gets within `tol` of it, or when the LUT is locally
    flat there (clipped plateau: the preimage is ambiguous and carries
    no information — min singular value of the local Jacobian < min_slope).
    """
    targets = np.asarray(targets, dtype=np.float64)
    grid_in, grid_out = lattice_points(lut, resolution=17)
    span = lut.domain_max - lut.domain_min
    grid_in = lut.domain_min + grid_in * span

    inverted = np.full_like(targets, np.nan)
    residuals = np.full(len(targets), np.inf)
    reachable = np.zeros(len(targets), dtype=bool)
    eps = 1e-3 * max(float(span.max()), 1e-6)

    for i, y in enumerate(targets):
        if np.isnan(y).any():
            continue
        start = grid_in[int(np.argmin(((grid_out - y) ** 2).sum(axis=1)))]
        with np.errstate(all="ignore"):  # plateau Jacobians degenerate harmlessly
            sol = least_squares(
                lambda v: apply_lut(lut, v[None, :])[0] - y,
                start,
                bounds=(lut.domain_min, lut.domain_max),
                xtol=1e-12, ftol=1e-14, gtol=None, method="trf",
            )
        inverted[i] = sol.x
        residuals[i] = float(np.linalg.norm(sol.fun))
        if residuals[i] > tol:
            continue
        # Local flatness check: finite-difference Jacobian at the solution.
        jac = np.empty((3, 3))
        for axis in range(3):
            hi = sol.x.copy(); hi[axis] = min(hi[axis] + eps, lut.domain_max[axis])
            lo = sol.x.copy(); lo[axis] = max(lo[axis] - eps, lut.domain_min[axis])
            step = hi[axis] - lo[axis]
            jac[:, axis] = (apply_lut(lut, hi[None, :])[0] - apply_lut(lut, lo[None, :])[0]) / step
        if np.linalg.svd(jac, compute_uv=False)[-1] >= min_slope:
            reachable[i] = True

    return inverted, reachable, residuals


def _mean_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, axis=1).mean())


def solve_match(
    source: np.ndarray,
    target: np.ndarray,
    use_matrix: bool = True,
    layers: int = 10,
    smoothing: float = 0.001,
    radius: float = 5.0,
    strength: float = 1.0,
    output_transform: CubeLUT | None = None,
) -> MatchResult:
    """Fit source -> target. With `output_transform` (a fixed DRT), the
    sandwich fit is solved instead: model such that DRT(model(source))
    matches the display-referred target; error metrics go through the DRT."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("source and target must be (N, 3) arrays")
    if source.shape[0] != target.shape[0]:
        raise ValueError(
            f"source has {source.shape[0]} rows but target has {target.shape[0]} — "
            "the two sets must pair up row for row"
        )

    valid = ~(np.isnan(source).any(axis=1) | np.isnan(target).any(axis=1))
    dropped = int((~valid).sum())
    source, target = source[valid], target[valid]

    unreachable = 0
    if output_transform is not None:
        # Solve "what must the model output so the DRT lands on the scan":
        # consult the DRT's preimage at each target patch. Targets the DRT
        # cannot reach or resolve (clipped plateaus) constrain nothing.
        fit_target, reachable, _ = invert_lut_at(output_transform, target)
        unreachable = int((~reachable).sum())
        source, target = source[reachable], target[reachable]
        fit_target = fit_target[reachable]
    else:
        fit_target = target

    if source.shape[0] < 4:
        raise ValueError("Need at least 4 valid patch pairs to fit")

    def display(values: np.ndarray) -> np.ndarray:
        return apply_lut(output_transform, values) if output_transform is not None else values

    error_before = _mean_dist(display(source), target)

    matrix = None
    error_matrix = None
    stage_source = source
    if use_matrix:
        # Plain least-squares 3x3, no offset: exposure invariant.
        matrix, *_ = np.linalg.lstsq(source, fit_target, rcond=None)
        matrix = matrix.T
        stage_source = source @ matrix.T
        error_matrix = _mean_dist(display(stage_source), target)

    rbf = None
    if layers > 0:
        rbf = HierarchicalRBF(radius=radius, layers=layers, smoothing=smoothing)
        rbf.fit(stage_source, fit_target)

    model = MatchModel(matrix=matrix, rbf=rbf, strength=strength)
    fitted = display(model(source))
    per_patch = np.linalg.norm(fitted - target, axis=1)
    return MatchResult(
        model=model,
        pairs_used=int(source.shape[0]),
        pairs_dropped=dropped,
        error_before=error_before,
        error_matrix=error_matrix,
        error_after=float(per_patch.mean()),
        error_after_max=float(per_patch.max()),
        per_patch_error=per_patch,
        pairs_unreachable=unreachable,
        display_referred=output_transform is not None,
    )


# ------------------------------------------------------------------ export

def write_cube(
    model: MatchModel,
    path: str | Path,
    size: int = 33,
    domain_min: float = 0.0,
    domain_max: float = 1.0,
    title: str = "Color Checker match",
) -> None:
    """Bake the model into a .cube 3D LUT (R varies fastest, per spec)."""
    if domain_max - domain_min < 0.05:
        raise ValueError(
            f"Domain max ({domain_max:g}) must be clearly above domain min "
            f"({domain_min:g}) — a zero-width domain collapses the whole LUT "
            "to a single constant color."
        )
    grid = np.linspace(domain_min, domain_max, size)
    b, g, r = np.meshgrid(grid, grid, grid, indexing="ij")
    pts = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1)
    values = model(pts)

    lines = [
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {size}",
        f"DOMAIN_MIN {domain_min:g} {domain_min:g} {domain_min:g}",
        f"DOMAIN_MAX {domain_max:g} {domain_max:g} {domain_max:g}",
    ]
    lines.extend(f"{v[0]:.10f} {v[1]:.10f} {v[2]:.10f}" for v in values)
    Path(path).write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------- data input

def load_patch_csv(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Read patch values from CSV: our rich export format (header with
    R,G,B columns) or any file whose last three columns are numbers.
    Returns (values (N, 3), row labels)."""
    text = Path(path).read_text()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("File is empty")

    values: list[list[float]] = []
    labels: list[str] = []

    header = [c.strip().lower() for c in lines[0].replace(";", ",").split(",")]
    if "r" in header and "g" in header and "b" in header:
        ir, ig, ib = header.index("r"), header.index("g"), header.index("b")
        ilabel = header.index("label") if "label" in header else None
        for n, line in enumerate(lines[1:], start=2):
            cols = _split_csv_line(line)
            try:
                values.append([float(cols[ir]), float(cols[ig]), float(cols[ib])])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Line {n}: cannot read R,G,B — {line!r}") from exc
            labels.append(cols[ilabel] if ilabel is not None else f"row {n - 1}")
    else:
        for n, line in enumerate(lines, start=1):
            cols = line.replace(",", " ").replace(";", " ").split()
            try:
                values.append([float(c) for c in cols[-3:]])
            except ValueError as exc:
                raise ValueError(f"Line {n}: last 3 columns must be numbers — {line!r}") from exc
            labels.append(f"row {n}")

    return np.asarray(values, dtype=np.float64), labels


def _split_csv_line(line: str) -> list[str]:
    """Minimal CSV splitting with double-quote support."""
    out, field, in_quotes = [], [], False
    i = 0
    while i < len(line):
        ch = line[i]
        if in_quotes:
            if ch == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    field.append('"')
                    i += 1
                else:
                    in_quotes = False
            else:
                field.append(ch)
        elif ch == '"':
            in_quotes = True
        elif ch == ",":
            out.append("".join(field))
            field = []
        else:
            field.append(ch)
        i += 1
    out.append("".join(field))
    return [f.strip() for f in out]


def session_patch_rows(store) -> tuple[np.ndarray, list[str]]:
    """Patch values from the current project, in export order (included
    entries with results; overlay order; patches row-major). NaN rows are
    kept — solve_match drops and counts them."""
    values: list[list[float]] = []
    labels: list[str] = []
    if store is None:
        return np.empty((0, 3)), labels
    for entry in store.images:
        if not entry.include or not entry.patch_results:
            continue
        for result in entry.patch_results:
            values.append([float(v) for v in result["rgb"]])
            labels.append(
                f"{entry.label} [{result.get('overlay', 'Overlay 1')}] "
                f"r{result['row']}c{result['col']}"
            )
    if not values:
        return np.empty((0, 3)), []
    return np.asarray(values, dtype=np.float64), labels
