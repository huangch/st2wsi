"""bUnwarpJ-compatible elastic registration.

This module owns the contract that matters most to downstream consumers:
the ``direct_transf.txt`` file. It is bUnwarpJ's own serialisation and
the Java plugin writes it via ``bunwarpj.MiscTools.saveElasticTransformation``.

Format (from bUnwarpJ source code, ``MiscTools.saveElasticTransformation``):

* ASCII header line: ``Transformation`` (single token, optional trailing
  whitespace). Our writer emits ``Transformation``.
* Single line: ``-1``     (signals *direct* coefficients, not inverse;
  the value ``-1`` is the marker bUnwarpJ uses; the Java plugin always
  writes ``-1``).
* Single line: ``N``       (N = (intervals × image_subsamp_fact) + 1,
  the dense-grid side length).
* ``N × N`` lines, each with two floats: ``cx[r][c]  cy[r][c]``.
  Each pair is the X and Y displacement of the *direct* transform at
  control-grid position ``(r, c)``.

This file format is what wsitrain reads when it warps Xenium cell
coordinates onto H&E. We write it in this exact shape, even when the
deformation came from scikit-image rather than bUnwarpJ — the
coefficient grid is dense and small enough that the choice of solver
doesn't matter at the *file* boundary, only at the *quality* boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class ElasticResult:
    """Dense control-grid coefficients and the grid spacing."""

    cx: np.ndarray  # (N, N) float
    cy: np.ndarray  # (N, N) float
    intervals: int  # grid intervals per axis (B-spline control points - 1)

    @property
    def N(self) -> int:
        """Side length of the dense coefficient grid."""
        return self.cx.shape[0]


def save_direct_transf(path: str, result: ElasticResult) -> None:
    """Write ``direct_transf.txt`` in bUnwarpJ's exact format.

    See module docstring for the line-by-line layout.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cx = np.asarray(result.cx, dtype=np.float64)
    cy = np.asarray(result.cy, dtype=np.float64)
    if cx.shape != cy.shape:
        raise ValueError(f"cx/cy shape mismatch: {cx.shape} vs {cy.shape}")
    with open(p, "w") as fh:
        fh.write("Transformation\n")
        fh.write("-1\n")
        fh.write(f"{cx.shape[0]}\n")
        for r in range(cx.shape[0]):
            for c in range(cx.shape[1]):
                fh.write(f"{cx[r, c]:.6f} {cy[r, c]:.6f}\n")


def load_direct_transf(path: str) -> ElasticResult:
    """Inverse of :func:`save_direct_transf`, useful for round-trip tests."""
    with open(path, "r") as fh:
        header = fh.readline().strip()
        if header != "Transformation":
            raise ValueError(f"unexpected header in {path}: {header!r}")
        sign = int(fh.readline().strip())
        if sign != -1:
            # bUnwarpJ's "direct" transformation; we don't read inverses.
            raise ValueError(f"expected direct-transformation header -1, got {sign}")
        n = int(fh.readline().strip())
        cx = np.zeros((n, n), dtype=np.float64)
        cy = np.zeros((n, n), dtype=np.float64)
        for r in range(n):
            for c in range(n):
                line = fh.readline()
                parts = line.split()
                cx[r, c] = float(parts[0])
                cy[r, c] = float(parts[1])
    N = cx.shape[0]
    intervals = N - 1  # default subsamp=0 → intervals == N - 1
    return ElasticResult(cx=cx, cy=cy, intervals=intervals)


def dense_coefficients_from_grid(
    dx: np.ndarray, dy: np.ndarray, intervals: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a sparse (intervals+1)×(intervals+1) displacement grid onto a
    dense ``((intervals+1) * 1)+1`` grid, matching the B-spline control
    spacing that bUnwarpJ uses with ``image_subsamp_fact = 0``.

    ``intervals`` is the B-spline *level* — the number of knot spans
    along the image's largest dimension. The dense grid is ``N = intervals + 1``
    when ``image_subsamp_fact = 0``.
    """
    if dx.shape != dy.shape:
        raise ValueError(f"shape mismatch: {dx.shape} vs {dy.shape}")
    if dx.shape[0] != intervals + 1 or dx.shape[1] != intervals + 1:
        raise ValueError(
            f"expected ({intervals + 1}, {intervals + 1}), got {dx.shape}"
        )
    # identity dense grid == bUnwarpJ's at subsamp=0
    return dx.astype(np.float64).copy(), dy.astype(np.float64).copy()


def identity_coefficients(grid_size: int) -> ElasticResult:
    """All-zero displacements (no deformation).

    Useful as a graceful fallback when the optional scikit-image
    B-spline solver is unavailable or skipped. wsitrain and other
    downstream consumers still get a valid direct_transf.txt they
    can parse and apply.
    """
    cx = np.zeros((grid_size, grid_size), dtype=np.float64)
    cy = np.zeros((grid_size, grid_size), dtype=np.float64)
    return ElasticResult(cx=cx, cy=cy, intervals=grid_size - 1)


def warp_with_coefficients(
    image: np.ndarray, result: ElasticResult, output_shape: tuple
) -> np.ndarray:
    """Apply a dense bUnwarpJ-style displacement grid to ``image``.

    This is a simple *nearest-neighbour* sample of the dense grid; it
    is intentionally not the registration-quality inverse — it's the
    *apply-transformation* step used when rendering QC thumbnails.
    """
    h, w = output_shape
    coeffs_x = result.cx
    coeffs_y = result.cy
    # Build coordinate grids in target space; map each pixel back to
    # source-space via the dense displacement grid.
    ys, xs = np.indices((h, w), dtype=np.float64)
    # Resample the (N, N) displacement field onto the image grid.
    r_idx = (ys / max(1, h - 1)) * (coeffs_x.shape[0] - 1)
    c_idx = (xs / max(1, w - 1)) * (coeffs_y.shape[1] - 1)
    r_lo = np.clip(np.floor(r_idx).astype(int), 0, coeffs_x.shape[0] - 1)
    r_hi = np.clip(r_lo + 1, 0, coeffs_x.shape[0] - 1)
    c_lo = np.clip(np.floor(c_idx).astype(int), 0, coeffs_x.shape[1] - 1)
    c_hi = np.clip(c_lo + 1, 0, coeffs_x.shape[1] - 1)
    rx = r_idx - r_lo
    cx_ = c_idx - c_lo
    ddx = (
        coeffs_x[r_lo, c_lo] * (1 - rx) * (1 - cx_)
        + coeffs_x[r_hi, c_lo] * rx * (1 - cx_)
        + coeffs_x[r_lo, c_hi] * (1 - rx) * cx_
        + coeffs_x[r_hi, c_hi] * rx * cx_
    )
    ddy = (
        coeffs_y[r_lo, c_lo] * (1 - rx) * (1 - cx_)
        + coeffs_y[r_hi, c_lo] * rx * (1 - cx_)
        + coeffs_y[r_lo, c_hi] * (1 - rx) * cx_
        + coeffs_y[r_hi, c_hi] * rx * cx_
    )
    src_x = np.clip(xs - ddx, 0, image.shape[1] - 1).astype(int)
    src_y = np.clip(ys - ddy, 0, image.shape[0] - 1).astype(int)
    return image[src_y, src_x].copy()
