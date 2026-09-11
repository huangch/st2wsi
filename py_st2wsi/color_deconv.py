"""Ruifrok & Johnston colour-deconvolution (numpy port of the Java code).

Reproduces :func:`deconvolveHEDHeadless` in
``ST2WSI_Registration.java``. The stain vectors default to the two
``colourdeconvolution.txt`` rows the plugin recognises (``H&E`` and
``H&E 2``); callers can supply extra rows by passing a custom path
or a custom vector matrix.

Two-stage normalisation matches the Java code exactly:

1. ``h`` and ``e`` vectors are L2-normalised.
2. If the third vector is all zero (the default in
   :file:`colourdeconvolution.txt` for two-stain recipes), it is set
   to ``h × e`` and then L2-normalised. This is the only place the
   3x3 matrix becomes invertible — necessary for the inverse-multiply
   below.
3. The 3x3 ``M = [h|e|d]`` matrix is inverted once.
4. For each pixel ``(R, G, B)`` we form OD-space concentrations
   ``c = M_inv @ (-log((rgb + 1) / 255))`` and pick the channel
   matching :data:`channel_names`.
5. The selected OD concentration ``c_ch`` is converted back to an
   8-bit intensity with ``255 * exp(-c_ch)`` and clipped to ``[0,255]``,
   matching the Java plugin's ``intensity = round(...)`` step.
"""
from __future__ import annotations

from pathlib import Path
import csv
from typing import Dict, Iterable, Optional

import numpy as np


CHANNEL_NAMES = ("Hematoxylon", "Eosin", "Residual")


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 0:
        return v / n
    return v


def _cross_fill(h: np.ndarray, e: np.ndarray, d: np.ndarray) -> np.ndarray:
    """If ``d`` is all-zero, replace with ``h × e`` (Java side)."""
    if d[0] == 0 and d[1] == 0 and d[2] == 0:
        return np.array(
            [
                h[1] * e[2] - h[2] * e[1],
                h[2] * e[0] - h[0] * e[2],
                h[0] * e[1] - h[1] * e[0],
            ],
            dtype=float,
        )
    return d


def load_stain_table(path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Parse ``colourdeconvolution.txt`` into ``{stain_name: shape(3,3)}``.

    The Java plugin only uses two rows (H&E and H&E 2); the rest are
    kept here for completeness so callers can pick a different stain
    set without editing the plugin.
    """
    if path is None:
        here = Path(__file__).resolve().parent.parent
        path = str(here / "colourdeconvolution.txt")
    table: Dict[str, np.ndarray] = {}
    with open(path, "r", newline="") as fh:
        reader = csv.reader(fh)
        for raw in reader:
            row = [c.strip() for c in raw if c.strip()]
            if not row or row[0].startswith("#"):
                continue
            try:
                vals = [float(x) for x in row[1:10]]
            except ValueError:
                continue
            if len(vals) < 9:
                continue
            mat = np.array(vals[:9], dtype=float).reshape(3, 3)
            table[row[0]] = mat
    return table


def stain_matrix(stain: str, table: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """Return the 3x3 row-major stain matrix for ``stain``.

    Default fallback matches the Java plugin's hard-coded behaviour
    if the stain name is not found: assume ``H&E``.
    """
    if table is None:
        table = load_stain_table()
    if stain not in table:
        stain = "H&E"
    return table[stain].astype(float).copy()


def deconvolve(
    rgb: np.ndarray,
    channel: str,
    stain: str = "H&E",
    eps: float = 1.0,
) -> np.ndarray:
    """Apply Ruifrok & Johnston colour deconvolution.

    Parameters
    ----------
    rgb : ``(H, W, 3)`` uint8 array.
    channel : one of "Hematoxylon", "Eosin", "Residual".
    stain : row name from ``colourdeconvolution.txt`` (default ``H&E``).
    eps : OD floor, matches the Java ``eps = 1.0`` constant.

    Returns
    -------
    ``(H, W)`` uint8 array of the requested channel.
    """
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)

    if channel not in CHANNEL_NAMES:
        raise ValueError(f"channel must be one of {CHANNEL_NAMES!r}, got {channel!r}")

    ch_idx = CHANNEL_NAMES.index(channel)
    mat = stain_matrix(stain)
    h_vec, e_vec, d_vec = mat[0].copy(), mat[1].copy(), mat[2].copy()
    h_vec = _l2(h_vec)
    e_vec = _l2(e_vec)
    d_vec = _l2(_cross_fill(h_vec, e_vec, d_vec))

    m = np.column_stack([h_vec, e_vec, d_vec])  # columns: H, E, D
    inv_m = np.linalg.inv(m)

    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)

    od_r = -np.log((r + eps) / 255.0)
    od_g = -np.log((g + eps) / 255.0)
    od_b = -np.log((b + eps) / 255.0)

    od = np.stack([od_r, od_g, od_b], axis=-1)  # (H, W, 3)
    concentrations = od @ inv_m.T  # (H, W, 3) — channels last axis
    c = concentrations[..., ch_idx]
    c = np.clip(c, 0, None)

    intensity = np.round(255.0 * np.exp(-c)).astype(np.uint8)
    return intensity
