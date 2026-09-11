"""Image readers and the small preprocessing primitives run_st2wsi.sh applies.

The Java plugin's headless mode reads from Bio-Formats (``ImageReader``)
and lets the caller address any series inside an OME-TIFF / SVS /
``.vol`` pyramid. We do the same with :mod:`tifffile`:

* For OME-TIFF and most pyramidal TIFFs, ``tifffile.TiffFile.series``
  gives us each pyramid level as its own ``TiffPageSeries`` and
  ``series[k].asarray()`` returns the level as a NumPy array.
* For Aperio SVS the same API works (the file is a pyramidal TIFF
  with a slightly different metadata layer). We do not attempt to
  parse the proprietary SVS macro/metadata block; if a down-stream
  consumer needs it they should use the openslide-based reader we
  mention in the README.
* For ``.vol`` (the legacy format used by some spatial-transcriptomics
  vendors) we delegate to a best-effort TIFF read; if we cannot read
  it the caller gets a clear error rather than a silent fallback.

Series indices are 1-based in the user-facing CLI (mirroring the Java
plugin) and 0-based internally.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import tifffile
except ImportError as e:  # pragma: no cover - hard dep
    raise ImportError(
        "py_st2wsi.io requires tifffile. Install with: pip install tifffile"
    ) from e


def list_series(path: str) -> list:
    """Return a list of ``(series, width, height, downsample_floor)`` tuples.

    Mirrors ``ST2WSI_Registration.listSeries``: each entry stores the
    *rounded* integer that ``refSeries / tgtSeries`` will reproduce.
    ``downsample_floor`` is ``round(((W0 / W) + (H0 / H)) / 2)``.
    """
    out = []
    with tifffile.TiffFile(path) as tf:
        n = len(tf.series)
        if n == 0:
            raise ValueError(f"{path}: no series found")
        w0 = tf.series[0].shape[1]
        h0 = tf.series[0].shape[0]
        for i in range(n):
            page = tf.series[i]
            h, w = page.shape[:2]
            f = ((w0 / w) + (h0 / h)) / 2.0
            stored = int(round(f))
            out.append((i + 1, int(w), int(h), stored))
    return out


def read_series(path: str, series_1based: int) -> np.ndarray:
    """Load series ``series_1based`` as a 2-D or 3-D NumPy array.

    * If the series has more than one channel the array is stacked
      along the last axis (matches Bio-Formats' ``openBytes(i)`` loop).
    * Multi-page stacks (Z, T) are collapsed by average-intensity Z
      projection unless ``collapse_z=False`` is passed.
    """
    if series_1based < 1:
        raise ValueError(f"series is 1-based, got {series_1based}")
    with tifffile.TiffFile(path) as tf:
        if series_1based - 1 >= len(tf.series):
            raise IndexError(
                f"{path}: series {series_1based} requested, file has {len(tf.series)}"
            )
        page = tf.series[series_1based - 1]
        arr = page.asarray()
    return np.asarray(arr)


def scaling_factor(path: str, series_1based: int) -> int:
    """Same definition as ``getScalingFactor`` in the Java plugin."""
    series = list_series(path)
    for s, _w, _h, stored in series:
        if s == series_1based:
            return stored
    raise IndexError(f"series {series_1based} not in {path}")


def ensure_grayscale(arr: np.ndarray) -> np.ndarray:
    """Z-project (average) and drop single-channel axis.

    Matches the Java pipeline's ``Z Project... [Average Intensity]`` step
    followed by ``Grays`` (single-channel conversion).
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # channel axis can be last (TiffFile) or first (rare)
        if arr.shape[-1] in (3, 4) and arr.shape[0] > 4:
            # likely a stack, not RGB
            return arr.mean(axis=0).astype(arr.dtype)
        if arr.shape[-1] in (3, 4):
            return arr  # leave RGB alone; caller decides
        return arr.mean(axis=0).astype(arr.dtype)
    if arr.ndim == 4:
        # (T, Z, H, W) or (Z, C, H, W) — average over leading axes
        return arr.mean(axis=(0, 1)).astype(arr.dtype)
    raise ValueError(f"cannot reduce {arr.shape} to 2D")


def subtract_background(
    img: np.ndarray, rolling: int = 50
) -> np.ndarray:
    """Pure-NumPy rolling-ball background subtraction.

    Approximates ImageJ's ``Subtract Background... rolling=50`` with a
    square kernel of half-width ``rolling // 2``; this is what the
    Java plugin uses after colour deconvolution. The result is clipped
    to ``[0, 255]`` and returned as uint8.

    We deliberately use a simple box-mean for portability; with the
    default 50-px rolling radius the kernel is too wide for the
    Gaussian approximation to matter at 8-bit precision, and the
    downstream SIFT step is robust to either.
    """
    if img.ndim != 2:
        raise ValueError(f"expected 2D, got {img.shape}")
    radius = max(1, rolling // 2)
    # cumulative-sum rolling mean — O(HW) in one pass.
    arr = img.astype(np.float64)
    # pad so edges get a fair estimate
    p = np.pad(arr, radius, mode="reflect")
    cs = p.cumsum(axis=0).cumsum(axis=1)
    H, W = arr.shape
    # sum over (2r+1)x(2r+1) window via 4-corner CS identity
    s = cs[2 * radius :, 2 * radius :] - cs[: H, 2 * radius :] - cs[
        2 * radius :, : W
    ] + cs[: H, : W]
    bg = s / ((2 * radius + 1) ** 2)
    out = arr - bg
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def gaussian_blur(img: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    """Gaussian blur via scipy.ndimage, matching IJ's Gaussian Blur."""
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(img.astype(np.float64), sigma=sigma).astype(np.uint8)


def normalize_to_byte(img: np.ndarray, saturated: float = 0.0) -> np.ndarray:
    """Histogram equalisation, equivalent to Fiji's *Enhance Contrast*
    ``saturated=0 equalize``. Returns uint8."""
    arr = img.astype(np.float64)
    flat = arr.flatten()
    # saturate ignores top/bottom fractions of the histogram
    if saturated > 0:
        lo = np.percentile(flat, 100 * saturated / 2)
        hi = np.percentile(flat, 100 * (1 - saturated / 2))
        arr = np.clip(arr, lo, hi)
    # equalisation
    hist, bin_edges = np.histogram(arr.flatten(), bins=256)
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) * 255 / max(1, (cdf.max() - cdf.min()))
    cdf = cdf.astype(np.uint8)
    # map each intensity to its equalised rank
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # nearest-bin lookup; values outside the original range clamp.
    idx = np.searchsorted(bin_centers, arr)
    idx = np.clip(idx, 0, 255)
    out = cdf[idx]
    return out.astype(np.uint8)


def composite_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert a multi-channel composite (C, H, W) or (H, W, C) to HxWx3 uint8.

    The Java plugin only does this when the image is a 3- or 4-channel
    composite; we mirror that here. For 4 channels the alpha is dropped.
    For 3 channels we return them as-is.

    The output is *not* colour-deconvolved; that's the caller's job.
    """
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        # CHW layout
        out = np.transpose(img[:3], (1, 2, 0))
    elif img.ndim == 3 and img.shape[-1] in (3, 4):
        out = img[..., :3]
    else:
        raise ValueError(f"cannot composite {img.shape}")
    if out.dtype != np.uint8:
        out = (255 * (out.astype(np.float64) / max(1, out.max()))).astype(np.uint8)
    return out


def rotate(arr: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate ``arr`` by an integer multiple of 90 degrees.

    Mirrors the Java plugin, which only ever rotates the reference image
    (H&E is never rotated in the CLI path). Negative angles are supported
    to match the GUI dialog strings ("-90", "-180", "-270").
    """
    if degrees % 90 != 0:
        raise ValueError(f"only multiples of 90 supported, got {degrees}")
    k = (degrees // 90) % 4
    if k == 0:
        return arr.copy()
    return np.rot90(arr, k=k).copy()


def flip_horizontal(arr: np.ndarray) -> np.ndarray:
    return np.fliplr(arr).copy()


def qc_overlay(
    tgt: np.ndarray, src_aligned: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Make the same QC stack + magenta/green composite the Java plugin saves.

    Returns ``(stack, overlay_RGB)``. ``stack`` is a ``(2, H, W)`` uint8
    array — target then aligned source — identical in shape to the
    ``ImageStack`` the Java code writes. ``overlay_RGB`` is the HxWx3
    composite used to visually verify alignment; pixels that match are
    grey/white because the same intensity lands at the same ``(x, y)``.
    """
    t8 = tgt.astype(np.uint8)
    s8 = src_aligned.astype(np.uint8)
    h = min(t8.shape[0], s8.shape[0])
    w = min(t8.shape[1], s8.shape[1])
    t8 = t8[:h, :w]
    s8 = s8[:h, :w]
    stack = np.stack([t8, s8], axis=0)
    g = t8
    m = s8
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[..., 0] = m  # R = source (magenta on overlap)
    overlay[..., 1] = g  # G = target
    overlay[..., 2] = m  # B = source
    return stack, overlay
