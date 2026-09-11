"""SIFT-based affine alignment and the ``AffineTransform2D`` it produces.

The Java plugin uses Stephan Saalfeld's MPICBG SIFT implementation
(``mpicbg.ij.SIFT``). scikit-image 0.20+ ships an API-compatible SIFT
in :class:`skimage.feature.SIFT`, so we use it directly to keep the
runtime dependency footprint small (no OpenCV required).

The affine matrix returned by scikit-image uses the same row-major 6-tuple
convention as bUnwarpJ/MPICBG:

    [[a c e],
     [b d f]]

so that ``[x', y', 1] = [[a c e], [b d f], [0 0 1]] @ [x, y, 1]``.

This is what ``AffinityTransform.params`` returns. We pass it straight
through to :mod:`py_st2wsi.elastic` which then writes
``registration_params.json`` in the same row-major order the Java code
uses (see :mod:`py_st2wsi.io`'s note of the column-major write — that's
the consumer-facing JSON, not anything the SIFT solver produces here).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    from skimage.feature import SIFT, match_descriptors
    from skimage.transform import AffineTransform, warp
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "py_st2wsi.affine requires scikit-image >= 0.20. "
        "Install with: pip install scikit-image"
    ) from e


@dataclass
class AffineResult:
    """6-tuple affine, plus the matches that produced it.

    Matches
    -------
    matrix : shape ``(3, 3)`` with the last row ``[0, 0, 1]``.
    """

    matrix: np.ndarray  # (3, 3)
    n_inliers: int
    n_candidates: int
    inlier_ratio: float

    def to_six_tuple(self) -> Tuple[float, float, float, float, float, float]:
        """Return ``(m00, m10, m01, m11, m02, m12)``.

        Matches the row-major order the Java plugin's
        ``AffineModel2D.toArray`` writes into ``affineMatrix``.
        """
        m = self.matrix
        return (
            float(m[0, 0]),
            float(m[1, 0]),
            float(m[0, 1]),
            float(m[1, 1]),
            float(m[0, 2]),
            float(m[1, 2]),
        )


def _detect(sift: SIFT, img: np.ndarray):
    sift.detect_and_extract(img.astype(np.float32) / 255.0)
    return sift.keypoints, sift.descriptors


def sift_match(
    ref: np.ndarray,
    tgt: np.ndarray,
    rod: float = 0.92,
    max_epsilon: float = 25.0,
    min_inlier_ratio: float = 0.05,
    min_num_inliers: int = 7,
    sift_kwargs: dict | None = None,
) -> AffineResult:
    """Two-image SIFT alignment.

    Parameters mirror the Java plugin parameters of the same name.
    The defaults reproduce the Java plugin's defaults.

    Returns
    -------
    :class:`AffineResult` with a 3x3 ``matrix`` whose last row is ``[0, 0, 1]``.

    Notes
    -----
    ``min_inlier_ratio`` and ``min_num_inliers`` are taken from the Java
    plugin's ``filterRansac`` arguments; we don't expose RANSAC iterations
    or a seed because the result is statistical anyway and the Java plugin
    hard-codes 1000.
    """
    # Map our (Java-shaped) parameter names onto the active SIFT solver's
    # accepted kwargs. skimage 0.26 SIFT signature:
    #   sigma_min (= initial blur), n_octaves, n_scales, n_hist, n_ori
    kwargs = dict(sift_kwargs or {})
    sk_kwargs = {
        "sigma_min": float(kwargs.get("sigma_min", kwargs.get("sigma", 1.6))),
        "n_octaves": int(kwargs.get("n_octaves", kwargs.get("octaves", 8))),
        "n_scales": int(kwargs.get("n_scales", kwargs.get("scales", 3))),
        "n_hist": int(kwargs.get("n_hist", 4)),
        "n_ori": int(kwargs.get("n_ori", 8)),
    }
    sift_ref = SIFT(**sk_kwargs)
    sift_tgt = SIFT(**sk_kwargs)

    try:
        kp1, des1 = _detect(sift_ref, ref)
        kp2, des2 = _detect(sift_tgt, tgt)
    except IndexError as exc:
        # skimage SIFT raises "index 0 out of bounds for axis 0 with size 0"
        # when the image is too uniform for its scalespace (or its size
        # is below n_octaves). Surface a friendlier message.
        raise RuntimeError(
            f"SIFT scalespace is empty for one of the (post-deconvolution) "
            f"images — try a smaller --sift-steps or a different "
            f"--tgt-channel/--tgt-stain combination. Underlying error: {exc}"
        ) from exc

    if len(kp1) == 0 or len(kp2) == 0:
        raise RuntimeError("No SIFT features found in one or both images")

    matches = match_descriptors(des1, des2, cross_check=True, max_ratio=rod)
    if len(matches) < 4:
        raise RuntimeError(f"Insufficient SIFT matches: {len(matches)}")

    src = kp1[matches[:, 0]][:, ::-1]  # scikit-image: (row, col); we want (x, y)
    dst = kp2[matches[:, 1]][:, ::-1]

    # Robust affine estimation. We split the 2-D affine into two
    # independent 1-D linear regressions in (x, y) so we can use
    # sklearn's stock `RANSACRegressor` (which only handles 1-D
    # targets). The inlier masks are AND-combined so a match has to
    # agree with both regressions, which approximates MPICBG's
    # `AffineModel2D.filterRansac`. Falls back to LSQ+threshold if
    # sklearn isn't installed.
    try:
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        X = PolynomialFeatures(degree=1, include_bias=True).fit_transform(src)
        r_x = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=3,
            residual_threshold=max_epsilon,
            max_trials=1000,
            random_state=0,
        )
        r_y = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=3,
            residual_threshold=max_epsilon,
            max_trials=1000,
            random_state=0,
        )
        r_x.fit(X, dst[:, 0])
        r_y.fit(X, dst[:, 1])
        inliers = r_x.inlier_mask_ & r_y.inlier_mask_
    except ImportError:
        affine_full = AffineTransform()
        affine_full.estimate(src, dst)
        pred = affine_full(src)
        residuals = np.linalg.norm(pred - dst, axis=1)
        inliers = residuals < max_epsilon

    n_inliers = int(inliers.sum())
    n_candidates = len(matches)

    if n_inliers < min_num_inliers or (n_inliers / max(1, n_candidates)) < min_inlier_ratio:
        raise RuntimeError(
            f"Failed to find reliable transformation model: "
            f"{n_inliers}/{n_candidates} inliers"
        )

    affine_full = AffineTransform()
    affine_full.estimate(src[inliers], dst[inliers])

    # skimage's AffineTransform.params is already a (3, 3) homogeneous
    # matrix. Older versions exposed (2, 3), so handle both for safety.
    if affine_full.params.shape == (2, 3):
        matrix = np.vstack([affine_full.params, [0, 0, 1]])
    else:
        matrix = affine_full.params

    return AffineResult(
        matrix=matrix,
        n_inliers=n_inliers,
        n_candidates=n_candidates,
        inlier_ratio=n_inliers / max(1, n_candidates),
    )


def warp_affine(image: np.ndarray, matrix: np.ndarray, output_shape) -> np.ndarray:
    """Apply an affine matrix to ``image`` and return the size-of-``target`` result.

    ``output_shape`` is ``(H, W)`` — matching the Java plugin which
    resizes the warped reference to the *target* image's shape.

    ``matrix`` may be either (3, 3) (homogeneous) or (2, 3) (skimage-style);
    we normalise to the 3x3 form because ``skimage.transform.warp`` only
    accepts a 3x3 affine in this version of scikit-image.
    """
    if matrix.shape == (2, 3):
        m33 = np.vstack([matrix, [0, 0, 1]])
    elif matrix.shape == (3, 3):
        m33 = matrix
    else:
        raise ValueError(
            f"matrix must be (2, 3) or (3, 3), got {matrix.shape}"
        )

    tform = AffineTransform(matrix=m33)
    warped = warp(
        image.astype(np.float64) / 255.0,
        inverse_map=tform.inverse,
        output_shape=output_shape,
        preserve_range=True,
    )
    out = np.clip(warped * 255, 0, 255).astype(np.uint8)
    return out
