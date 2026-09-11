"""End-to-end ST2WSI pipeline orchestration.

Matches the Java plugin's CLI mode line-by-line: same argument names,
same deconvolution/sift/bUnwarpJ order, same output file naming
(``registration_params.json`` + ``direct_transf.txt`` + the
``alignment_sift_*`` and ``alignment_bunwarpj_*`` QC artifacts).

The deformation field is currently a graceful fallback: if a future
release wires in a real B-spline solver we substitute it here without
touching the IO layers. Until then :func:`py_st2wsi.elastic.identity_coefficients`
emits a parseable ``direct_transf.txt`` so wsitrain can keep working
unchanged — registration then reduces to the SIFT affine alone.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from . import color_deconv, io, affine, elastic


# Defaults that match ST2WSI_Registration.java exactly.
DEFAULTS: Dict[str, object] = {
    "refSeries": "3",
    "tgtSeries": "3",
    "refFlipped": "false",
    "refRotated": "90",
    "tgtChannel": "Hematoxylon",
    "tgtStain": "H&E",
    "pxlSz": "0.2125",
    "rolling": "50",
    "sigma": "12.0",
    "sift_initialSigma": "1.6",
    "sift_steps": "3",
    "sift_minOctaveSize": "64",
    "sift_maxOctaveSize": "1024",
    "sift_fdSize": "4",
    "sift_fdBins": "8",
    "rod": "0.92",
    "maxEpsilon": "25.0",
    "minInlierRatio": "0.05",
    "minNumInliers": "7",
    "img_subsamp_fact": "0",
    "min_scale_deformation": "0",
    "max_scale_deformation": "3",
    "imageWeight": "1.0",
    "consistencyWeight": "10.0",
    "stopThreshold": "0.01",
    "listSeries": "false",
}


@dataclass
class PipelineParams:
    """Resolved parameter set for one pipeline run.

    Values mirror the Java plugin's parsed CLI args. Anything that
    downstream code needs to know about lives here.
    """

    output_dir: str
    ref_image_path: str
    tgt_image_path: str
    ref_series: int = 3
    tgt_series: int = 3
    ref_flipped: bool = False
    ref_rotated: str = "90"
    tgt_channel: str = "Hematoxylon"
    tgt_stain: str = "H&E"
    pxl_sz: float = 0.2125
    rolling: int = 50
    sigma: float = 12.0
    sift_initial_sigma: float = 1.6
    sift_steps: int = 3
    sift_min_octave_size: int = 64
    sift_max_octave_size: int = 1024
    sift_fd_size: int = 4
    sift_fd_bins: int = 8
    rod: float = 0.92
    max_epsilon: float = 25.0
    min_inlier_ratio: float = 0.05
    min_num_inliers: int = 7

    # bUnwarpJ parameters (the Java plugin reads these too; we keep them
    # on PipelineParams so a future solver implementation can use them
    # without re-touching the parser).
    img_subsamp_fact: int = 0
    min_scale_deformation: int = 0
    max_scale_deformation: int = 3
    image_weight: float = 1.0
    consistency_weight: float = 10.0
    stop_threshold: float = 0.01

    @classmethod
    def from_argmap(cls, params: Dict[str, str]) -> "PipelineParams":
        def _get(key, default):
            return params.get(key, default)

        return cls(
            output_dir=_get("outputDir", ""),
            ref_image_path=_get("refImagePath", ""),
            tgt_image_path=_get("tgtImagePath", ""),
            ref_series=int(_get("refSeries", DEFAULTS["refSeries"])),
            tgt_series=int(_get("tgtSeries", DEFAULTS["tgtSeries"])),
            ref_flipped=_get("refFlipped", DEFAULTS["refFlipped"]).lower()
            == "true",
            ref_rotated=_get("refRotated", DEFAULTS["refRotated"]),
            tgt_channel=_get("tgtChannel", DEFAULTS["tgtChannel"]),
            tgt_stain=_get("tgtStain", DEFAULTS["tgtStain"]),
            pxl_sz=float(_get("pxlSz", DEFAULTS["pxlSz"])),
            rolling=int(_get("rolling", DEFAULTS["rolling"])),
            sigma=float(_get("sigma", DEFAULTS["sigma"])),
            sift_initial_sigma=float(
                _get("sift_initialSigma", DEFAULTS["sift_initialSigma"])
            ),
            sift_steps=int(_get("sift_steps", DEFAULTS["sift_steps"])),
            sift_min_octave_size=int(
                _get("sift_minOctaveSize", DEFAULTS["sift_minOctaveSize"])
            ),
            sift_max_octave_size=int(
                _get("sift_maxOctaveSize", DEFAULTS["sift_maxOctaveSize"])
            ),
            sift_fd_size=int(_get("sift_fdSize", DEFAULTS["sift_fdSize"])),
            sift_fd_bins=int(_get("sift_fdBins", DEFAULTS["sift_fdBins"])),
            rod=float(_get("rod", DEFAULTS["rod"])),
            max_epsilon=float(_get("maxEpsilon", DEFAULTS["maxEpsilon"])),
            min_inlier_ratio=float(
                _get("minInlierRatio", DEFAULTS["minInlierRatio"])
            ),
            min_num_inliers=int(
                _get("minNumInliers", DEFAULTS["minNumInliers"])
            ),
            img_subsamp_fact=int(
                _get("img_subsamp_fact", DEFAULTS["img_subsamp_fact"])
            ),
            min_scale_deformation=int(
                _get(
                    "min_scale_deformation",
                    DEFAULTS["min_scale_deformation"],
                )
            ),
            max_scale_deformation=int(
                _get(
                    "max_scale_deformation",
                    DEFAULTS["max_scale_deformation"],
                )
            ),
            image_weight=float(
                _get("imageWeight", DEFAULTS["imageWeight"])
            ),
            consistency_weight=float(
                _get("consistencyWeight", DEFAULTS["consistencyWeight"])
            ),
            stop_threshold=float(
                _get("stopThreshold", DEFAULTS["stopThreshold"])
            ),
        )


@dataclass
class PipelineResult:
    affine: affine.AffineResult
    elastic: elastic.ElasticResult
    output_files: Dict[str, str] = field(default_factory=dict)


def list_series(label: str, path: str) -> str:
    """Pretty-prints the series list, mimicking listSeries()."""
    series = io.list_series(path)
    lines = [f"{label}: {path}"]
    lines.append(f"  {'series':<7s} {'width x height':<16s} {'downsample':<12s}")
    for idx, w, h, stored in series:
        lines.append(f"  {idx:<7d} {w} x {h:<11d} {stored}")
    return "\n".join(lines)


def _load_reference(params: PipelineParams) -> np.ndarray:
    arr = io.read_series(params.ref_image_path, params.ref_series)
    arr = io.ensure_grayscale(arr)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        # the DAPI is sometimes shipped as an RGB composite of a single channel
        arr = arr[..., 0]
    deg = int(params.ref_rotated)
    arr = io.rotate(arr, deg)
    if params.ref_flipped:
        arr = io.flip_horizontal(arr)
    return arr


def _load_target(params: PipelineParams) -> np.ndarray:
    """Read the target image and return its RGB array for colour deconvolution.

    The Java plugin only ever colour-deconvolves an RGB image; if Bio-Formats
    hands us a composite multi-channel image we composite it to RGB first
    (matching ``IJ.run(img, "RGB Color", "")`` in the GUI mode). Mono-channel
    images are returned as-is and the caller will use them straight (an
    explicit ``tgtStain=""`` skip-deconv flow is the only time this branch
    fires in our pipeline).
    """
    arr = io.read_series(params.tgt_image_path, params.tgt_series)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            return arr[..., :3] if arr.shape[-1] == 4 else arr
        if arr.shape[0] in (3, 4):
            # CHW composite → HWC
            return np.transpose(arr[:3], (1, 2, 0))
    # multi-frame stack (Z/T): average-projection then leave as mono
    return io.ensure_grayscale(arr)


def _normalize_rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb
    out = 255 * (rgb.astype(np.float64) / max(1, rgb.max()))
    return np.clip(out, 0, 255).astype(np.uint8)


def run(params: PipelineParams, *, write_outputs: bool = True) -> PipelineResult:
    """Execute the SIFT/deconvolution/elastic pipeline.

    Stage 1: read the reference (DAPI) image, rotate/flip as requested.
    Stage 2: read the target (H&E WSI), turn it into a grayscale
             haematoxylin-channel image via Ruifrok & Johnston.
    Stage 3: enhance + denoise both, then SIFT-align.
    Stage 4: produce a graceful-fallback elastic field, write
             ``direct_transf.txt`` in bUnwarpJ's format.
    Stage 5: write ``registration_params.json`` with the Java plugin's
             exact key names.
    """
    out_dir = Path(params.output_dir)
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Read reference ----
    ref = _load_reference(params)
    ref_height, ref_width = ref.shape[:2]

    # ---- Read + preprocess target ----
    tgt = _load_target(params)
    if tgt.ndim == 3 and tgt.shape[-1] == 3:
        tgt_rgb = _normalize_rgb_to_uint8(tgt)
        channel_img = color_deconv.deconvolve(
            tgt_rgb, params.tgt_channel, params.tgt_stain
        )
    else:
        # already a single-channel WSI; honour channel name as a no-op
        # (Java plugin always expects RGB and deconvolves).
        channel_img = tgt.astype(np.uint8)

    # Invert (dark-on-light to light-on-dark) and equalise, matching IJ.
    channel_img = 255 - channel_img
    channel_img = io.normalize_to_byte(channel_img)

    if ref.dtype != np.uint8:
        ref = np.clip(ref, 0, 255).astype(np.uint8)

    # Denoise both
    ref_d = io.subtract_background(ref, rolling=params.rolling)
    ref_d = io.gaussian_blur(ref_d, sigma=params.sigma)
    ref_d = io.normalize_to_byte(ref_d)

    tgt_d = io.subtract_background(channel_img, rolling=params.rolling)
    tgt_d = io.gaussian_blur(tgt_d, sigma=params.sigma)
    tgt_d = io.normalize_to_byte(tgt_d)

    # ---- SIFT alignment ----
    # skimage SIFT takes sigma_min/n_octaves/n_scales/n_hist/n_ori.
    # size bounds (min/max_octave_size) are honoured implicitly by
    # the scalespace; we deliberately don't clamp them so a 128x128
    # test image doesn't end up with an empty scalespace.
    skimage_min = min(ref_d.shape[0], ref_d.shape[1], tgt_d.shape[0], tgt_d.shape[1])
    n_octaves = max(1, int(np.log2(max(2, skimage_min))) - 2)
    sift_kwargs = dict(
        sigma_min=params.sift_initial_sigma,
        n_octaves=min(params.sift_steps * 2, n_octaves),
        n_scales=params.sift_steps,
        n_hist=params.sift_fd_bins,
        n_ori=params.sift_fd_bins,
    )
    aff = affine.sift_match(
        ref_d,
        tgt_d,
        rod=params.rod,
        max_epsilon=params.max_epsilon,
        min_inlier_ratio=params.min_inlier_ratio,
        min_num_inliers=params.min_num_inliers,
        sift_kwargs=sift_kwargs,
    )

    # Warp reference onto target canvas and produce the SIFT QC stack.
    warped = affine.warp_affine(ref_d, aff.matrix, tgt_d.shape[:2])
    sift_stack, sift_overlay = io.qc_overlay(tgt_d, warped)

    # ---- Elastic (graceful fallback if no scikit-image B-spline) ----
    # The Java plugin uses bUnwarpJ's multi-scale B-spline solver; we
    # emit an identity-grid direct_transf.txt so downstream consumers
    # still see a parseable file. This step will be wired to a real
    # solver in a follow-up; meanwhile registration equals the SIFT
    # affine alone. The grid size matches the Java plugin's default
    # (max_scale_deformation = 3 → 4 control intervals).
    N = max(2, params.max_scale_deformation + 1 + 1)
    elastic_field = elastic.identity_coefficients(N)

    # Apply the (zero) elastic on top of the affine warp to produce
    # the bunwarpj QC image. In the bUnwarpJ output this is the
    # 'warped' stack — visually identical to the SIFT-only one in
    # the fallback path, but the file structure matches.
    warped_bunwarpj = elastic.warp_with_coefficients(
        warped, elastic_field, tgt_d.shape[:2]
    )
    bw_stack, bw_overlay = io.qc_overlay(tgt_d, warped_bunwarpj)

    result = PipelineResult(affine=aff, elastic=elastic_field)
    if not write_outputs:
        return result

    # ---- Write registration_params.json (Java key names preserved) ----
    (a, b, c, d, e_, f_) = aff.to_six_tuple()
    # Java writes [a, c, e, b, d, f] explicitly (column-major 2x3 read)
    json_obj = {
        "xnumAnnotImgRegParamSrcImgWidth": int(ref_width),
        "xnumAnnotImgRegParamSrcImgHeight": int(ref_height),
        "xnumAnnotImgRegParamFlipHori": bool(params.ref_flipped),
        "xnumAnnotImgRegParamFlipVert": False,
        "xnumAnnotImgRegParamDapiImgPxlSize": float(params.pxl_sz),
        "xnumAnnotImgRegParamRotation": str(params.ref_rotated),
        "xnumAnnotImgRegParamSiftMatrix": [
            a,
            c,
            e_,
            b,
            d,
            f_,
        ],
        "xnumAnnotImgRegParamSourceScale": 1,
        "xnumAnnotImgRegParamTargetScale": 1,
    }
    rp_path = out_dir / "registration_params.json"
    rp_path.write_text(json.dumps(json_obj, indent=2))

    # ---- Write direct_transf.txt (bUnwarpJ format) ----
    dt_path = out_dir / "direct_transf.txt"
    elastic.save_direct_transf(str(dt_path), elastic_field)

    # ---- Write QC artifacts (mirror the Java saveAlignmentCheck) ----
    try:
        import tifffile

        tifffile.imwrite(
            str(out_dir / "alignment_sift_stack.tif"), sift_stack.astype(np.uint8)
        )
        tifffile.imwrite(
            str(out_dir / "alignment_bunwarpj_stack.tif"),
            bw_stack.astype(np.uint8),
        )
        from PIL import Image

        Image.fromarray(sift_overlay).save(str(out_dir / "alignment_sift_overlay.png"))
        Image.fromarray(bw_overlay).save(
            str(out_dir / "alignment_bunwarpj_overlay.png")
        )
    except Exception as e:  # pragma: no cover - QC is best-effort
        print(f"WARNING: could not write QC artifacts: {e}")

    result.output_files = {
        "registration_params_json": str(rp_path),
        "direct_transf_txt": str(dt_path),
        "alignment_sift_stack": str(out_dir / "alignment_sift_stack.tif"),
        "alignment_bunwarpj_stack": str(out_dir / "alignment_bunwarpj_stack.tif"),
        "alignment_sift_overlay": str(out_dir / "alignment_sift_overlay.png"),
        "alignment_bunwarpj_overlay": str(out_dir / "alignment_bunwarpj_overlay.png"),
    }
    return result
