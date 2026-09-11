"""``py_st2wsi`` command-line interface.

Mirrors the surface of ``run_st2wsi.sh`` so existing shell callers can
swap in the Python port by changing one path:

    ./run_st2wsi.sh        ./py_st2wsi.sh
    # same flags below.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import cli_args
from .pipeline import DEFAULTS, PipelineParams, list_series, run


def _add_defaults(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ref-series", default=DEFAULTS["refSeries"], type=int)
    parser.add_argument("--tgt-series", default=DEFAULTS["tgtSeries"], type=int)
    parser.add_argument(
        "--ref-flipped",
        action="store_true",
        default=DEFAULTS["refFlipped"] == "true",
    )
    parser.add_argument("--ref-rotated", default=DEFAULTS["refRotated"])
    parser.add_argument("--tgt-channel", default=DEFAULTS["tgtChannel"])
    parser.add_argument("--tgt-stain", default=DEFAULTS["tgtStain"])
    parser.add_argument("--pxlSz", default=DEFAULTS["pxlSz"], type=float)
    parser.add_argument("--rolling", default=DEFAULTS["rolling"], type=int)
    parser.add_argument("--sigma", default=DEFAULTS["sigma"], type=float)
    parser.add_argument(
        "--list-series", action="store_true", help="dump series list and exit"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="py_st2wsi",
        description="Pure-Python ST2WSI_Registration: "
        "DAPI → H&E registration.",
    )
    p.add_argument("-o", "--output-dir", required=True)
    p.add_argument("-r", "--ref-image", required=True)
    p.add_argument("-t", "--tgt-image", required=True)
    _add_defaults(p)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Translate argparse Namespace → CLI argmap in the Java shape,
    # so we go through one validated code path.
    cli_argmap = {
        "outputDir": args.output_dir,
        "refImagePath": args.ref_image,
        "tgtImagePath": args.tgt_image,
        "refSeries": str(args.ref_series),
        "tgtSeries": str(args.tgt_series),
        "refFlipped": "true" if args.ref_flipped else "false",
        "refRotated": str(args.ref_rotated),
        "tgtChannel": str(args.tgt_channel),
        "tgtStain": str(args.tgt_stain),
        "pxlSz": str(args.pxlSz),
        "rolling": str(args.rolling),
        "sigma": str(args.sigma),
    }

    if args.list_series:
        print(list_series("reference (DAPI)", args.ref_image))
        print(list_series("target (H&E)", args.tgt_image))
        return 0

    params = PipelineParams.from_argmap(cli_argmap)

    out_dir = Path(params.output_dir)
    if (out_dir / "direct_transf.txt").exists() or (
        out_dir / "registration_params.json"
    ).exists():
        print(
            f"WARNING: existing registration outputs in {out_dir}; will overwrite."
        )

    print("Running py_st2wsi…")
    print(f"  ref       : {params.ref_image_path} (series={params.ref_series})")
    print(f"  tgt       : {params.tgt_image_path} (series={params.tgt_series})")
    print(f"  output    : {params.output_dir}")
    print(
        f"  deconv    : stain={params.tgt_stain} channel={params.tgt_channel}"
    )
    print(
        f"  preproc   : rolling={params.rolling} sigma={params.sigma}"
    )

    result = run(params)
    print(
        f"SIFT inliers: {result.affine.n_inliers}/{result.affine.n_candidates} "
        f"(ratio={result.affine.inlier_ratio:.3f})"
    )
    print(f"bUnwarpJ grid: {result.elastic.N}x{result.elastic.N} (identity fallback)")
    print("Outputs:")
    for k, v in result.output_files.items():
        print(f"  {k}: {v}")
    print("Done.")
    return 0


def main_from_java_arg(arg: str) -> int:
    """Entry point for the IJ2-style ``--run "key=value…"`` invocation.

    Used by the ``ST2WSI_ARG`` environment-variable fallback path and by
    the roundtrip test (``tests/test_roundtrip_args.py``).
    """
    params_map = cli_args.parse_arg(arg)
    if not params_map:
        print(
            "py_st2wsi: empty argument string (set ST2WSI_ARG or pass "
            "--run '…' / -o…)",
            file=sys.stderr,
        )
        return 2

    if params_map.get("listSeries", "false").lower() == "true":
        print(list_series("reference (DAPI)", params_map.get("refImagePath", "")))
        print(list_series("target (H&E)", params_map.get("tgtImagePath", "")))
        return 0

    params = PipelineParams.from_argmap(params_map)
    result = run(params)
    print(json.dumps(result.output_files))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
