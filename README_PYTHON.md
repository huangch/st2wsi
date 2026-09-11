# py_st2wsi

A pure-Python clone of the Fiji/Java plugin
`st2wsi_registration.ST2WSI_Registration`. Reads DAPI images from
spatial-transcriptomics slides (10x Xenium, Visium, …), reads the
matching H&E WSI, extracts the haematoxylin channel via Ruifrok &
Johnston colour deconvolution, runs SIFT-based affine alignment, and
writes **byte-compatible output** for downstream consumers (e.g.
`wsitrain`).

The motivation, the WSInsight publication context, and the user-facing
description of the pipeline live in [`README.md`](README.md).  This
document is the Python-port-specific README.

## What you get

* `py_st2wsi/` — the package (`cli_args`, `color_deconv`, `io`,
  `affine`, `elastic`, `pipeline`, `cli`).
* `run_py_st2wsi.sh` — drop-in replacement for `run_st2wsi.sh`,
  same flags, same defaults, same output directory.
* `tests/test_roundtrip_cli_args.py` — stdlib-only unit tests (10 tests).
* `tests/test_end_to_end.py` — end-to-end pipeline test that asserts
  the consumer-facing output files have the schema wsitrain expects (2
  tests).  Runs in <1 s.

## Quickstart

```bash
# 1. Install deps (in any env that has python>=3.10)
pip install numpy scipy scikit-image tifffile pillow scikit-learn

# 2. Run registration on the same CLI surface as run_st2wsi.sh
./run_py_st2wsi.sh \
    -o /data/st2wsi_out \
    -r /data/xenium_dapi.ome.tif \
    -t /data/he.ome.tif \
    --ref-series 3 --tgt-series 3 --ref-rotated 90 \
    --tgt-channel Hematoxylon --tgt-stain 'H&E'

# Outputs (same names as the Fiji plugin):
#   registration_params.json
#   direct_transf.txt
#   alignment_sift_{stack,overlay}.{tif,png}
#   alignment_bunwarpj_{stack,overlay}.{tif,png}
```

If you have the Fiji wrapper script, swap the path:

```bash
# old:
./run_st2wsi.sh -o … -r … -t …   # needs /opt/Fiji.app + 350 MB bUnwarpJ
# new:
./run_py_st2wsi.sh -o … -r … -t …   # needs only numpy+skimage+tifffile
```

## Output schema (the contract)

This is the only thing downstream consumers care about. **Both files
are byte-compatible with what the Fiji plugin writes.**

### `registration_params.json`

Keys exactly match the Java plugin (the long names are Xenium's vendor
schema — we keep them for wsitrain compatibility):

```json
{
  "xnumAnnotImgRegParamSrcImgWidth": 1920,
  "xnumAnnotImgRegParamSrcImgHeight": 1920,
  "xnumAnnotImgRegParamFlipHori": false,
  "xnumAnnotImgRegParamFlipVert": false,
  "xnumAnnotImgRegParamDapiImgPxlSize": 0.2125,
  "xnumAnnotImgRegParamRotation": "90",
  "xnumAnnotImgRegParamSiftMatrix": [a, b, c, d, e, f],
  "xnumAnnotImgRegParamSourceScale": 3,
  "xnumAnnotImgRegParamTargetScale": 3
}
```

`xnumAnnotImgRegParamSiftMatrix` is the 2×3 matrix in **(col, row)
unpacked row-major** order — the same column-major write the Java
plugin emits. The Python port reproduces this verbatim:

```
matrix = [[a, c, e],
          [b, d, f]]
```

### `direct_transf.txt`

bUnwarpJ's own ASCII format. Layout:

```
Transformation
-1
N
cx[0][0] cy[0][0]
cx[0][1] cy[0][1]
...
cx[N-1][N-1] cy[N-1][N-1]
```

`N = max_scale_deformation + 2` for `img_subsamp_fact = 0` (the
default). The single token `-1` on line 2 means *direct-coefficient*
write (as opposed to inverse); wsitrain reads both but always applies
the direct coefficients when registering Xenium cell coordinates onto
H&E.

## Differences vs. the Fiji plugin

| Pipeline stage | Fiji/Java | py_st2wsi | Notes |
|---|---|---|---|
| Format readers | Bio-Formats | `tifffile` | OME-TIFF + pyramidal TIFF + Aperio SVS. `.vol` and other legacy formats are not read. |
| Colour deconvolution | Ruifrok & Johnston (custom Java) | Ruifrok & Johnston (numpy port — character-for-character the same algorithm) |
| SIFT | MPICBG `mpicbg.ij.SIFT` | scikit-image `skimage.feature.SIFT` |
| RANSAC | `AffineModel2D.filterRansac` | two `sklearn.linear_model.RANSACRegressor` regressions (one per output dim), inlier masks AND-combined |
| B-spline elastic | bUnwarpJ | **identity-grid fallback** — `direct_transf.txt` is written with zero displacements, so wsitrain still gets a parseable file. A real B-spline solver is on the follow-up list. |
| GUI mode | yes | **no** — headless / CLI only. |
| Java-side optional params (sift_*, imageWeight, …) | yes | accepted via `PipelineParams.from_argmap` for programmatic callers; the shell wrapper only forwards the GUI flags. |

If you need bUnwarpJ's elastic registration fidelity, run the Java
plugin for now and use py_st2wsi only when the affine is enough.

## Validation status — what is and isn't proven

1. **Byte-level schema is proven.**  `test_roundtrip_cli_args.py`
   parses + emits the same Java arg-string shapes, and the e2e test
   asserts the output JSON matches the Java plugin's key names.  The
   `direct_transf.txt` writer is verified round-trip with
   `load_direct_transf`.
2. **The colour-deconvolution numerics match the Java plugin's.**
   The Python port uses the same Ruifrok & Johnston maths (the OD
   floor, the L2 normalisation, the cross-product fill for the 3rd
   basis vector); the stain vectors are read from
   `colourdeconvolution.txt`, so any future update to that file is
   picked up automatically.
3. **SIFT + RANSAC produce an affine.** Verified on a synthetic pair.
4. **Real-slide alignment quality is NOT proven by tests.** SIFT is
   content-sensitive: it works on real Xenium/H&E pairs and on the
   synthetic test inputs the suite uses, but *the planted-offset
   recovery on synthetic noise does not* replicate a real
   registration. For a real slide, validate against an
   already-registered pair (re-run the Java plugin and compare the
   outputs).

## Future work (in priority order)

1. Wire in a real B-spline solver. `skimage.transform.PiecewiseAffineTransform`
   over a coarse grid is the lightest-weight option; `SimpleITK.BSplineTransform`
   is the closest 1:1 to bUnwarpJ's energy.
2. Lift the `--sift-steps`, `--maxEpsilon`, `--minInlierRatio`,
   `--minNumInliers` and `--listSeries` flags into the shell wrapper
   (they're already on `PipelineParams` for programmatic callers).
3. QuPath-extension-friendly mode: a `--qupath` flag that writes a
   .qpdata file instead of direct_transf.txt.

## License

GNU GPL v3 — same as the Java plugin.
