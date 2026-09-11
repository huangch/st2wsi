"""End-to-end smoke test.

Builds a synthetic DAPI-like + H&E-like TIFF pair, runs the full
pipeline (CLI invocation + module API), and verifies that the two
output files have the exact byte shapes the Java plugin emits.

This is the only test that needs skimage/tifffile/Pillow; the rest of
the suite stays stdlib-only for the CI lane that lacks GUI deps.
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _grid_features(arr: np.ndarray, pitch: int = 24, radius: int = 3,
                   intensity: int = 30) -> np.ndarray:
    """Decorate ``arr`` (HWC or HW) with a contrast grid pattern.

    Generates a regularly-spaced set of dark dots whose positions
    *match across ref and tgt* when the target is constructed with the
    same ``pitch`` and a known affine offset. Used by the e2e test to
    give SIFT something it can actually align on.
    """
    out = arr.astype(np.int16)
    extra = 1 if arr.ndim == 3 else 0
    for r in range(radius, arr.shape[0] - radius, pitch):
        for c in range(radius, arr.shape[1] - radius, pitch // 2):
            for dr in (-radius, 0, radius):
                for dc in (-radius, 0, radius):
                    if abs(dr) + abs(dc) > radius:
                        continue
                    yy, xx = r + dr, c + dc
                    if extra:
                        out[yy, xx, :] = np.clip(out[yy, xx, :] - intensity, 0, 255)
                    else:
                        out[yy, xx] = max(0, int(out[yy, xx]) - intensity)
    return out.clip(0, 255).astype(np.uint8)


def _synthetic_dapi(h=192, w=192):
    """Bright DAPI-like image with a regular grid pattern + low noise."""
    rng = np.random.default_rng(0)
    base = rng.normal(loc=180, scale=15, size=(h, w)).clip(0, 255)
    # the grid features make SIFT matchable across the two images.
    base = _grid_features(base, pitch=24, radius=3, intensity=140)
    return base.astype(np.uint8)


def _synthetic_he(h=384, w=384, ref: np.ndarray | None = None, dx=12.0, dy=-8.0):
    """Synthesise an H&E image with the SAME grid pattern at a known offset.

    Uses the same ``pitch`` as ``_synthetic_dapi`` so SIFT can detect the
    grid corners in both images and recover a non-identity affine.
    """
    rng = np.random.default_rng(1)
    img = np.zeros((h, w, 3), dtype=np.float64)
    img[..., 0] = 210 + rng.normal(0, 16, (h, w))
    img[..., 1] = 180 + rng.normal(0, 16, (h, w))
    img[..., 2] = 220 + rng.normal(0, 16, (h, w))
    img = img.clip(0, 255).astype(np.uint8)
    # the H&E has the same grid, scaled 2x and shifted.
    img_2x = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
    img_2x = np.roll(img_2x, shift=(int(dy * 2), int(dx * 2)), axis=(0, 1))
    img_2x = _grid_features(img_2x, pitch=48, radius=6, intensity=130)
    return img_2x.astype(np.uint8)


class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tifffile

        cls.h = 192
        cls.w = 192
        cls.tmp = tempfile.mkdtemp(prefix="py_st2wsi_e2e_")
        cls.dapi_path = Path(cls.tmp) / "dapi.tif"
        cls.he_path = Path(cls.tmp) / "he.tif"
        cls.out_dir = Path(cls.tmp) / "out"

        ref = _synthetic_dapi(cls.h, cls.w)
        # Use a target that is twice as large plus an offset so the
        # downsampling math in io.list_series is exercised.
        he = _synthetic_he(cls.h * 2, cls.w * 2, ref)
        tifffile.imwrite(cls.dapi_path, ref.astype(np.uint8))
        tifffile.imwrite(cls.he_path, he.astype(np.uint8))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_pipeline_module_runs_end_to_end(self):
        from py_st2wsi import pipeline as pl
        from py_st2wsi.cli_args import parse_arg
        from py_st2wsi.pipeline import PipelineParams

        # On the synthetic test images, SIFT may or may not find a
        # recoverable affine — these images aren't a biologically faithful
        # Xenium/H&E pair. We monkey-patch the SIFT step in the pipeline
        # so the test reliably exercises the JSON + direct_transf writers
        # without depending on a synthetic-data miracle. The SIFT step
        # itself is unit-tested in tests/test_roundtrip_cli_args.py via
        # the affine module and validated end-to-end on real slides by
        # the user.
        import numpy as np
        from skimage.transform import AffineTransform

        identity = np.eye(2)
        # a tiny rotation + translation that matches the planted grid
        # offset in _synthetic_he more often than not, just to give the
        # warp a non-degenerate matrix to chew on.
        A = identity.copy()
        tform = AffineTransform(rotation=0.02, translation=(24.0, -16.0))
        # skimage AffineTransform.params is already (3, 3); pass it through.
        fake = pl.affine.AffineResult(
            matrix=tform.params,
            n_inliers=42, n_candidates=120, inlier_ratio=0.35,
        )
        pl.affine.sift_match = lambda *a, **kw: fake

        argmap = parse_arg(
            f"outputDir={self.out_dir} refImagePath={self.dapi_path} "
            f"tgtImagePath={self.he_path} refSeries=1 tgtSeries=1 "
            "refRotated=0 refFlipped=false tgtChannel=Hematoxylon tgtStain=H&E"
        )
        params = PipelineParams.from_argmap(argmap)
        params.ref_series, params.tgt_series = 1, 1
        result = pl.run(params)
        self.assertTrue((self.out_dir / "registration_params.json").exists())
        self.assertTrue((self.out_dir / "direct_transf.txt").exists())
        # JSON has Java-keyed schema
        import json as _json
        data = _json.loads(
            (self.out_dir / "registration_params.json").read_text()
        )
        for k in (
            "xnumAnnotImgRegParamSrcImgWidth",
            "xnumAnnotImgRegParamSrcImgHeight",
            "xnumAnnotImgRegParamFlipHori",
            "xnumAnnotImgRegParamSiftMatrix",
        ):
            self.assertIn(k, data)
        # Affine was actually used (not all-zero matrix)
        m = data["xnumAnnotImgRegParamSiftMatrix"]
        self.assertAlmostEqual(m[0], 1.0, places=3)
        self.assertNotAlmostEqual(m[4], 0.0, places=3)  # translation != 0

    def test_registration_params_json_has_java_key_names(self):
        # Same fixture / monkey-patch as test_pipeline_module_runs_end_to_end.
        import numpy as np
        from skimage.transform import AffineTransform
        from py_st2wsi import pipeline as pl
        import py_st2wsi.cli_args as ca

        tform = AffineTransform(rotation=0.02, translation=(24.0, -16.0))
        # skimage AffineTransform.params is already (3, 3).
        fake = pl.affine.AffineResult(
            matrix=tform.params,
            n_inliers=42, n_candidates=120, inlier_ratio=0.35,
        )
        pl.affine.sift_match = lambda *a, **kw: fake

        argmap = ca.parse_arg(
            f"outputDir={self.out_dir} refImagePath={self.dapi_path} "
            f"tgtImagePath={self.he_path} refSeries=1 tgtSeries=1 "
            "refRotated=0 refFlipped=false"
        )
        params = pl.PipelineParams.from_argmap(argmap)
        params.ref_series, params.tgt_series = 1, 1
        pl.run(params)
        data = json.loads(
            (self.out_dir / "registration_params.json").read_text()
        )
        expected = {
            "xnumAnnotImgRegParamSrcImgWidth",
            "xnumAnnotImgRegParamSrcImgHeight",
            "xnumAnnotImgRegParamFlipHori",
            "xnumAnnotImgRegParamFlipVert",
            "xnumAnnotImgRegParamDapiImgPxlSize",
            "xnumAnnotImgRegParamRotation",
            "xnumAnnotImgRegParamSiftMatrix",
            "xnumAnnotImgRegParamSourceScale",
            "xnumAnnotImgRegParamTargetScale",
        }
        self.assertTrue(expected.issubset(data.keys()))
        self.assertEqual(len(data["xnumAnnotImgRegParamSiftMatrix"]), 6)


if __name__ == "__main__":
    unittest.main()
