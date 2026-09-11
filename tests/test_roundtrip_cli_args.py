"""Round-trip and equivalence tests for py_st2wsi.

Run from the repo root::

    PYTHONPATH=. python3 tests/test_roundtrip_cli_args.py

These are deliberately stdlib-only — they don't require skimage/SIFT to
be importable, so they can run in a CI lane that only has the basics.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


class TestParseArg(unittest.TestCase):
    def test_simple_pairs(self):
        from py_st2wsi.cli_args import parse_arg

        out = parse_arg("a=1 b=2 c=3")
        self.assertEqual(out, {"a": "1", "b": "2", "c": "3"})

    def test_brackets_strip_value_with_space(self):
        from py_st2wsi.cli_args import parse_arg

        out = parse_arg("tgtStain=[H&E 2]")
        self.assertEqual(out, {"tgtStain": "H&E 2"})

    def test_mixed_quotes_and_brackets(self):
        from py_st2wsi.cli_args import parse_arg

        out = parse_arg("a=1 b='two words' c=[three four]")
        self.assertEqual(out, {"a": "1", "b": "two words", "c": "three four"})

    def test_empty(self):
        from py_st2wsi.cli_args import parse_arg

        self.assertEqual(parse_arg(""), {})
        self.assertEqual(parse_arg("   "), {})

    def test_pairs_without_equals_are_dropped(self):
        from py_st2wsi.cli_args import parse_arg

        out = parse_arg("a=1 orphanToken b=2")
        self.assertEqual(out, {"a": "1", "b": "2"})


class TestColorDeconv(unittest.TestCase):
    def test_h_e_default_matches_java_constants(self):
        from py_st2wsi.color_deconv import stain_matrix

        m = stain_matrix("H&E")
        self.assertAlmostEqual(m[0, 0], 0.644211, places=6)
        self.assertAlmostEqual(m[1, 0], 0.092789, places=6)

    def test_h_e_2_matches_java_constants(self):
        from py_st2wsi.color_deconv import stain_matrix

        m = stain_matrix("H&E 2")
        self.assertAlmostEqual(m[0, 0], 0.49015734, places=6)

    def test_deconvolve_outputs_image_correct_shape(self):
        import numpy as np

        from py_st2wsi.color_deconv import deconvolve

        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        out = deconvolve(rgb, "Hematoxylon", "H&E")
        self.assertEqual(out.shape, (64, 64))
        self.assertEqual(out.dtype, np.uint8)


class TestCliFromJavaArg(unittest.TestCase):
    def test_java_arg_round_trip(self):
        """parse_arg must produce a PipelineParams that round-trips through to_java_arg_string."""
        from py_st2wsi.cli_args import parse_arg, to_java_arg_string

        original = (
            "outputDir=/tmp/out refImagePath=/data/dapi.tif "
            "tgtImagePath=/data/he.tif refSeries=3 tgtSeries=3 "
            "refFlipped=false refRotated=90 tgtChannel=Hematoxylon "
            "tgtStain=[H&E 2]"
        )
        round1 = parse_arg(original)
        serialized = to_java_arg_string(round1)
        round2 = parse_arg(serialized)
        self.assertEqual(round1, round2)


class TestOutputSchema(unittest.TestCase):
    def test_transf_writer_format(self):
        import numpy as np

        from py_st2wsi.elastic import (
            ElasticResult,
            identity_coefficients,
            load_direct_transf,
            save_direct_transf,
        )

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "direct_transf.txt"
            field = identity_coefficients(5)
            save_direct_transf(str(p), field)
            text = p.read_text().splitlines()
            self.assertEqual(text[0], "Transformation")
            self.assertEqual(text[1], "-1")
            self.assertEqual(int(text[2]), 5)
            # 5*5 = 25 lines of "cx cy"
            pairs_lines = [
                ln for ln in text[3:] if ln.strip()
            ]
            self.assertEqual(len(pairs_lines), 25)
            # round-trip
            loaded = load_direct_transf(str(p))
            np.testing.assert_array_equal(loaded.cx, field.cx)
            np.testing.assert_array_equal(loaded.cy, field.cy)


if __name__ == "__main__":
    unittest.main()
