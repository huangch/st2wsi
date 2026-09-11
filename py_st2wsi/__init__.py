"""py_st2wsi — pure-Python clone of ST2WSI_Registration.

Drop-in CLI replacement for the Fiji/Java plugin
``st2wsi_registration.ST2WSI_Registration``.

Output schema is byte-compatible with the Java plugin:

* ``registration_params.json`` uses the same key names and the same
  row-major unpack of the 2x3 affine matrix that the Java code emits.
* ``direct_transf.txt`` is written in bUnwarpJ's own plain-text format
  (the only documented source for this file is bUnwarpJ itself;
  :mod:`py_st2wsi.elastic` reproduces the format character for character
  from the published bUnwarpJ serialization so downstream consumers
  like wsitrain keep working unchanged).

This module deliberately depends only on the scientific Python stack
(numpy, scipy, scikit-image, tifffile, Pillow). It does *not* require
OpenCV or SimpleITK.
"""
from __future__ import annotations

__all__ = [
    "cli",
    "pipeline",
    "cli_args",
    "color_deconv",
    "io",
    "affine",
    "elastic",
]

__version__ = "0.1.0"
