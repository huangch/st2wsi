#!/bin/bash
# run_py_st2wsi.sh — Drop-in pure-Python replacement for run_st2wsi.sh.
#
# Same flags, same defaults, same output files.  Differs from the Fiji
# wrapper in two ways:
#   * Doesn't need a Fiji install (everything is in the conda env).
#   * The bUnwarpJ elastic step is an identity-grid fallback until the
#     follow-up that wires in a real B-spline solver (so this script
#     does NOT depend on Java/Bio-Formats/bUnwarpJ/MPICBG at all).
#
# Usage:
#   ./run_py_st2wsi.sh \
#     -o /path/to/out \
#     -r /path/to/dapi.ome.tif \
#     -t /path/to/he.ome.tif \
#     --ref-series 3 --tgt-series 3 --ref-rotated 90 \
#     --tgt-channel Hematoxylon --tgt-stain 'H&E'
#
# Or in IJ2-arg style (matches Fiji's --run "key=value…"):
#   ./run_py_st2wsi.sh --java-arg "outputDir=/data/out refImagePath=… tgtImagePath=… refSeries=3 tgtSeries=3 refRotated=90 tgtChannel=Hematoxylon tgtStain=[H&E 2]"
set -euo pipefail

# Locate the python interpreter and the package sibling to this script.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

# Decide which entry point to call.
if [[ "${1:-}" == "--java-arg" ]]; then
    shift
    PYTHONPATH="$HERE" exec "$PYTHON" -c "
from py_st2wsi.cli import main_from_java_arg
import sys
sys.exit(main_from_java_arg(sys.argv[1]))
" "$1"
fi

PYTHONPATH="$HERE" exec "$PYTHON" -m py_st2wsi "$@"
