#!/usr/bin/env bash
set -euo pipefail

FIJI_BIN="${FIJI_BIN:-/opt/Fiji.app/ImageJ-linux64}"

print_help() {
    cat <<'EOF'
Usage:
  st2wsi gui [fiji_args...]
  st2wsi cli [plugin_args...]
  st2wsi help

Modes:
  gui  Launch Fiji GUI. Requires DISPLAY and X11 socket forwarding.
  cli  Run ST2WSI in headless mode via run_st2wsi.sh argument set.

CLI mode arguments (forwarded to st2wsi-cli):
  -o, --output-dir DIR
  -r, --ref-image FILE
  -t, --tgt-image FILE
  --ref-series INT
  --tgt-series INT
  --ref-flipped
  --ref-rotated DEG
  --tgt-channel NAME

Examples:
  st2wsi gui
  st2wsi cli -o /data/out -r /data/dapi.ome.tif -t /data/wsi.ome.tif
EOF
}

mode="${1:-help}"

case "$mode" in
    gui)
        shift
        if [[ -z "${DISPLAY:-}" ]]; then
            echo "Error: DISPLAY is not set. For GUI mode, pass DISPLAY and mount /tmp/.X11-unix from host." >&2
            exit 1
        fi
        exec "$FIJI_BIN" "$@"
        ;;
    cli)
        shift
        exec /usr/local/bin/st2wsi-cli "$@"
        ;;
    help|-h|--help)
        print_help
        ;;
    *)
        exec "$@"
        ;;
esac
