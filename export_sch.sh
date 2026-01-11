#!/usr/bin/env bash
set -euo pipefail

SCH="${1:-bucket_buckets.kicad_sch}"
OUTDIR="${2:-exports}"

mkdir -p "$OUTDIR"

# PDF export
kicad-cli sch export pdf -o "$OUTDIR/$(basename "$SCH" .kicad_sch).pdf" "$SCH"

# SVG export (folder of SVG sheets)
kicad-cli sch export svg -o "$OUTDIR/svg" "$SCH"

echo "Wrote:"
echo "  $OUTDIR/$(basename "$SCH" .kicad_sch).pdf"
echo "  $OUTDIR/svg/"
