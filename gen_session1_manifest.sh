#!/bin/bash
# Build the recording -> bouts_csv manifest for the Session1 parallel
# prediction array job. Each line: <recording_dir><TAB><bouts_csv>.
# Fails fast if a mapped recording is missing calibration/ or videos.
set -euo pipefail

BROOT=${BROOT:-/gscratch/portia/eabe/data/Johnson_lab/courtship/Session1_bouts_04172026}
VID=${VID:-/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1}
OUT=${1:-session1_manifest.tsv}

: > "$OUT"
n=0
while IFS= read -r csv; do
  # recording name = grandparent dir of the CSV (.../<rec>/Predictions_3D_*/csv)
  rec=$(basename "$(dirname "$(dirname "$csv")")")
  recdir="$VID/$rec"
  [[ -d "$recdir" ]]              || { echo "ERROR: no recording dir: $recdir (from $csv)" >&2; exit 1; }
  [[ -d "$recdir/calibration" ]] || { echo "ERROR: no calibration/ in $recdir" >&2; exit 1; }
  ls "$recdir"/*.mp4 >/dev/null 2>&1 || { echo "ERROR: no .mp4 in $recdir" >&2; exit 1; }
  printf '%s\t%s\n' "$recdir" "$csv" >> "$OUT"
  n=$((n+1))
done < <(find "$BROOT" -name courtship_bouts_unified_summary.csv | sort)

echo "wrote $n manifest lines to $OUT" >&2
