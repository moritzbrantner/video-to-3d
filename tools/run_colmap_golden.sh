#!/usr/bin/env bash
set -euo pipefail

images_dir="${1:?images directory required}"
work_dir="${2:?work directory required}"
case_name="${3:-lateral}"
# PINHOLE fx,fy,cx,cy; reference fixtures pass the truth intrinsics they generated.
camera_params="${4:-520,520,320,240}"

rm -rf "$work_dir"
mkdir -p "$work_dir/sparse" "$work_dir/models"

# One pinned pycolmap runs every COLMAP stage (see tools/colmap_golden.py).
if ! python3 "$(dirname "${BASH_SOURCE[0]}")/colmap_golden.py" \
  "$images_dir" "$work_dir" "$camera_params" >&2; then
  echo "COLMAP reference pipeline reported failure for case $case_name" >&2
fi

best_model=""
best_registered=-1
for text_dir in "$work_dir"/models/*; do
  test -f "$text_dir/images.txt" || continue
  registered_lines="$(grep -v '^#' "$text_dir/images.txt" | sed '/^[[:space:]]*$/d' | wc -l)"
  registered="$((registered_lines / 2))"
  if (( registered > best_registered )); then
    best_model="$text_dir"
    best_registered="$registered"
  fi
done

if test -z "$best_model"; then
  # No model is evidence too (the pure-rotation control expects it); the golden checks
  # reject it wherever COLMAP must register.
  printf 'golden-colmap case=%s registered_images=0 points=0 mean_reprojection_error_pixels=nan\n' "$case_name"
  exit 0
fi

registered_images="$best_registered"
points="$(grep -v '^#' "$best_model/points3D.txt" | sed '/^[[:space:]]*$/d' | wc -l)"
mean_reprojection_error="$(awk '!/^#/ && NF >= 8 {sum += $8; count += 1} END {if (count == 0) print "nan"; else printf "%.6f", sum / count}' "$best_model/points3D.txt")"

printf 'golden-colmap case=%s registered_images=%s points=%s mean_reprojection_error_pixels=%s\n' \
  "$case_name" "$registered_images" "$points" "$mean_reprojection_error"
