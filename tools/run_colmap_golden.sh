#!/usr/bin/env bash
set -euo pipefail

images_dir="${1:?images directory required}"
work_dir="${2:?work directory required}"

rm -rf "$work_dir"
mkdir -p "$work_dir/sparse" "$work_dir/text"

database="$work_dir/database.db"

colmap feature_extractor \
  --database_path "$database" \
  --image_path "$images_dir" \
  --ImageReader.camera_model PINHOLE \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_params 520,520,320,240 \
  --SiftExtraction.use_gpu 0 \
  >/dev/null

colmap exhaustive_matcher \
  --database_path "$database" \
  --SiftMatching.use_gpu 0 \
  >/dev/null

colmap mapper \
  --database_path "$database" \
  --image_path "$images_dir" \
  --output_path "$work_dir/sparse" \
  --Mapper.ba_refine_focal_length 0 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 0 \
  >/dev/null

model_dir="$(find "$work_dir/sparse" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)"
test -n "$model_dir"

colmap model_converter \
  --input_path "$model_dir" \
  --output_path "$work_dir/text" \
  --output_type TXT \
  >/dev/null

registered_lines="$(grep -v '^#' "$work_dir/text/images.txt" | sed '/^[[:space:]]*$/d' | wc -l)"
registered_images="$((registered_lines / 2))"
points="$(grep -v '^#' "$work_dir/text/points3D.txt" | sed '/^[[:space:]]*$/d' | wc -l)"
mean_reprojection_error="$(awk '!/^#/ && NF >= 8 {sum += $8; count += 1} END {if (count == 0) print "nan"; else printf "%.6f", sum / count}' "$work_dir/text/points3D.txt")"

printf 'golden-colmap registered_images=%s points=%s mean_reprojection_error_pixels=%s\n' \
  "$registered_images" "$points" "$mean_reprojection_error"
