#!/usr/bin/env bash
set -euo pipefail

images_dir="${1:?images directory required}"
work_dir="${2:?work directory required}"
case_name="${3:-lateral}"
# PINHOLE fx,fy,cx,cy; reference fixtures pass the truth intrinsics they generated.
camera_params="${4:-520,520,320,240}"

rm -rf "$work_dir"
mkdir -p "$work_dir/sparse" "$work_dir/models"

database="$work_dir/database.db"

# Keep stochastic geometry estimation seeded. CPU approximate SIFT matching
# in COLMAP 3.9.1 still varies with this seed, so use exact matching below.
colmap_random_args=(--random_seed 0)

colmap feature_extractor \
  --database_path "$database" \
  --image_path "$images_dir" \
  --ImageReader.camera_model PINHOLE \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_params "$camera_params" \
  --SiftExtraction.num_threads 1 \
  --SiftExtraction.use_gpu 0 \
  "${colmap_random_args[@]}" \
  >/dev/null

python3 "$(dirname "${BASH_SOURCE[0]}")/match_colmap_golden.py" "$database" >/dev/null

# COLMAP exits non-zero when it cannot initialize any model; that outcome is reported
# below as a zero-registration result instead of aborting the evidence run.
if ! colmap mapper \
  --database_path "$database" \
  --image_path "$images_dir" \
  --output_path "$work_dir/sparse" \
  --Mapper.num_threads 1 \
  --Mapper.init_min_tri_angle 8 \
  --Mapper.ba_refine_focal_length 0 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 0 \
  "${colmap_random_args[@]}" \
  >/dev/null; then
  echo "COLMAP mapper reported failure for case $case_name" >&2
fi

best_model=""
best_registered=-1
for model_dir in "$work_dir"/sparse/*; do
  test -d "$model_dir" || continue
  model_name="$(basename "$model_dir")"
  text_dir="$work_dir/models/$model_name"
  mkdir -p "$text_dir"
  colmap model_converter \
    --input_path "$model_dir" \
    --output_path "$text_dir" \
    --output_type TXT \
    >/dev/null
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
