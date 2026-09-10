#!/usr/bin/env bash
set -euo pipefail

fixture=${1:-target/golden-scene}
output=${2:-target/golden-scene/colmap-cameras.tsv}
fixture=$(realpath "$fixture")
output=$(realpath -m "$output")
workspace=$(mktemp -d)
trap 'rm -rf "$workspace"' EXIT
mkdir -p "$workspace/sparse" "$workspace/text" "$(dirname "$output")"

ground_truth="$fixture/ground_truth.tsv"
focal=$(awk -F '\t' 'NR==2 {print $7}' "$ground_truth")
width=$(awk -F '\t' 'NR==2 {print $8}' "$ground_truth")
height=$(awk -F '\t' 'NR==2 {print $9}' "$ground_truth")
principal_x=$(python3 -c "print(float('$width') * 0.5)")
principal_y=$(python3 -c "print(float('$height') * 0.5)")

feature_help=$(colmap feature_extractor -h 2>&1 || true)
match_help=$(colmap exhaustive_matcher -h 2>&1 || true)
if grep -q -- '--FeatureExtraction.use_gpu' <<<"$feature_help"; then
  feature_gpu=(--FeatureExtraction.use_gpu 0)
else
  feature_gpu=(--SiftExtraction.use_gpu 0)
fi
if grep -q -- '--FeatureMatching.use_gpu' <<<"$match_help"; then
  match_gpu=(--FeatureMatching.use_gpu 0)
else
  match_gpu=(--SiftMatching.use_gpu 0)
fi

export QT_QPA_PLATFORM=offscreen
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

colmap feature_extractor \
  --database_path "$workspace/database.db" \
  --image_path "$fixture/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_PINHOLE \
  --ImageReader.camera_params "$focal,$principal_x,$principal_y" \
  "${feature_gpu[@]}" >/dev/null

colmap exhaustive_matcher \
  --database_path "$workspace/database.db" \
  "${match_gpu[@]}" >/dev/null

colmap mapper \
  --database_path "$workspace/database.db" \
  --image_path "$fixture/images" \
  --output_path "$workspace/sparse" \
  --Mapper.ba_refine_focal_length 0 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 0 >/dev/null

model_dir=$(find "$workspace/sparse" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)
if [[ -z "$model_dir" ]]; then
  echo "COLMAP did not produce a sparse model" >&2
  exit 1
fi

colmap model_converter \
  --input_path "$model_dir" \
  --output_path "$workspace/text" \
  --output_type TXT >/dev/null

python3 scripts/extract_colmap_cameras.py "$workspace/text/images.txt" "$output"
