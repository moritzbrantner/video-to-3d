#!/usr/bin/env bash
set -euo pipefail

images_dir="${1:?images directory required}"
work_dir="${2:?work directory required}"
case_name="${3:-lateral}"

rm -rf "$work_dir"
mkdir -p "$work_dir/sparse" "$work_dir/models"

database="$work_dir/database.db"

colmap feature_extractor \
  --database_path "$database" \
  --image_path "$images_dir" \
  --ImageReader.camera_model PINHOLE \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_params 520,520,320,240 \
  --SiftExtraction.num_threads 1 \
  --SiftExtraction.use_gpu 0 \
  >/dev/null

colmap exhaustive_matcher \
  --database_path "$database" \
  --SiftMatching.num_threads 1 \
  --SiftMatching.use_gpu 0 \
  >/dev/null

image_count=""
init_pairs=()
while IFS=$'\t' read -r kind first second third fourth; do
  case "$kind" in
    count)
      image_count="$first"
      ;;
    pair)
      init_pairs+=("$first"$'\t'"$second"$'\t'"$third"$'\t'"$fourth")
      ;;
  esac
done < <(python3 tools/colmap_golden_init.py "$database")

test -n "$image_count"
(( image_count >= 2 ))
(( ${#init_pairs[@]} > 0 ))

minimum_registered=$(( image_count > 2 ? image_count - 2 : image_count ))
best_model=""
best_registered=-1
best_points=-1
best_pair=""
attempts=0

for pair in "${init_pairs[@]}"; do
  attempts=$((attempts + 1))
  IFS=$'\t' read -r init_image_id1 init_image_id2 init_name1 init_name2 <<< "$pair"
  attempt_sparse="$work_dir/sparse/attempt-$attempts"
  mkdir -p "$attempt_sparse"

  if ! colmap mapper \
    --database_path "$database" \
    --image_path "$images_dir" \
    --output_path "$attempt_sparse" \
    --Mapper.num_threads 1 \
    --Mapper.init_image_id1 "$init_image_id1" \
    --Mapper.init_image_id2 "$init_image_id2" \
    --Mapper.init_min_tri_angle 8 \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 0 \
    >/dev/null; then
    printf 'colmap-golden-init attempt=%s pair=%s,%s status=mapper-failed\n' \
      "$attempts" "$init_name1" "$init_name2" >&2
    continue
  fi

  attempt_registered=-1
  for model_dir in "$attempt_sparse"/*; do
    test -d "$model_dir" || continue
    model_name="$(basename "$model_dir")"
    text_dir="$work_dir/models/attempt-$attempts-$model_name"
    mkdir -p "$text_dir"
    colmap model_converter \
      --input_path "$model_dir" \
      --output_path "$text_dir" \
      --output_type TXT \
      >/dev/null

    registered_lines="$(grep -v '^#' "$text_dir/images.txt" | sed '/^[[:space:]]*$/d' | wc -l)"
    registered="$((registered_lines / 2))"
    points="$(grep -v '^#' "$text_dir/points3D.txt" | sed '/^[[:space:]]*$/d' | wc -l)"
    if (( registered > attempt_registered )); then
      attempt_registered="$registered"
    fi
    if (( registered > best_registered || (registered == best_registered && points > best_points) )); then
      best_model="$text_dir"
      best_registered="$registered"
      best_points="$points"
      best_pair="$init_name1,$init_name2"
    fi
  done

  printf 'colmap-golden-init attempt=%s pair=%s,%s registered=%s/%s target=%s\n' \
    "$attempts" "$init_name1" "$init_name2" "$attempt_registered" "$image_count" "$minimum_registered" >&2
  if (( attempt_registered >= minimum_registered )); then
    break
  fi
done

test -n "$best_model"

registered_images="$best_registered"
points="$best_points"
mean_reprojection_error="$(awk '!/^#/ && NF >= 8 {sum += $8; count += 1} END {if (count == 0) print "nan"; else printf "%.6f", sum / count}' "$best_model/points3D.txt")"

printf 'golden-colmap case=%s registered_images=%s points=%s mean_reprojection_error_pixels=%s init_attempts=%s init_pair=%s\n' \
  "$case_name" "$registered_images" "$points" "$mean_reprojection_error" "$attempts" "$best_pair"
