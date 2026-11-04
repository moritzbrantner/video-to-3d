#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./scripts/setup_mapanything.sh [--check-only]

Prepare the optional local MapAnything GPU smoke environment. Safe to rerun.
Python dependencies and the upstream source stay in the user cache, not Git.
FFmpeg/ffprobe, git, uv, and the NVIDIA driver must already be available.

Overrides: MAPANYTHING_VENV, MAPANYTHING_SOURCE, XDG_CACHE_HOME.
USAGE
}

check_only=false
case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --check-only) check_only=true ;;
  "") ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

error() {
  printf 'MapAnything setup: %s\n' "$*" >&2
  exit 1
}

cache_root="${XDG_CACHE_HOME:-$HOME/.cache}/video-to-3d"
venv_dir="${MAPANYTHING_VENV:-$cache_root/mapanything-venv}"
source_dir="${MAPANYTHING_SOURCE:-$cache_root/map-anything-src}"
python_bin="$venv_dir/bin/python"

for command_name in git ffmpeg ffprobe nvidia-smi; do
  command -v "$command_name" >/dev/null 2>&1 || error "missing $command_name; install it first (on Ubuntu: sudo apt install git ffmpeg; NVIDIA driver provides nvidia-smi)"
done

python_check() {
  [[ -x "$python_bin" ]] && "$python_bin" -c "$1" >/dev/null 2>&1
}

if [[ ! -x "$python_bin" ]]; then
  "$check_only" && error "missing Python environment $venv_dir; run without --check-only"
  command -v uv >/dev/null 2>&1 || error "missing uv; install it (for example: pipx install uv)"
  mkdir -p "$cache_root"
  uv venv "$venv_dir" --python 3.11
fi

if ! python_check 'import torch; import torchvision; assert torch.cuda.is_available()'; then
  "$check_only" && error 'CUDA-enabled PyTorch is not ready; run without --check-only'
  command -v uv >/dev/null 2>&1 || error 'missing uv; install it (for example: pipx install uv)'
  uv pip install --python "$python_bin" --torch-backend auto --reinstall-package torch torch torchvision
fi

if ! python_check 'from mapanything.models import MapAnything; from mapanything.utils.image import load_images; from mapanything.utils.viz import predictions_to_glb'; then
  "$check_only" && error 'MapAnything is not installed; run without --check-only'
  command -v uv >/dev/null 2>&1 || error 'missing uv; install it (for example: pipx install uv)'
  if [[ ! -d "$source_dir" ]]; then
    mkdir -p "$(dirname "$source_dir")"
    git clone --depth 1 https://github.com/facebookresearch/map-anything.git "$source_dir"
  elif ! git -C "$source_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    error "existing MapAnything source directory is not a Git checkout: $source_dir"
  fi
  uv pip install --python "$python_bin" -e "$source_dir"
fi

if ! python_check 'import yt_dlp'; then
  "$check_only" && error 'yt-dlp is not installed; run without --check-only'
  command -v uv >/dev/null 2>&1 || error 'missing uv; install it (for example: pipx install uv)'
  uv pip install --python "$python_bin" yt-dlp
fi

python_check 'import torch; import torchvision; assert torch.cuda.is_available()' || error 'PyTorch still cannot access CUDA; check the installed driver and CUDA-compatible PyTorch wheel'
python_check 'from mapanything.models import MapAnything; from mapanything.utils.image import load_images; from mapanything.utils.viz import predictions_to_glb; import yt_dlp' || error 'MapAnything or yt-dlp import verification failed'

printf 'MapAnything GPU environment ready: %s\n' "$python_bin"
"$python_bin" -c 'import torch; print("CUDA device:", torch.cuda.get_device_name(0))'
