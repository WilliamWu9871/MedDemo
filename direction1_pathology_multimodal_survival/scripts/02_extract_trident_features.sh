#!/usr/bin/env bash
set -euo pipefail

# Stage 2: One-shot WSI tissue segmentation, 20x 256x256 patching, and offline feature extraction
# using mahmoodlab/TRIDENT (supports conch_v15, uni_v1, ctranspath, resnet50, etc.).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR1_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRIDENT_DIR="${DIR1_ROOT}/external/TRIDENT"

WSI_DIR="${WSI_DIR:-${DIR1_ROOT}/data/tcga_lihc/svs}"
JOB_DIR="${JOB_DIR:-${DIR1_ROOT}/outputs/tcga_lihc/trident_feats}"
PATCH_ENCODER="${PATCH_ENCODER:-hibou_l}"
SLIDE_ENCODER="${SLIDE_ENCODER:-}"
MAG="${MAG:-20}"
PATCH_SIZE="${PATCH_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FEAT_BATCH_SIZE="${FEAT_BATCH_SIZE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wsi-dir) WSI_DIR="$2"; shift 2 ;;
    --job-dir) JOB_DIR="$2"; shift 2 ;;
    --patch-encoder) PATCH_ENCODER="$2"; shift 2 ;;
    --slide-encoder) SLIDE_ENCODER="$2"; shift 2 ;;
    --mag) MAG="$2"; shift 2 ;;
    --patch-size) PATCH_SIZE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --feat-batch-size) FEAT_BATCH_SIZE="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--wsi-dir DIR] [--job-dir DIR] [--patch-encoder hibou_l|conch_v15|uni_v1|ctranspath|resnet50] [--slide-encoder titan] [--mag 20] [--patch-size 224] [--feat-batch-size 256]" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${TRIDENT_DIR}/run_batch_of_slides.py" ]]; then
  echo "[Stage 2][ERROR] TRIDENT not found at ${TRIDENT_DIR}." >&2
  echo "Run: bash ${DIR1_ROOT}/scripts/setup_repos.sh" >&2
  exit 1
fi

if [[ ! -d "${WSI_DIR}" ]]; then
  echo "[Stage 2][ERROR] WSI directory does not exist: ${WSI_DIR}" >&2
  exit 1
fi

mkdir -p "${JOB_DIR}"

CMD=(
  python "${SCRIPT_DIR}/run_trident_fast.py"
  --task all
  --wsi_dir "${WSI_DIR}"
  --job_dir "${JOB_DIR}"
  --patch_encoder "${PATCH_ENCODER}"
  --mag "${MAG}"
  --patch_size "${PATCH_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --remove_artifacts
  --remove_penmarks
  --skip_errors
  --clear_dead_locks
  --dead_lock_max_age_hours 0.01
)

if [[ -n "${FEAT_BATCH_SIZE}" ]]; then
  CMD+=(--feat_batch_size "${FEAT_BATCH_SIZE}")
fi

if [[ -n "${SLIDE_ENCODER}" ]]; then
  CMD+=(--slide_encoder "${SLIDE_ENCODER}")
fi

echo "[Stage 2] Executing TRIDENT:"
echo "  ${CMD[*]}"
echo "[Stage 2] Tip: If gated HuggingFace weights (${PATCH_ENCODER}) are not yet approved, retry with --patch-encoder resnet50 or ctranspath."

"${CMD[@]}"
