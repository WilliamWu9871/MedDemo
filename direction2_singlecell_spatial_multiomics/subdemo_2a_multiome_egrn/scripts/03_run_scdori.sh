#!/usr/bin/env bash
set -euo pipefail

# Sub-demo 2A Step 3: Execute official bioFAM/scDoRI v2.1+ notebooks via papermill.
# Runs strictly inside subdemo_2a_multiome_egrn/ so relative config paths resolve cleanly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB2A_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIR2_ROOT="$(cd "${SUB2A_ROOT}/.." && pwd)"
SCDORI_DIR="${DIR2_ROOT}/external/scDoRI"
NB_DIR="${SCDORI_DIR}/docs/notebooks"

CONFIG_JSON="${SUB2A_ROOT}/configs/scdori_pbmc_hg38.json"
STAGE="all"   # preprocessing | training | downstream | all
KERNEL_NAME=""
CLEAN_CHECKPOINTS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_JSON="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --kernel) KERNEL_NAME="$2"; shift 2 ;;
    --clean-checkpoints) CLEAN_CHECKPOINTS=true; shift ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--config configs/scdori_pbmc_hg38.json] [--stage preprocessing|training|downstream|all] [--kernel meddemo_dir2_py311] [--clean-checkpoints]" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "${NB_DIR}" ]]; then
  echo "[2A Step 3][ERROR] Official scDoRI notebooks not found at ${NB_DIR}." >&2
  echo "Run: bash ${DIR2_ROOT}/scripts/setup_repos.sh" >&2
  exit 1
fi

if [[ "${CONFIG_JSON}" != /* ]]; then
  CONFIG_JSON="${SUB2A_ROOT}/${CONFIG_JSON}"
fi
if [[ ! -f "${CONFIG_JSON}" ]]; then
  echo "[2A Step 3][ERROR] Config JSON not found: ${CONFIG_JSON}" >&2
  exit 1
fi

if [[ -z "${KERNEL_NAME}" ]] && command -v jupyter >/dev/null 2>&1; then
  if jupyter kernelspec list 2>/dev/null | grep -qw "meddemo_dir2_py311"; then
    KERNEL_NAME="meddemo_dir2_py311"
  fi
fi

if [[ "${CLEAN_CHECKPOINTS}" == "true" ]]; then
  echo "[2A Step 3] Cleaning existing phase1/phase2 checkpoints under ${SUB2A_ROOT}/outputs/scdori/..."
  rm -rf "${SUB2A_ROOT}/outputs/scdori/phase1" "${SUB2A_ROOT}/outputs/scdori/phase2"
fi

OUT_NB_DIR="${SUB2A_ROOT}/outputs/scdori/executed_notebooks"
mkdir -p "${OUT_NB_DIR}"

run_nb() {
  local nb_name="$1"
  local in_nb="${NB_DIR}/${nb_name}.ipynb"
  local out_nb="${OUT_NB_DIR}/${nb_name}.executed.ipynb"
  local kernel_args=()
  if [[ -n "${KERNEL_NAME}" ]]; then
    kernel_args=(-k "${KERNEL_NAME}")
  fi
  echo "[2A Step 3] Executing ${in_nb} with SCDORI_CONFIG_FILE=${CONFIG_JSON} (kernel=${KERNEL_NAME:-default})..."
  (
    cd "${SUB2A_ROOT}"
    SCDORI_CONFIG_FILE="${CONFIG_JSON}" papermill "${in_nb}" "${out_nb}" "${kernel_args[@]}"
  )
}

case "${STAGE}" in
  preprocessing)
    run_nb "preprocessing"
    ;;
  training)
    run_nb "training"
    ;;
  downstream)
    run_nb "downstream"
    ;;
  all)
    run_nb "preprocessing"
    run_nb "training"
    run_nb "downstream"
    ;;
  *)
    echo "[2A Step 3][ERROR] Invalid stage: ${STAGE}" >&2
    exit 1
    ;;
esac

echo "[2A Step 3] scDoRI notebooks finished. Executed logs in ${OUT_NB_DIR}"
