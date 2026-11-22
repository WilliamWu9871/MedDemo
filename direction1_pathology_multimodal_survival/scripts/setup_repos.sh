#!/usr/bin/env bash
set -euo pipefail

# Clone and wire upstream computational pathology repositories for Direction 1:
#   - mahmoodlab/TRIDENT  (WSI segmentation, 20x 256x256 tiling, CONCH v1.5 / UNI / ResNet50 feature extraction)
#   - mahmoodlab/SurvPath (Discrete-time NLLSurvLoss, GatedABMIL, multimodal token survival reference)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR1_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXT_DIR="${DIR1_ROOT}/external"
WITH_PATCH_ENCODERS=false

for arg in "$@"; do
  case "$arg" in
    --with-patch-encoders)
      WITH_PATCH_ENCODERS=true
      ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: $0 [--with-patch-encoders]" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${EXT_DIR}"

clone_or_update() {
  local repo_url="$1"
  local target_dir="$2"
  if [[ -d "${target_dir}/.git" ]]; then
    echo "[setup_repos] Updating existing repo at ${target_dir}..."
    git -C "${target_dir}" pull --ff-only || true
  else
    echo "[setup_repos] Shallow cloning ${repo_url} -> ${target_dir}..."
    git clone --depth 1 "${repo_url}" "${target_dir}"
  fi
}

clone_or_update "https://github.com/mahmoodlab/TRIDENT.git" "${EXT_DIR}/TRIDENT"
clone_or_update "https://github.com/mahmoodlab/SurvPath.git" "${EXT_DIR}/SurvPath"

echo "[setup_repos] Installing mahmoodlab/TRIDENT in editable mode..."
if [[ "${WITH_PATCH_ENCODERS}" == "true" ]]; then
  python -m pip install -e "${EXT_DIR}/TRIDENT[patch-encoders]"
else
  python -m pip install -e "${EXT_DIR}/TRIDENT"
fi

if command -v trident-doctor >/dev/null 2>&1; then
  echo "[setup_repos] Running TRIDENT preflight check (non-fatal)..."
  trident-doctor --profile patch-encoders --check-gated || true
else
  echo "[setup_repos] Note: trident-doctor not on PATH yet; verify after activating meddemo_patho."
fi

echo "[setup_repos] Done. Cloned repositories:"
echo "  - ${EXT_DIR}/TRIDENT"
echo "  - ${EXT_DIR}/SurvPath"
