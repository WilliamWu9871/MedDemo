#!/usr/bin/env bash
set -euo pipefail

# Clone and wire official Paper 2 & Paper 3 repositories under external/:
#   - bioFAM/scDoRI              (Paper 2 core framework, cisBP motifs, official notebooks)
#   - PMBio/GBM_analysis         (Paper 2 plasticity scoring & TAP/TRS transition scripts)
#   - BayraktarLab/GBM_space_paper (Paper 3 cell2location + NMF TM + trajectory notebooks)
# Optional flags:
#   --with-spacetree   Also clone PMBio/spaceTree (~463 MB)
#   --with-hest        Also clone mahmoodlab/HEST (not needed for selective HF downloads)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXT_DIR="${DIR2_ROOT}/external"

WITH_SPACETREE=false
WITH_HEST=false

for arg in "$@"; do
  case "$arg" in
    --with-spacetree) WITH_SPACETREE=true ;;
    --with-hest) WITH_HEST=true ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: $0 [--with-spacetree] [--with-hest]" >&2
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

clone_or_update "https://github.com/bioFAM/scDoRI.git" "${EXT_DIR}/scDoRI"
clone_or_update "https://github.com/PMBio/GBM_analysis.git" "${EXT_DIR}/GBM_analysis"
clone_or_update "https://github.com/BayraktarLab/GBM_space_paper.git" "${EXT_DIR}/GBM_space_paper"

if [[ "${WITH_SPACETREE}" == "true" ]]; then
  clone_or_update "https://github.com/PMBio/spaceTree.git" "${EXT_DIR}/spaceTree"
fi
if [[ "${WITH_HEST}" == "true" ]]; then
  clone_or_update "https://github.com/mahmoodlab/HEST.git" "${EXT_DIR}/HEST"
fi

echo "[setup_repos] Checking Python version compatibility for bioFAM/scDoRI (>=3.10,<3.13)..."
PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
if ! python -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)'; then
  echo "[setup_repos][ERROR] bioFAM/scDoRI requires Python >=3.10,<3.13, but active python is ${PY_VER} ($(command -v python))." >&2
  echo "  Activate a Python 3.10-3.12 environment first (e.g. conda activate meddemo_sc_spatial, or run Step 3 in colab_launcher.ipynb)." >&2
  exit 1
fi

echo "[setup_repos] Installing bioFAM/scDoRI in editable mode (preserving env pins)..."
python -m pip install -e "${EXT_DIR}/scDoRI" --no-build-isolation --no-deps || python -m pip install -e "${EXT_DIR}/scDoRI"

echo "[setup_repos] Done. Official repositories ready under ${EXT_DIR}:"
ls -1 "${EXT_DIR}"
