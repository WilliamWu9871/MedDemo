#!/usr/bin/env bash
set -euo pipefail

# Provision an isolated Python 3.11 venv for Direction 2 on Google Colab
# (Colab hosts default to Python 3.13, while bioFAM/scDoRI requires >=3.10,<3.13 and numpy<2).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${1:-/content/venv_meddemo_dir2_py311}"

apt-get update -qq
apt-get install -y -qq bedtools samtools tabix

command -v uv >/dev/null 2>&1 || pip install -q uv
uv venv --clear --seed --python 3.11 "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python --version

uv pip install --python "${VENV_DIR}/bin/python" \
  'setuptools>=78,<81' torch torchvision \
  'numpy>=1.24,<2' 'pandas>=1.5,<3' 'scipy>=1.10,<1.17' \
  'anndata>=0.10,<0.12' 'scanpy>=1.10,<1.12' 'pyarrow<26' 'pynndescent<0.7' \
  'harmonypy>=0.0.10,<0.1' 'gtfparse>=2.5,<3' 'tangermeme>=0.4.0,<0.5' 'decoupler>=1.8,<1.9' \
  'pyro-ppl>=1.8.0,<1.9.2' mudata muon squidpy scvi-tools cell2location \
  leidenalg igraph umap-learn papermill ipykernel jupyter session-info2 hatchling hatch-vcs \
  huggingface_hub requests pyyaml tqdm matplotlib seaborn Pillow

python -m ipykernel install --user --name meddemo_dir2_py311 --display-name "Python 3.11 (meddemo_dir2)"
bash "${DIR2_ROOT}/scripts/setup_repos.sh"

python -c "import sys, numpy, torch, anndata, scanpy, muon, squidpy, scvi, cell2location, scdori, decoupler; print('OK | Python:', sys.version.split()[0], '| NumPy:', numpy.__version__, '| PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| scDoRI:', scdori.__file__)"
