#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export JUPYTER_CONFIG_DIR="$PROJECT_ROOT/data/jupyter/config"
export JUPYTER_DATA_DIR="$PROJECT_ROOT/data/jupyter/data"
export JUPYTER_RUNTIME_DIR="$PROJECT_ROOT/data/jupyter/runtime"
export IPYTHONDIR="$PROJECT_ROOT/data/jupyter/ipython"
export MPLCONFIGDIR="$PROJECT_ROOT/data/jupyter/matplotlib"
export PS_TORCH_THREADS=4
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR" "$MPLCONFIGDIR"
exec .venv/bin/jupyter lab --no-browser --ip=127.0.0.1 --ServerApp.root_dir="$PROJECT_ROOT" notebooks/01_offline_ppo.ipynb "$@"
