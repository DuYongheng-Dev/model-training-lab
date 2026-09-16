#!/usr/bin/env bash
# Load the local environment script before running; --download is opt-in.
set -euo pipefail
: "${LAB_ROOT:?Load the local environment script first}"
: "${LAB_ENVS_DIR:?Load the local environment script first}"
: "${LAB_DATASETS_DIR:?Load the local environment script first}"
: "${LAB_RUNS_DIR:?Load the local environment script first}"
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="$LAB_ENVS_DIR/01-ViT-CIFAR-10/py312-cu130/bin/python"
[[ -x $python_bin ]] || { printf '%s\n' 'Run scripts/setup_env.sh first.' >&2; exit 1; }
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --download ) ]]; then
    printf '%s\n' 'Usage: bash scripts/inspect_data.sh [--download]' >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg
export MPLCONFIGDIR="$LAB_ROOT/cache/matplotlib"
export PYTHONUNBUFFERED=1
run_id="01-data-$($python_bin -c 'import uuid; print(uuid.uuid4().hex[:12])')"
run_dir="$LAB_RUNS_DIR/01-ViT-CIFAR-10/$run_id"
umask 027
mkdir -p -- "$LAB_RUNS_DIR/01-ViT-CIFAR-10" "$MPLCONFIGDIR"
mkdir -- "$run_dir"
mkdir -- "$run_dir/logs"
"$python_bin" "$experiment_dir/src/inspect_data.py" \
    --config "$experiment_dir/configs/01-data.json" \
    --data-root "$LAB_DATASETS_DIR/raw/cifar10" \
    --split-path "$LAB_DATASETS_DIR/processed/cifar10/stratified-45000-5000-seed42.json" \
    --run-dir "$run_dir" "$@" 2>&1 | tee "$run_dir/logs/inspect-data.log"
