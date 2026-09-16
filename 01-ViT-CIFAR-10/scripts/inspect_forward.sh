#!/usr/bin/env bash
# Forward inspection on the assigned GPU; --cpu reproduces the earlier CPU lesson.
set -euo pipefail
: "${LAB_ROOT:?Load the local environment script first}"
: "${LAB_ENVS_DIR:?Load the local environment script first}"
: "${LAB_DATASETS_DIR:?Load the local environment script first}"
: "${LAB_RUNS_DIR:?Load the local environment script first}"
device=cuda
if [[ $# -eq 1 && $1 == --cpu ]]; then
    device=cpu
elif [[ $# -ne 0 ]]; then
    printf '%s\n' 'Usage: bash scripts/inspect_forward.sh [--cpu]' >&2
    exit 2
fi
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="$LAB_ENVS_DIR/01-ViT-CIFAR-10/py312-cu130/bin/python"
[[ -x $python_bin ]] || { printf '%s\n' 'Run scripts/setup_env.sh first.' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=''
if [[ $device == cuda ]]; then
    CUDA_VISIBLE_DEVICES=$("$python_bin" "$experiment_dir/scripts/select_gpu.py")
fi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg MPLCONFIGDIR="$LAB_ROOT/cache/matplotlib" PYTHONUNBUFFERED=1
run_id="02-forward-$($python_bin -c 'import uuid; print(uuid.uuid4().hex[:12])')"
run_dir="$LAB_RUNS_DIR/01-ViT-CIFAR-10/$run_id"
umask 027
mkdir -p -- "$LAB_RUNS_DIR/01-ViT-CIFAR-10" "$MPLCONFIGDIR"
mkdir -- "$run_dir"
mkdir -- "$run_dir/logs"
CUDA_VISIBLE_DEVICES='' "$python_bin" -m unittest discover -s "$experiment_dir/tests" -p test_model.py -v \
    2>&1 | tee "$run_dir/logs/model-tests.log"
"$python_bin" "$experiment_dir/src/inspect_forward.py" \
    --device "$device" \
    --config "$experiment_dir/configs/02-forward.json" \
    --data-root "$LAB_DATASETS_DIR/raw/cifar10" \
    --split-path "$LAB_DATASETS_DIR/processed/cifar10/stratified-45000-5000-seed42.json" \
    --run-dir "$run_dir" 2>&1 | tee "$run_dir/logs/inspect-forward.log"
