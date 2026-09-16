#!/usr/bin/env bash
# Measure bounded inference workloads after loading the local environment script.
set -euo pipefail
: "${LAB_ROOT:?Load the local environment script first}"
: "${LAB_ENVS_DIR:?Load the local environment script first}"
: "${LAB_DATASETS_DIR:?Load the local environment script first}"
: "${LAB_RUNS_DIR:?Load the local environment script first}"
[[ $# -eq 0 ]] || { printf '%s\n' 'Usage: bash scripts/benchmark_devices.sh' >&2; exit 2; }
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="$LAB_ENVS_DIR/01-ViT-CIFAR-10/py312-cu130/bin/python"
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$("$python_bin" "$experiment_dir/scripts/select_gpu.py")
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg MPLCONFIGDIR="$LAB_ROOT/cache/matplotlib" PYTHONUNBUFFERED=1
run_id="02-device-benchmark-$($python_bin -c 'import uuid; print(uuid.uuid4().hex[:12])')"
run_dir="$LAB_RUNS_DIR/01-ViT-CIFAR-10/$run_id"
umask 027
mkdir -p -- "$LAB_RUNS_DIR/01-ViT-CIFAR-10" "$MPLCONFIGDIR"
mkdir -- "$run_dir"
mkdir -- "$run_dir/logs"
"$python_bin" "$experiment_dir/src/benchmark_devices.py" \
    --config "$experiment_dir/configs/02-device-benchmark.json" \
    --data-root "$LAB_DATASETS_DIR/raw/cifar10" \
    --split-path "$LAB_DATASETS_DIR/processed/cifar10/stratified-45000-5000-seed42.json" \
    --run-dir "$run_dir" 2>&1 | tee "$run_dir/logs/benchmark.log"
