#!/usr/bin/env bash
set -euo pipefail
: "${LAB_ROOT:?Load the local environment script first}"
: "${LAB_ENVS_DIR:?Load the local environment script first}"
: "${LAB_DATASETS_DIR:?Load the local environment script first}"
: "${LAB_RUNS_DIR:?Load the local environment script first}"
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="$LAB_ENVS_DIR/01-ViT-CIFAR-10/py312-cu130/bin/python"
config="$experiment_dir/configs/07-final.json"
[[ $# == 0 ]] || { printf '%s\n' 'This entry uses the frozen configs/07-final.json.' >&2; exit 2; }
[[ -x $python_bin ]] || { printf '%s\n' 'Prepare the experiment environment first.' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$("$python_bin" "$experiment_dir/scripts/select_gpu.py")
"$python_bin" - <<'PY'
import os, shutil
from pathlib import Path
available_kib = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:')))
assert available_kib > 4 * 1024**2, 'Less than 4 GiB available host memory'
assert shutil.disk_usage(os.environ['LAB_ROOT']).free > 2 * 1024**3, 'Less than 2 GiB free SSD space'
PY
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg MPLCONFIGDIR="$LAB_ROOT/cache/matplotlib" PYTHONUNBUFFERED=1
run_id="07-final-$($python_bin -c 'import uuid; print(uuid.uuid4().hex[:12])')"
run_dir="$LAB_RUNS_DIR/01-ViT-CIFAR-10/$run_id"
source_id=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["training_run_id"])' "$config")
umask 027
mkdir -p -- "$LAB_RUNS_DIR/01-ViT-CIFAR-10" "$MPLCONFIGDIR"
mkdir -- "$run_dir"
mkdir -- "$run_dir/logs" "$run_dir/metrics"
"$python_bin" "$experiment_dir/tests/test_final_metrics.py" --report "$run_dir/metrics/unit-checks.json" \
    2>&1 | tee "$run_dir/logs/checks.log"
# Recheck occupancy after the synthetic-check process has released CUDA.
CUDA_VISIBLE_DEVICES=$("$python_bin" "$experiment_dir/scripts/select_gpu.py")
"$python_bin" "$experiment_dir/src/final_evaluate.py" \
    --config "$config" \
    --source-run "$LAB_RUNS_DIR/01-ViT-CIFAR-10/$source_id" \
    --data-root "$LAB_DATASETS_DIR/raw/cifar10" \
    --split-path "$LAB_DATASETS_DIR/processed/cifar10/stratified-45000-5000-seed42.json" \
    --run-dir "$run_dir" 2>&1 | tee "$run_dir/logs/evaluate.log"
