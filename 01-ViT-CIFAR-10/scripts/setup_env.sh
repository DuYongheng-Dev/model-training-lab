#!/usr/bin/env bash
# Run after loading the repository's local environment script.
set -euo pipefail
: "${LAB_ROOT:?Load the local environment script first}"
: "${LAB_ENVS_DIR:?Load the local environment script first}"
: "${LAB_RUNS_DIR:?Load the local environment script first}"
: "${PIP_CACHE_DIR:?Load the local environment script first}"
: "${TMPDIR:?Load the local environment script first}"
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
env_dir="$LAB_ENVS_DIR/01-ViT-CIFAR-10/py312-cu130"
setup_dir="$LAB_RUNS_DIR/01-ViT-CIFAR-10/setup-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')"
export CUDA_VISIBLE_DEVICES=''
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST
python3 - <<'PY'
import os, sys
from pathlib import Path
assert sys.version_info[:2] == (3, 12), 'This environment specification uses Python 3.12'
root = Path(os.environ['LAB_ROOT']).resolve(strict=True)
for key in ('LAB_ENVS_DIR', 'LAB_RUNS_DIR', 'PIP_CACHE_DIR', 'TMPDIR'):
    path = Path(os.environ[key]).resolve(strict=True)
    assert path.is_relative_to(root), f'{key} must be under LAB_ROOT'
    assert os.access(path, os.W_OK), f'{key} is not writable'
PY
umask 027
mkdir -p -- "$(dirname -- "$env_dir")" "$(dirname -- "$setup_dir")"
mkdir -- "$setup_dir"
if [[ ! -e $env_dir ]]; then
    python3 -m venv "$env_dir"
fi
"$env_dir/bin/python" -c 'import sys; assert sys.prefix != sys.base_prefix; print(sys.executable)'
if [[ -f $experiment_dir/requirements-lock.txt ]]; then
    "$env_dir/bin/python" -m pip install --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --only-binary=:all: --progress-bar off --timeout 60 --retries 3 \
        --report "$setup_dir/locked-install.json" \
        -r "$experiment_dir/requirements-lock.txt" 2>&1 | tee "$setup_dir/locked-install.log"
else
    "$env_dir/bin/python" -m pip install --only-binary=:all: --progress-bar off \
    --timeout 60 --retries 3 --report "$setup_dir/torch-install.json" \
    -r "$experiment_dir/requirements-torch.txt" 2>&1 | tee "$setup_dir/torch-install.log"
    "$env_dir/bin/python" -m pip install --index-url https://pypi.org/simple \
    --only-binary=:all: --progress-bar off --timeout 60 --retries 3 \
    --report "$setup_dir/other-install.json" \
    -r "$experiment_dir/requirements.txt" 2>&1 | tee "$setup_dir/other-install.log"
fi
"$env_dir/bin/python" -m pip check | tee "$setup_dir/pip-check.txt"
"$env_dir/bin/python" -m pip freeze > "$setup_dir/requirements-resolved.txt"
printf 'Environment: %s\nInstallation records: %s\n' "$env_dir" "$setup_dir"
