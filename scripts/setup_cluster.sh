#!/usr/bin/env bash
# Run explicitly in a JupyterLab terminal on the cluster; never in the global env.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ECG_ENV="${ECG_ENV:-$HOME/.venvs/ecg-a100}"
ECG_PYTHON="${ECG_PYTHON:-python3.11}"
if [[ -e "$ECG_ENV" ]]; then
  echo "Environment already exists: $ECG_ENV. Preserved; choose a new ECG_ENV to install." >&2
  exit 1
fi
"$ECG_PYTHON" -m venv "$ECG_ENV"
"$ECG_ENV/bin/python" -m pip install --upgrade pip
"$ECG_ENV/bin/python" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
"$ECG_ENV/bin/python" -m pip install -r "$PROJECT_ROOT/requirements/cluster-a100.txt"
"$ECG_ENV/bin/python" -m pip install --no-deps -e "$PROJECT_ROOT"
"$ECG_ENV/bin/python" -m pip check
"$ECG_ENV/bin/python" -m ipykernel install --user --name "${ECG_KERNEL:-ecg-a100}" --display-name 'ECG A100 (Python 3.11)'
"$ECG_ENV/bin/python" -m pip freeze > "$ECG_ENV/requirements-resolved.txt"
echo "Select the ECG A100 kernel in JupyterLab. Data/artifacts were not modified."
