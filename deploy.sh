#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
source ./worker-env.sh
command -v ffprobe >/dev/null || { echo 'Install ffmpeg/ffprobe before deploying.' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'A CUDA GPU host is required.' >&2; exit 1; }
"${H3_PYTHON:-python3.12}" -m h3_service.check_runtime --host-only
"${H3_PYTHON:-python3.12}" -m venv "$H3_VENV_DIR"
"$H3_VENV_DIR/bin/python" -m pip install -r requirements-gpu.txt
"$H3_VENV_DIR/bin/python" -m pip check
"$H3_VENV_DIR/bin/python" -m h3_service.check_runtime
"$H3_VENV_DIR/bin/python" -m h3_service.download_models
mkdir -p data
"$H3_VENV_DIR/bin/python" -m pip freeze > data/installed-requirements.txt
echo 'Installed and downloaded. Run benchmark.py on this GPU before accepting production tasks.'
