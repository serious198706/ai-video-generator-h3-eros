#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
source ./worker-env.sh
command -v ffprobe >/dev/null || { echo 'Install ffmpeg/ffprobe before deploying.' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'A CUDA GPU host is required.' >&2; exit 1; }
"${H3_PYTHON:-python3.12}" -m h3_service.check_runtime --host-only
if [[ ! -x "$H3_VENV_DIR/bin/python" ]]; then
  "${H3_PYTHON:-python3.12}" -m venv "$H3_VENV_DIR"
fi
dependency_stamp="$("$H3_VENV_DIR/bin/python" -c 'import hashlib,pathlib; print(hashlib.sha256(b"".join(pathlib.Path(p).read_bytes() for p in ["requirements-gpu.txt", "requirements-service.txt"])).hexdigest())')"
if [[ "${H3_REINSTALL:-0}" == 1 || ! -f "$H3_VENV_DIR/.h3-dependencies" || "$(cat "$H3_VENV_DIR/.h3-dependencies")" != "$dependency_stamp" ]]; then
  "$H3_VENV_DIR/bin/python" -m pip install -r requirements-gpu.txt
  printf '%s' "$dependency_stamp" > "$H3_VENV_DIR/.h3-dependencies"
fi
"$H3_VENV_DIR/bin/python" -m pip check
if [[ "${H3_AWS_SETUP:-1}" == 1 ]]; then ./setup-aws.sh; fi
"$H3_VENV_DIR/bin/python" -m h3_service.check_runtime
"$H3_VENV_DIR/bin/python" -m h3_service.download_models
mkdir -p data
"$H3_VENV_DIR/bin/python" -m pip freeze > data/installed-requirements.txt
echo 'Installed and downloaded. Run benchmark.py on this GPU before accepting production tasks.'
