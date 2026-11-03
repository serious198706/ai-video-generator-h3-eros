#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
source ./worker-env.sh
[[ -x "$H3_VENV_DIR/bin/python" ]] || { echo 'Run ./deploy.sh first.' >&2; exit 1; }
exec "$H3_VENV_DIR/bin/python" -m h3_service.ops.service "$@"
