#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
source ./worker-env.sh
exec "$H3_VENV_DIR/bin/python" -m uvicorn h3_service.api.app:app --host "${H3_BIND:-127.0.0.1}" --port "${H3_PORT:-8001}" --workers 1 --no-access-log
