#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
if [[ "${1:-}" != --foreground ]]; then
  exec bash ./service.sh "${1:-start}"
fi
shift
if [[ -f .env ]]; then set -a; source .env; set +a; fi
source ./worker-env.sh
[[ -x "$H3_VENV_DIR/bin/python" ]] || { echo 'H3 venv missing; run ./deploy.sh' >&2; exit 1; }
# Worker startup verifies STS/queues inside the guarded process before model loading.
exec "$H3_VENV_DIR/bin/python" -m h3_service.serve
