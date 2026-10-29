#!/usr/bin/env bash
# Hidden input, STS verification, atomic .env persistence; does not start/restart GPU worker.
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
source ./worker-env.sh
force=()
if [[ "${1:-}" == --force ]]; then force=(--force); shift; fi
exec "$H3_VENV_DIR/bin/python" -m h3_service.aws_setup --save "$PWD/.env" "${force[@]}" -- "$@"
