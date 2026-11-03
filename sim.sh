#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
if [[ -f .env.simulator ]]; then set -a; source .env.simulator; set +a; fi
export H3_VENV_DIR="${H3_SIM_VENV_DIR:-$PWD/.venv-simulator}"
export H3_WORK_DIR="${H3_SIM_WORK_DIR:-$PWD/data/simulator}"
export H3_LOG_DIR="${H3_SIM_LOG_DIR:-$PWD/logs/simulator}"
export H3_PORT="${H3_SIM_PORT:-8021}"
export H3_BIND=127.0.0.1
export H3_WORKER_ID="${H3_SIM_WORKER_ID:-local-sim-$(hostname -s)}"
export H3_MODEL_LIFETIME=dual_gpu
export H3_SERVICE_ENTRY="$PWD/sim.sh"
export H3_SIMULATION=1
export AWS_EC2_METADATA_DISABLED=true
[[ -x "$H3_VENV_DIR/bin/python" ]] || { echo 'Run ./deploy-simulator.sh first.' >&2; exit 1; }
action="${1:-start}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  --foreground) exec "$H3_VENV_DIR/bin/python" -m h3_service.simulator.app "$@" ;;
  report|check|submit) exec "$H3_VENV_DIR/bin/python" -m h3_service.simulator.cli "$action" "$@" ;;
  trace) exec "$H3_VENV_DIR/bin/python" -m h3_service.ops.logs "$@" ;;
  aws) exec "$H3_VENV_DIR/bin/python" -m h3_service.aws_setup --save .env.simulator "$@" ;;
  *) exec "$H3_VENV_DIR/bin/python" -m h3_service.ops.service "$action" "$@" ;;
esac
