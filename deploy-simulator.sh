#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
if [[ ! -f .env.simulator ]]; then cp .env.simulator.example .env.simulator; fi
set -a; source .env.simulator; set +a
sim_venv="${H3_SIM_VENV_DIR:-$PWD/.venv-simulator}"
if [[ ! -x "$sim_venv/bin/python" ]]; then
  "${H3_SIM_PYTHON:-python3}" -m venv "$sim_venv"
fi
"$sim_venv/bin/python" -m pip install -r requirements-service.txt
./sim.sh aws
./sim.sh check
./sim.sh install
echo 'Local simulator installed. ./sim.sh start; ./sim.sh status; ./sim.sh report --expected 50 --watch'
