#!/usr/bin/env bash
H3_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_MODEL_DIR="${H3_MODEL_DIR:-$H3_ROOT/models/base}"
export H3_EROS_FILE="${H3_EROS_FILE:-$H3_ROOT/models/eros/10Eros_Max_h3_TURBO-hybrid_beta5.safetensors}"
export H3_WORK_DIR="${H3_WORK_DIR:-$H3_ROOT/data}"
export H3_VENV_DIR="${H3_VENV_DIR:-$H3_ROOT/.venv}"
export TOKENIZERS_PARALLELISM=false
