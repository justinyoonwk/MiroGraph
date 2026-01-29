#!/usr/bin/env bash
set -euo pipefail

# Sequential runner for SIGIR/CIKM IR=10 with gpt-5-mini
# Settings: transductive/inductive × with_subgraph/without_subgraph

PY="/home/mshwang825/anaconda3/envs/cast/bin/python3"
ROOT="/home/mshwang825/Impact_Forecast"
DATA_ROOT="${ROOT}/dataset"
OUT_ROOT="${ROOT}/results"

# You can override these via environment variables, e.g.:
#   MODEL=gpt-5.2 BS=20 SLEEP_SEC=12 ./run_simple_sigir_cikm_ir10.sh
MODEL="${MODEL:-gpt-5-mini}"
IR="${IR:-10}"
API_KEY_NAME="${API_KEY_NAME:-1}"
BS="${BS:-40}"
SLEEP_SEC="${SLEEP_SEC:-8}"

# Safe tag for filenames
MODEL_TAG="${MODEL//\//_}"

mkdir -p "${OUT_ROOT}"

run_one () {
  local dataset="$1"
  local setting="$2"          # transductive|inductive
  local subgraph_flag="$3"    # "with"|"without"

  local test_file="${DATA_ROOT}/${dataset}/${setting}/test_data_${IR}.csv"
  local out_file="${OUT_ROOT}/${dataset}/${setting}/simple_prediction_ir${IR}_${MODEL_TAG}_${subgraph_flag}.jsonl"

  local disable_arg=""
  if [[ "${subgraph_flag}" == "without" ]]; then
    disable_arg="--disable_subgraph"
  fi

  echo "================================================================================"
  echo "[RUN] dataset=${dataset} setting=${setting} subgraph=${subgraph_flag}"
  echo "      test_file=${test_file}"
  echo "      out_file=${out_file}"
  echo "================================================================================"

  ${PY} -u "${ROOT}/simple_prediction.py" \
    --test_file "${test_file}" \
    --ir "${IR}" \
    --model "${MODEL}" \
    --api_key_name "${API_KEY_NAME}" \
    --batch_size "${BS}" \
    --output "${out_file}" \
    ${disable_arg}

  echo "[DONE] ${out_file}"
  echo
  sleep "${SLEEP_SEC}"
}

# SIGIR
run_one "SIGIR" "transductive" "with"
run_one "SIGIR" "transductive" "without"
run_one "SIGIR" "inductive" "with"
run_one "SIGIR" "inductive" "without"

# CIKM
run_one "CIKM" "transductive" "with"
run_one "CIKM" "transductive" "without"
run_one "CIKM" "inductive" "with"
run_one "CIKM" "inductive" "without"

echo "All runs finished."

