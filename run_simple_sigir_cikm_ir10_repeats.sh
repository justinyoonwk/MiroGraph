#!/usr/bin/env bash
set -euo pipefail

# Repeat runner for SIGIR/CIKM IR=10
# Runs: datasets (SIGIR, CIKM) × settings (transductive, inductive) × subgraph (with, without) × repeats
#
# Usage example:
#   MODEL=gpt-5-mini BS=20 SLEEP_SEC=12 REPEATS=3 ./run_simple_sigir_cikm_ir10_repeats.sh
#
# Outputs:
#   results/<DATASET>/<SETTING>/repeats/<RUN_TAG>/simple_prediction_ir10_<MODEL>_<with|without>_run{1..R}.jsonl

PY="/home/mshwang825/anaconda3/envs/cast/bin/python3"
ROOT="/home/mshwang825/Impact_Forecast"
DATA_ROOT="${ROOT}/dataset"
OUT_ROOT="${ROOT}/results"

MODEL="${MODEL:-gpt-5-mini}"
IR="${IR:-10}"
API_KEY_NAME="${API_KEY_NAME:-1}"
BS="${BS:-20}"
SLEEP_SEC="${SLEEP_SEC:-12}"
REPEATS="${REPEATS:-3}"

# Safe tag for filenames
MODEL_TAG="${MODEL//\//_}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

echo "================================================================================"
echo "[REPEATS RUNNER] MODEL=${MODEL} IR=${IR} BS=${BS} REPEATS=${REPEATS} SLEEP_SEC=${SLEEP_SEC}"
echo "[RUN_TAG] ${RUN_TAG}"
echo "================================================================================"
echo

run_one () {
  local dataset="$1"
  local setting="$2"          # transductive|inductive
  local subgraph_flag="$3"    # with|without
  local rep="$4"              # 1..REPEATS

  local test_file="${DATA_ROOT}/${dataset}/${setting}/test_data_${IR}.csv"
  local out_dir="${OUT_ROOT}/${dataset}/${setting}/repeats/${RUN_TAG}"
  mkdir -p "${out_dir}"
  local out_file="${out_dir}/simple_prediction_ir${IR}_${MODEL_TAG}_${subgraph_flag}_run${rep}.jsonl"

  local disable_arg=""
  if [[ "${subgraph_flag}" == "without" ]]; then
    disable_arg="--disable_subgraph"
  fi

  echo "--------------------------------------------------------------------------------"
  echo "[RUN] dataset=${dataset} setting=${setting} subgraph=${subgraph_flag} rep=${rep}/${REPEATS}"
  echo "      test_file=${test_file}"
  echo "      out_file=${out_file}"
  echo "--------------------------------------------------------------------------------"

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

for rep in $(seq 1 "${REPEATS}"); do
  # SIGIR
  run_one "SIGIR" "transductive" "with" "${rep}"
  run_one "SIGIR" "transductive" "without" "${rep}"
  run_one "SIGIR" "inductive" "with" "${rep}"
  run_one "SIGIR" "inductive" "without" "${rep}"

  # CIKM
  run_one "CIKM" "transductive" "with" "${rep}"
  run_one "CIKM" "transductive" "without" "${rep}"
  run_one "CIKM" "inductive" "with" "${rep}"
  run_one "CIKM" "inductive" "without" "${rep}"
done

echo "All repeats finished. RUN_TAG=${RUN_TAG}"

