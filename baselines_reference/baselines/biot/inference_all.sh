#!/usr/bin/env bash
# BIOT: clip-level inference for every task x fold.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"

STAGE_NAME="BIOT inference"
TEST_DATA_DIR="$(model_test_dir biot)"

biot_inference () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"
  local ckpt_dir="${IESSEEG_MODEL_OUTPUT}/ckpts/${tag}"
  local res_dir="${IESSEEG_MODEL_OUTPUT}/result/inference/${tag}"
  mkdir -p "${res_dir}"

  "${PYTHON_BIN}" inference.py \
    --inference_csv "$(split_csv "${task}" "${fold}" test)" \
    --data_root "${TEST_DATA_DIR}" \
    --model_ckpt "${ckpt_dir}/best_BIOT.ckpt" \
    --out_csv "${res_dir}/inference_results.csv" \
    --epoch_length 30 \
    --sfreq 200 \
    --batch_size 64 \
    --cuda "${gpu}" \
    --label_key "${label_key}"
}

for_each_task_fold biot_inference
