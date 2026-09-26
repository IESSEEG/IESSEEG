#!/usr/bin/env bash
# LUNA: clip-level inference for every task x fold.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"

STAGE_NAME="LUNA inference"
TEST_DATA_DIR="$(model_test_dir luna)"

luna_inference () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"
  local res_dir="${IESSEEG_MODEL_OUTPUT}/result/inference/${tag}"
  mkdir -p "${res_dir}"

  "${PYTHON_BIN}" inference_luna.py \
    --inference_csv "$(split_csv "${task}" "${fold}" test)" \
    --data_root "${TEST_DATA_DIR}" \
    --model_file "${IESSEEG_MODEL_OUTPUT}/ckpts/${tag}.pth" \
    --out_csv "${res_dir}/inference_results.csv" \
    --label_key "${label_key}" \
    --epoch_length 30 \
    --source_sfreq 200 \
    --source_channels 22 \
    --batch_size 64 \
    --num_workers 4 \
    --cuda "${gpu}"
}

for_each_task_fold luna_inference
