#!/usr/bin/env bash
# EEGPT: clip-level inference for every task x fold.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"

STAGE_NAME="EEGPT inference"
TEST_DATA_DIR="$(model_test_dir eegpt)"
EEGPT_PYTHON="${PYTHON_BIN}"

eegpt_inference () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"
  local res_dir="${IESSEEG_MODEL_OUTPUT}/result/inference/${tag}"
  mkdir -p "${res_dir}"

  "${EEGPT_PYTHON}" inference_eegpt.py \
    --inference_csv "$(split_csv "${task}" "${fold}" test)" \
    --data_root "${TEST_DATA_DIR}" \
    --model_file "${IESSEEG_MODEL_OUTPUT}/ckpts/${tag}.pth" \
    --out_csv "${res_dir}/inference_results.csv" \
    --label_key "${label_key}" \
    --epoch_length 4 \
    --source_sfreq 200 \
    --source_channels 19 \
    --batch_size 64 \
    --num_workers 4 \
    --cuda "${gpu}"
}

for_each_task_fold eegpt_inference
