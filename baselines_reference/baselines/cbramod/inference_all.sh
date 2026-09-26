#!/usr/bin/env bash
# CBraMod: clip-level inference for every task x fold.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"

STAGE_NAME="CBraMod inference"
TEST_DATA_DIR="$(model_test_dir cbramod)"

cbramod_inference () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"
  local res_dir="${IESSEEG_MODEL_OUTPUT}/result/inference/${tag}"
  mkdir -p "${res_dir}"

  "${PYTHON_BIN}" inference_cbramod.py \
    --inference_csv "$(split_csv "${task}" "${fold}" test)" \
    --data_root "${TEST_DATA_DIR}" \
    --model_file "${IESSEEG_MODEL_OUTPUT}/ckpts/${tag}/best.pth" \
    --out_csv "${res_dir}/inference_results.csv" \
    --epoch_length 30 \
    --sfreq 200 \
    --label_key "${label_key}" \
    --cuda "${gpu}"
}

for_each_task_fold cbramod_inference
