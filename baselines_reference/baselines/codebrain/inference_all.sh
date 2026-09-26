#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

STAGE_NAME="CodeBrain inference"
TEST_DATA_DIR="$(model_test_dir codebrain)"
PRETRAINED="${IESSEEG_PRETRAINED_DIR:-${IESSEEG_REPO_ROOT}/pretrained-models}/CodeBrain.pth"

codebrain_inference () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"
  local res_dir="${IESSEEG_MODEL_OUTPUT}/result/inference/${tag}"
  mkdir -p "${res_dir}"
  "${PYTHON_BIN}" inference_codebrain.py \
    --inference_csv "$(split_csv "${task}" "${fold}" test)" \
    --data_root "${TEST_DATA_DIR}" \
    --pretrained "${PRETRAINED}" \
    --model_file "${IESSEEG_MODEL_OUTPUT}/ckpts/${tag}.pth" \
    --out_csv "${res_dir}/inference_results.csv" \
    --label_key "${label_key}" --batch_size 32 --num_workers 4 --cuda "${gpu}"
}

for_each_task_fold codebrain_inference
