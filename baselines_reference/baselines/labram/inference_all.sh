#!/usr/bin/env bash
# LaBraM: clip-level inference for every task x fold.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"

STAGE_NAME="LaBraM inference"
TEST_DATA_DIR="$(model_test_dir labram)"

labram_inference () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"
  local res_dir="${IESSEEG_MODEL_OUTPUT}/result/inference/${tag}"
  mkdir -p "${res_dir}"

  "${PYTHON_BIN}" inference.py \
    --inference_csv "$(split_csv "${task}" "${fold}" test)" \
    --data_root "${TEST_DATA_DIR}" \
    --model_ckpt "${IESSEEG_MODEL_OUTPUT}/checkpoints/${tag}/checkpoint-best.pth" \
    --out_csv "${res_dir}/inference_results.csv" \
    --epoch_length 10 \
    --sfreq 200 \
    --nb_classes 2 \
    --device "cuda:${gpu}" \
    --disable_qkv_bias \
    --abs_pos_emb \
    --disable_rel_pos_bias \
    --label_key "${label_key}"
}

for_each_task_fold labram_inference
