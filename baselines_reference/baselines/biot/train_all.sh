#!/usr/bin/env bash
# BIOT: fine-tune the released checkpoint for every task x fold.
#
# Set FORCE_RETRAIN=1 to retrain folds that already have a checkpoint.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"

STAGE_NAME="BIOT fine-tune"
MODEL_NAME="BIOT"
TRAIN_DATA_DIR="$(model_data_dir biot)"
PRETRAIN_PATH="${IESSEEG_PRETRAINED_DIR:-${IESSEEG_REPO_ROOT}/pretrained-models}/EEG-six-datasets-18-channels.ckpt"

EPOCHS=50
BATCH=64
LR=3e-5
WORKERS=8

if [ ! -f "${PRETRAIN_PATH}" ]; then
  echo "BIOT pretrained checkpoint not found at ${PRETRAIN_PATH}." >&2
  echo "Download it from the official BIOT repository, or set" >&2
  echo "IESSEEG_PRETRAINED_DIR to the directory that holds it." >&2
  exit 1
fi

biot_train () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"
  local ckpt_dir="${IESSEEG_MODEL_OUTPUT}/ckpts/${tag}"
  local preds_dir="${IESSEEG_MODEL_OUTPUT}/result/val_preds/${tag}"
  mkdir -p "${ckpt_dir}" "${preds_dir}"

  # Resume support: a shared machine means jobs get interrupted by disk
  # and GPU contention, so a completed fold is not redone by default.
  if [ -f "${ckpt_dir}/best_${MODEL_NAME}.ckpt" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "${tag} already has a checkpoint, skipping (FORCE_RETRAIN=1 to redo)"
    return 0
  fi

  "${PYTHON_BIN}" run_binary_supervised.py \
    --train_csv "$(split_csv "${task}" "${fold}" train)" \
    --data_root "${TRAIN_DATA_DIR}" \
    --seed 42 \
    --test_size 0.2 \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH} \
    --lr ${LR} \
    --weight_decay 5e-2 \
    --num_workers ${WORKERS} \
    --dataset "CASE_CTRL" \
    --model "${MODEL_NAME}" \
    --in_channels 18 \
    --sample_length 30 \
    --sfreq 200 \
    --token_size 200 \
    --hop_length 100 \
    --pretrain_model_path "${PRETRAIN_PATH}" \
    --model_dir "${ckpt_dir}" \
    --model_out "best_${MODEL_NAME}" \
    --save_preds_dir "${preds_dir}" \
    --cuda "${gpu}" \
    --label_key "${label_key}"
}

for_each_task_fold biot_train
