#!/usr/bin/env bash
# CBraMod: fine-tune the released checkpoint for every task x fold.
#
# Set FORCE_RETRAIN=1 to retrain folds that already have a checkpoint.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"

STAGE_NAME="CBraMod fine-tune"
TRAIN_DATA_DIR="$(model_data_dir cbramod)"
FOUNDATION_CKPT="${IESSEEG_PRETRAINED_DIR:-${IESSEEG_REPO_ROOT}/pretrained-models}/pretrained_weights.pth"

if [ ! -f "${FOUNDATION_CKPT}" ]; then
  echo "CBraMod pretrained weights not found at ${FOUNDATION_CKPT}." >&2
  echo "Download them from the official CBraMod repository, or set" >&2
  echo "IESSEEG_PRETRAINED_DIR to the directory that holds them." >&2
  exit 1
fi

cbramod_train () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"
  local ckpt_dir="${IESSEEG_MODEL_OUTPUT}/ckpts/${tag}"
  local preds_dir="${IESSEEG_MODEL_OUTPUT}/result/val_preds/${tag}"
  mkdir -p "${ckpt_dir}" "${IESSEEG_MODEL_OUTPUT}/result/train/${tag}" "${preds_dir}"

  if [ -f "${ckpt_dir}/best.pth" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "${tag} already has a checkpoint, skipping (FORCE_RETRAIN=1 to redo)"
    return 0
  fi

  "${PYTHON_BIN}" finetune_main.py \
    --train_csv "$(split_csv "${task}" "${fold}" train)" \
    --data_root "${TRAIN_DATA_DIR}" \
    --epoch_length 30 \
    --sfreq 200 \
    --test_size 0.2 \
    --epochs 50 \
    --batch_size 64 \
    --lr 3e-5 \
    --optimizer AdamW \
    --weight_decay 5e-2 \
    --clip_value 1.0 \
    --dropout 0.1 \
    --num_workers 8 \
    --num_of_classes 2 \
    --model_dir "${ckpt_dir}" \
    --model_out "${ckpt_dir}/best.pth" \
    --cuda "${gpu}" \
    --use_pretrained_weights \
    --foundation_dir "${FOUNDATION_CKPT}" \
    --save_preds_dir "${preds_dir}" \
    --label_key "${label_key}"
}

for_each_task_fold cbramod_train
