#!/usr/bin/env bash
# EEGPT: fine-tune the released encoder for every task x fold.
#
# EEGPT needs braindecode >= 1.3 for its model class, which is newer than
# the rest of the benchmark pins, so it runs from its own interpreter.
# Point IESSEEG_PYTHON_EEGPT at that environment (see README.md).
#
# Set FORCE_RETRAIN=1 to retrain folds that already have a checkpoint.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"

STAGE_NAME="EEGPT fine-tune"
TRAIN_DATA_DIR="$(model_data_dir eegpt)"
EEGPT_PYTHON="${IESSEEG_PYTHON_EEGPT:-${PYTHON_BIN}}"

eegpt_train () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"
  mkdir -p "${IESSEEG_MODEL_OUTPUT}/ckpts" "${IESSEEG_MODEL_OUTPUT}/result/val_preds/${tag}"

  if [ -f "${IESSEEG_MODEL_OUTPUT}/ckpts/${tag}.pth" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "${tag} already has a checkpoint, skipping (FORCE_RETRAIN=1 to redo)"
    return 0
  fi

  "${EEGPT_PYTHON}" train_eegpt.py \
    --train_csv "$(split_csv "${task}" "${fold}" train)" \
    --data_root "${TRAIN_DATA_DIR}" \
    --model_out "${IESSEEG_MODEL_OUTPUT}/ckpts/${tag}.pth" \
    --save_preds_dir "${IESSEEG_MODEL_OUTPUT}/result/val_preds/${tag}" \
    --label_key "${label_key}" \
    --epoch_length 4 \
    --source_sfreq 200 \
    --source_channels 19 \
    --epochs 50 \
    --batch_size 64 \
    --lr 5e-4 \
    --weight_decay 5e-2 \
    --layer_decay 0.75 \
    --warmup_epochs 5 \
    --patience 8 \
    --train_iterations 5000 \
    --val_step_sec 120 \
    --num_workers 4 \
    --seed 42 \
    --cuda "${gpu}"
}

for_each_task_fold eegpt_train
