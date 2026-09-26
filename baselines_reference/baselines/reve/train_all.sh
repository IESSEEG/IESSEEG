#!/usr/bin/env bash
# REVE: fine-tune the released encoder for every task x fold.
#
# Weights are pulled from the gated brain-bzh/reve-base repository, so a
# Hugging Face token that has accepted REVE's terms must be available
# (hf auth login, or HF_TOKEN).
#
# Set FORCE_RETRAIN=1 to retrain folds that already have a checkpoint.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"

STAGE_NAME="REVE fine-tune"
TRAIN_DATA_DIR="$(model_data_dir reve)"

reve_train () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"
  mkdir -p "${IESSEEG_MODEL_OUTPUT}/ckpts" "${IESSEEG_MODEL_OUTPUT}/result/val_preds/${tag}"

  if [ -f "${IESSEEG_MODEL_OUTPUT}/ckpts/${tag}.pth" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "${tag} already has a checkpoint, skipping (FORCE_RETRAIN=1 to redo)"
    return 0
  fi

  "${PYTHON_BIN}" train_reve.py \
    --train_csv "$(split_csv "${task}" "${fold}" train)" \
    --data_root "${TRAIN_DATA_DIR}" \
    --model_out "${IESSEEG_MODEL_OUTPUT}/ckpts/${tag}.pth" \
    --save_preds_dir "${IESSEEG_MODEL_OUTPUT}/result/val_preds/${tag}" \
    --label_key "${label_key}" \
    --epoch_length 30 \
    --source_sfreq 200 \
    --source_channels 19 \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 5e-2 \
    --layer_decay 0.75 \
    --warmup_epochs 5 \
    --patience 8 \
    --train_iterations 2500 \
    --val_step_sec 240 \
    --num_workers 4 \
    --seed 42 \
    --cuda "${gpu}"
}

for_each_task_fold reve_train
