#!/usr/bin/env bash
# LUNA: fine-tune the released checkpoint for every task x fold.
#
# Set FORCE_RETRAIN=1 to retrain folds that already have a checkpoint.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"

STAGE_NAME="LUNA fine-tune"
TRAIN_DATA_DIR="$(model_data_dir luna)"
PRETRAINED="${IESSEEG_PRETRAINED_DIR:-${IESSEEG_REPO_ROOT}/pretrained-models}/LUNA_base.safetensors"

if [ ! -f "${PRETRAINED}" ]; then
  echo "LUNA weights not found at ${PRETRAINED}." >&2
  echo "Download LUNA_base.safetensors from https://huggingface.co/PulpBio/LUNA" >&2
  echo "(CC BY-ND 4.0: fine-tuning for internal use is permitted, redistributing" >&2
  echo "the fine-tuned weights is not), or set IESSEEG_PRETRAINED_DIR." >&2
  exit 1
fi

luna_train () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"
  local ckpt_dir="${IESSEEG_MODEL_OUTPUT}/ckpts"
  local preds_dir="${IESSEEG_MODEL_OUTPUT}/result/val_preds/${tag}"
  mkdir -p "${ckpt_dir}" "${preds_dir}"

  if [ -f "${ckpt_dir}/${tag}.pth" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "${tag} already has a checkpoint, skipping (FORCE_RETRAIN=1 to redo)"
    return 0
  fi

  "${PYTHON_BIN}" train_luna.py \
    --train_csv "$(split_csv "${task}" "${fold}" train)" \
    --data_root "${TRAIN_DATA_DIR}" \
    --pretrained "${PRETRAINED}" \
    --model_out "${ckpt_dir}/${tag}.pth" \
    --save_preds_dir "${preds_dir}" \
    --label_key "${label_key}" \
    --epoch_length 30 \
    --source_sfreq 200 \
    --source_channels 22 \
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

for_each_task_fold luna_train
