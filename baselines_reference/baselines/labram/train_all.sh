#!/usr/bin/env bash
# LaBraM: fine-tune the released base checkpoint for every task x fold.
#
# Set FORCE_RETRAIN=1 to retrain folds that already have a checkpoint.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"

STAGE_NAME="LaBraM fine-tune"
TRAIN_DATA_DIR="$(model_data_dir labram)"
LABRAM_BASE_CKPT="${IESSEEG_PRETRAINED_DIR:-${IESSEEG_REPO_ROOT}/pretrained-models}/labram-base.pth"

if [ ! -f "${LABRAM_BASE_CKPT}" ]; then
  echo "LaBraM base checkpoint not found at ${LABRAM_BASE_CKPT}." >&2
  echo "Download it from the official LaBraM repository, or set" >&2
  echo "IESSEEG_PRETRAINED_DIR to the directory that holds it." >&2
  exit 1
fi

labram_train () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"

  if [ -f "${IESSEEG_MODEL_OUTPUT}/checkpoints/${tag}/checkpoint-best.pth" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "${tag} already has a checkpoint, skipping (FORCE_RETRAIN=1 to redo)"
    return 0
  fi

  "${PYTHON_BIN}" run_class_finetuning.py \
    --train_csv "$(split_csv "${task}" "${fold}" train)" \
    --data_root "${TRAIN_DATA_DIR}" \
    --output_dir "${IESSEEG_MODEL_OUTPUT}/checkpoints/${tag}" \
    --log_dir "${IESSEEG_MODEL_OUTPUT}/log/finetune_${tag}_base" \
    --model labram_base_patch200_200 \
    --finetune "${LABRAM_BASE_CKPT}" \
    --weight_decay 0.05 \
    --batch_size 64 \
    --lr 5e-4 \
    --update_freq 1 \
    --warmup_epochs 5 \
    --epochs 50 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --save_ckpt_freq 5 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --dataset "${tag}" \
    --disable_qkv_bias \
    --seed 42 \
    --device "cuda:${gpu}" \
    --label_key "${label_key}"
}

for_each_task_fold labram_train
