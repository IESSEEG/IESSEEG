#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

STAGE_NAME="CodeBrain fine-tune"
TRAIN_DATA_DIR="$(model_data_dir codebrain)"
PRETRAINED="${IESSEEG_PRETRAINED_DIR:-${IESSEEG_REPO_ROOT}/pretrained-models}/CodeBrain.pth"

if [ ! -f "${PRETRAINED}" ]; then
  echo "CodeBrain weights not found at ${PRETRAINED}." >&2
  echo "Download CodeBrain.pth from https://huggingface.co/YjMajy/CodeBrain." >&2
  exit 1
fi

codebrain_train () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"
  mkdir -p "${IESSEEG_MODEL_OUTPUT}/ckpts" "${IESSEEG_MODEL_OUTPUT}/result/val_preds/${tag}"
  if [ -f "${IESSEEG_MODEL_OUTPUT}/ckpts/${tag}.pth" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "${tag} already has a checkpoint, skipping (FORCE_RETRAIN=1 to redo)"
    return 0
  fi
  "${PYTHON_BIN}" train_codebrain.py \
    --train_csv "$(split_csv "${task}" "${fold}" train)" \
    --data_root "${TRAIN_DATA_DIR}" \
    --pretrained "${PRETRAINED}" \
    --model_out "${IESSEEG_MODEL_OUTPUT}/ckpts/${tag}.pth" \
    --save_preds_dir "${IESSEEG_MODEL_OUTPUT}/result/val_preds/${tag}" \
    --label_key "${label_key}" \
    --epochs 50 --batch_size 32 --lr 1e-4 --weight_decay 5e-4 \
    --dropout 0.1 --patience 8 --train_iterations 2500 --val_step_sec 240 \
    --num_workers 4 --seed 42 --cuda "${gpu}"
}

for_each_task_fold codebrain_train
