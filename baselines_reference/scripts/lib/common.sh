#!/usr/bin/env bash
# Shared setup sourced by every train/inference runner.
#
# This exists because the per-model runners previously each carried their
# own copy of the same environment preamble, GPU selection, and task/fold
# loops -- twelve copies that drifted apart over time.
#
# Source it, don't execute it:
#   source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

set -euo pipefail

IESSEEG_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export IESSEEG_REPO_ROOT

# ---------------------------------------------------------------------
# Required configuration
# ---------------------------------------------------------------------
if [ -z "${IESSEEG_DATA_ROOT:-}" ]; then
  echo "IESSEEG_DATA_ROOT is not set. Point it at the directory holding the" >&2
  echo "preprocessed data trees (see iesseeg/config.py for the layout)." >&2
  exit 1
fi
if [ -z "${IESSEEG_OUTPUT_ROOT:-}" ]; then
  echo "IESSEEG_OUTPUT_ROOT is required for checkpoints and results outside the code tree." >&2
  exit 1
fi
export IESSEEG_MODEL_OUTPUT="${IESSEEG_OUTPUT_ROOT}/$(basename "$PWD")"
mkdir -p "${IESSEEG_MODEL_OUTPUT}"
export IESSEEG_SPLIT_ROOT="${IESSEEG_SPLIT_ROOT:-${IESSEEG_REPO_ROOT}/splits}"

PYTHON_BIN="${PYTHON_BIN:-python}"
N_FOLDS="${N_FOLDS:-5}"
TASKS_DEFAULT=(case_control immediate_responder meaningful_responder)

# ---------------------------------------------------------------------
# Scratch space
# ---------------------------------------------------------------------
# Some libraries stage writes through OS-default cache/temp locations
# regardless of the destination path (PyTorch Lightning's atomic
# checkpoint save goes through TMPDIR via fsspec; torch.hub and other
# XDG-compliant tools use the cache vars). On a machine whose root
# filesystem is small or near-full, that fails far from where it was
# configured, so point all three at a scratch directory the user chooses.
IESSEEG_SCRATCH="${IESSEEG_SCRATCH:-${IESSEEG_OUTPUT_ROOT}/scratch}"
export TMPDIR="${IESSEEG_SCRATCH}/tmp"
export XDG_CACHE_HOME="${IESSEEG_SCRATCH}/xdg_cache"
export TORCH_HOME="${IESSEEG_SCRATCH}/torch_cache"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${TORCH_HOME}"

# Cap per-worker math-library threads: each DataLoader worker is its own
# process and would otherwise try to use every core for its internal
# NumPy/OpenMP ops. On a many-core box that oversubscribes badly (seen:
# 10 workers -> 93 threads -> 4114% CPU, most of it context-switching).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

# The binary target column for a task, as named in the split CSVs.
label_key_for () {
  case "$1" in
    case_control)          echo "case_control_label" ;;
    immediate_responder)   echo "immediate_responder" ;;
    meaningful_responder)  echo "meaningful_responder" ;;
    *) echo "label_key_for: unknown task '$1'" >&2; return 1 ;;
  esac
}

split_csv () {  # task fold split -> path
  echo "${IESSEEG_SPLIT_ROOT}/$1/fold_$2/$3.csv"
}

# Each model consumes its own montage/preprocessing tree. These mirror
# MODEL_DATA_SUBDIR / MODEL_TEST_SUBDIR in iesseeg/config.py;
# tests/test_config_consistency.py fails if the two ever disagree.
model_data_dir () {  # model -> preprocessed training tree
  local subdir
  case "$1" in
    handcrafted|cnn_resnet|cnn_vit|luna) subdir="scalp_eeg_data_200HZ_np_format" ;;
    biot)                           subdir="scalp_eeg_data_200HZ_np_format_biot" ;;
    labram|eegpt|reve|codebrain|csbrain) subdir="scalp_eeg_data_200HZ_np_format_labram" ;;
    cbramod)                        subdir="scalp_eeg_data_200HZ_np_format_cbramod" ;;
    *) echo "model_data_dir: unknown model '$1'" >&2; return 1 ;;
  esac
  echo "${IESSEEG_DATA_ROOT}/${subdir}"
}

model_test_dir () {  # model -> Routine-Clip evaluation tree
  local subdir
  case "$1" in
    handcrafted|cnn_resnet|cnn_vit|luna) subdir="baseline_test" ;;
    biot)                           subdir="biot_test" ;;
    labram|eegpt|reve|codebrain|csbrain) subdir="labram_test" ;;
    cbramod)                        subdir="cbramod_test" ;;
    *) echo "model_test_dir: unknown model '$1'" >&2; return 1 ;;
  esac
  echo "${IESSEEG_DATA_ROOT}/${subdir}"
}

# Least-busy GPU with at least $1 MiB free (default 12000), or the pinned
# CUDA_DEVICE when the caller set one.
#
# The default threshold is deliberately generous rather than merely
# "enough for this job": on a shared machine another user's job can grow
# into memory that looked free moments ago (this bit us once -- a BIOT
# run OOM'd when another process ramped up on the GPU we had picked).
# The margin buys headroom against that; it does not eliminate the race.
pick_gpu () {
  if [ -n "${CUDA_DEVICE:-}" ]; then
    echo "${CUDA_DEVICE}"
    return 0
  fi

  local min_free="${1:-12000}"
  local candidates
  candidates="$(
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits |
    awk -F',[[:space:]]*' -v min_free="${min_free}" '
      { idx = $1 + 0; used = $2 + 0; total = $3 + 0; util = $4 + 0
        free = total - used
        if (free >= min_free) print idx, util, free }
    '
  )"

  if [ -z "${candidates}" ]; then
    echo "pick_gpu: no GPU with >= ${min_free} MiB free. Current state:" >&2
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv >&2
    return 1
  fi

  echo "${candidates}" | sort -k2,2n -k3,3nr | head -1 | awk '{print $1}'
}

# Run one stage for every task x fold. The caller supplies a function
# that takes (task, fold, label_key, gpu); everything else -- iteration
# order, GPU selection, and the progress banner -- is handled here.
for_each_task_fold () {
  local body="$1"
  local task fold label_key gpu
  local tasks=()
  local folds=()

  # Task selection, in order of precedence: a TASKS array set by the
  # calling script, then the IESSEEG_TASKS environment variable (a
  # space-separated string, since arrays cannot cross a process
  # boundary), then all three tasks.
  if [ -n "${TASKS+set}" ] && [ "${#TASKS[@]}" -gt 0 ]; then
    tasks=("${TASKS[@]}")
  elif [ -n "${IESSEEG_TASKS:-}" ]; then
    read -r -a tasks <<< "${IESSEEG_TASKS}"
  else
    tasks=("${TASKS_DEFAULT[@]}")
  fi

  if [ -n "${IESSEEG_FOLDS:-}" ]; then
    read -r -a folds <<< "${IESSEEG_FOLDS}"
  else
    for ((fold = 0; fold < N_FOLDS; fold++)); do
      folds+=("${fold}")
    done
  fi

  for task in "${tasks[@]}"; do
    label_key="$(label_key_for "${task}")"
    for fold in "${folds[@]}"; do
      if ! [[ "${fold}" =~ ^[0-9]+$ ]] || [ "${fold}" -ge "${N_FOLDS}" ]; then
        echo "for_each_task_fold: invalid fold '${fold}' for N_FOLDS=${N_FOLDS}" >&2
        return 1
      fi
      gpu="$(pick_gpu)"
      echo
      echo "################################################################"
      echo "# ${STAGE_NAME:-run}: ${task} fold ${fold} (GPU ${gpu})"
      echo "################################################################"
      "${body}" "${task}" "${fold}" "${label_key}" "${gpu}"
    done
  done
}
