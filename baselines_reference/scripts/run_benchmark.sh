#!/usr/bin/env bash
# Master runner: trains, runs inference, and aggregates metrics for every
# baseline across all folds and tasks.
#
# Usage:
#   bash scripts/run_benchmark.sh                       # everything
#   MODELS="biot cbramod" bash scripts/run_benchmark.sh  # subset of models
#   STAGE=train bash scripts/run_benchmark.sh            # training only
#   STAGE=inference bash scripts/run_benchmark.sh        # needs checkpoints
#   STAGE=eval bash scripts/run_benchmark.sh             # metrics only
#   CUDA_DEVICE=1 bash scripts/run_benchmark.sh          # pin to one GPU
#   N_FOLDS=3 bash scripts/run_benchmark.sh              # override fold count
#
# Required: IESSEEG_DATA_ROOT must point at the preprocessed data trees.
# Optional: IESSEEG_SCRATCH for cache/temp space (defaults under $TMPDIR).
#
# Leave CUDA_DEVICE unset to auto-pick the least-busy free GPU fresh
# before each run, which adapts to other jobs on a shared machine.
#
# For a long run, detach it so it survives your shell closing:
#   nohup bash scripts/run_benchmark.sh > run_benchmark.log 2>&1 &

if [ "${1:-}" = "--help" ]; then
  sed -n '2,20p' "$0"
  echo '# Required output root: IESSEEG_OUTPUT_ROOT (external checkpoints/results directory).'
  exit 0
fi
if [ "$#" -gt 0 ]; then
  echo 'This runner accepts configuration through environment variables; use --help.' >&2
  exit 2
fi
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

MODELS="${MODELS:-handcrafted cnn_resnet cnn_vit biot labram cbramod luna eegpt reve codebrain csbrain}"
STAGE="${STAGE:-all}"   # train | inference | eval | all

BASELINE_DIR="${IESSEEG_REPO_ROOT}/baselines"
STEP_LOG=()

# Each model directory owns a train_all.sh and an inference_all.sh; they
# read the same environment this script exports, so adding a baseline
# means dropping in a directory and naming it here.
declare -A MODEL_DIR=(
  [handcrafted]="handcrafted"
  [cnn_resnet]="cnn"
  [cnn_vit]="cnn"
  [biot]="biot"
  [labram]="labram"
  [cbramod]="cbramod"
  [luna]="luna"
  [eegpt]="eegpt"
  [reve]="reve"
  [codebrain]="codebrain"
  [csbrain]="csbrain"
)
declare -A TRAIN_SCRIPT=(
  [handcrafted]="train_all.sh"
  [cnn_resnet]="train_cnn_all.sh"
  [cnn_vit]="train_vit_all.sh"
  [biot]="train_all.sh"
  [labram]="train_all.sh"
  [cbramod]="train_all.sh"
  [luna]="train_all.sh"
  [eegpt]="train_all.sh"
  [reve]="train_all.sh"
  [codebrain]="train_all.sh"
  [csbrain]="train_all.sh"
)
declare -A INFER_SCRIPT=(
  [handcrafted]="inference_all.sh"
  [cnn_resnet]="inference_cnn_all.sh"
  [cnn_vit]="inference_vit_all.sh"
  [biot]="inference_all.sh"
  [labram]="inference_all.sh"
  [cbramod]="inference_all.sh"
  [luna]="inference_all.sh"
  [eegpt]="inference_all.sh"
  [reve]="inference_all.sh"
  [codebrain]="inference_all.sh"
  [csbrain]="inference_all.sh"
)

run_step () {
  local desc="$1" dir="$2" script="$3"
  local path="${BASELINE_DIR}/${dir}/${script}"

  if [ ! -f "${path}" ]; then
    echo ">> ${desc}: no ${script} in baselines/${dir}; skipping."
    STEP_LOG+=("SKIP  ${desc}")
    return 0
  fi

  echo
  echo "################################################################"
  echo "# ${desc}"
  echo "################################################################"

  local t0 t1
  t0=$(date +%s)
  if (cd "${BASELINE_DIR}/${dir}" && bash "${script}"); then
    t1=$(date +%s)
    STEP_LOG+=("OK    ${desc}  ($((t1 - t0))s)")
  else
    t1=$(date +%s)
    STEP_LOG+=("FAIL  ${desc}  ($((t1 - t0))s)")
    # Keep going: one model failing should not cost the whole sweep.
    echo ">> ${desc} FAILED; continuing with the remaining steps." >&2
  fi
}

for model in ${MODELS}; do
  if [ -z "${MODEL_DIR[$model]:-}" ]; then
    echo ">> unknown model '${model}'; skipping." >&2
    continue
  fi
  if [ "${STAGE}" = "train" ] || [ "${STAGE}" = "all" ]; then
    run_step "${model}: train" "${MODEL_DIR[$model]}" "${TRAIN_SCRIPT[$model]}"
  fi
  if [ "${STAGE}" = "inference" ] || [ "${STAGE}" = "all" ]; then
    run_step "${model}: inference" "${MODEL_DIR[$model]}" "${INFER_SCRIPT[$model]}"
  fi
done

if [ "${STAGE}" = "eval" ] || [ "${STAGE}" = "all" ]; then
  echo
  echo "################################################################"
  echo "# Scoring and aggregating"
  echo "################################################################"
  "${PYTHON_BIN}" "${IESSEEG_REPO_ROOT}/scripts/evaluate.py" \
    --results_root "${IESSEEG_OUTPUT_ROOT}" \
    ${MODELS:+--models ${MODELS}} \
    ${IESSEEG_LATEX_DIR:+--latex_dir "${IESSEEG_LATEX_DIR}"}
fi

echo
echo "================================================================"
echo "Summary"
echo "================================================================"
printf '%s\n' "${STEP_LOG[@]}"
