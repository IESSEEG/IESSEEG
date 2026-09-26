#!/usr/bin/env bash
# Serial reproduction of the current paper. Configure upstream weights first.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/reproduce_all.sh DATASET_DIRECTORY WORK_DIRECTORY cuda:N" >&2
  exit 2
fi
DATASET_ROOT="$(realpath "$1")"
mkdir -p "$2"
BENCHMARK_WORK="$(realpath "$2")"
DEVICE="$3"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${IESSEEG_PRETRAINED_DIR:?Set IESSEEG_PRETRAINED_DIR; see docs/MODELS.md}"
: "${IESSEEG_PYTHON_EEGPT:?Set IESSEEG_PYTHON_EEGPT; see docs/MODELS.md}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
common=(--data "$DATASET_ROOT" --work "$BENCHMARK_WORK")
"$PYTHON_BIN" -m iesseeg validate "${common[@]}"
"$PYTHON_BIN" -m iesseeg prepare "${common[@]}"
for task in diagnosis response; do
  "$PYTHON_BIN" -m iesseeg extract-qeeg "${common[@]}" --task "$task" --device "$DEVICE"
  "$PYTHON_BIN" -m iesseeg qeeg "${common[@]}" --task "$task"
done
# Prepare each shared input family once.
for model in biot labram cbramod luna; do
  "$PYTHON_BIN" -m iesseeg preprocess "${common[@]}" --model "$model" --device "$DEVICE"
done
for model in biot labram cbramod eegpt luna reve codebrain csbrain; do
  interpreter="$PYTHON_BIN"
  if [[ "$model" == eegpt ]]; then interpreter="$IESSEEG_PYTHON_EEGPT"; fi
  "$interpreter" -m iesseeg extract-diagnosis "${common[@]}" --model "$model" --device "$DEVICE"
  "$interpreter" -m iesseeg diagnosis "${common[@]}" --model "$model" --device "$DEVICE" --branches frozen finetuned
  "$interpreter" -m iesseeg response "${common[@]}" --model "$model" --device "$DEVICE" --branches frozen finetuned --visits PRE POST
done
for model in biot labram cbramod; do
  "$PYTHON_BIN" -m iesseeg extract-context "${common[@]}" --model "$model" --device "$DEVICE"
done
"$PYTHON_BIN" -m iesseeg context "${common[@]}"
"$PYTHON_BIN" -m iesseeg metrics "${common[@]}"
