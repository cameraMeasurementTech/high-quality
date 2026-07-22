#!/usr/bin/env bash
# Write pipeline runtime config pointing at merged LoRA weights and print restart command.
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"

MERGED="${MERGED:-data/checkpoints/merged_shiny_coder}"
MERGED_ABS="$TRAINING_ROOT/$MERGED"

if [[ ! -d "$MERGED_ABS" ]]; then
  echo "ERROR: merged model dir not found: $MERGED_ABS" >&2
  echo "Run ./run/04_merge_and_eval.sh first" >&2
  exit 1
fi

export MODEL_PATH="$MERGED_ABS"
mkdir -p "$PIPELINE_DIR/runs"

echo "==> Deploy merged coder to shiny-guide native pipeline"
echo "    MODEL_PATH=$MODEL_PATH"
echo ""
echo "Restart pipeline (stop old run-native.sh first):"
echo "  export MODEL_PATH=$MODEL_PATH"
echo "  $PIPELINE_DIR/run-native.sh"
echo ""
echo "Then eval:"
echo "  $PIPELINE_DIR/run-eval.sh data/splits/duel.txt --limit 50 --name merged"
