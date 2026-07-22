#!/usr/bin/env bash
# Merge LoRA → deploy to shiny-guide config → eval on held-out duel split.
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"
cd "$TRAINING_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

BASE_MODEL="${BASE_MODEL:-Tooony133/Qwen-3.6-27B-AstroWolf}"
ADAPTER="${ADAPTER:-data/checkpoints/dpo_shiny_27b/final}"
MERGED="${MERGED:-data/checkpoints/merged_shiny_coder}"
DUEL_LIMIT="${DUEL_LIMIT:-50}"
EVAL_LIMIT="${EVAL_LIMIT:-$DUEL_LIMIT}"

if [[ -n "${MODEL_PATH:-}" ]]; then
  BASE_MODEL="$MODEL_PATH"
fi

echo "==> [1/3] Merge LoRA"
python "$TRAINING_ROOT/scripts/merge_lora.py" \
  --base "$BASE_MODEL" \
  --adapter "$TRAINING_ROOT/$ADAPTER" \
  --out "$TRAINING_ROOT/$MERGED"

echo "==> [2/3] Deploy merged weights to pipeline"
export MODEL_PATH="$TRAINING_ROOT/$MERGED"
echo "    Merged model: $MODEL_PATH"
echo "    Restart pipeline with merged weights:"
echo "      export MODEL_PATH=$MODEL_PATH"
echo "      $PIPELINE_DIR/run-native.sh"
echo ""
echo "    Or run: MERGED=$MERGED ./run/05_deploy_merged.sh"

echo "==> [3/3] Eval gate (pipeline must be running on :10006)"
echo "    Baseline (king AstroWolf): unset MODEL_PATH, run pipeline, then:"
echo "      $PIPELINE_DIR/run-eval.sh data/splits/duel.txt --limit $DUEL_LIMIT --name baseline"
echo "    Challenger (merged): export MODEL_PATH=$MODEL_PATH, restart pipeline, then:"
echo "      $PIPELINE_DIR/run-eval.sh data/splits/duel.txt --limit $DUEL_LIMIT --name merged"
echo ""
echo "Ship when merged pass-rate / scores beat baseline on held-out duel.txt (>55–60%)."
