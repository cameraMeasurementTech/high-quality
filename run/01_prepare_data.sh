#!/usr/bin/env bash
# Step 1 — splits → download images → collect teacher JS → validate → pack HF datasets.
#
# Env knobs:
#   TRAIN_N=10000 VAL_N=500 DUEL_N=200 SEED=7
#   TEACHER_MODE=runs|pipeline|openai   (default: runs if local JS exists, else openai)
#   PIPELINE_URL=http://127.0.0.1:10006
#   TEACHER_MODEL=google/gemini-2.5-pro-preview
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"
cd "$TRAINING_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

TRAIN_N="${TRAIN_N:-10000}"
VAL_N="${VAL_N:-500}"
DUEL_N="${DUEL_N:-200}"
SEED="${SEED:-7}"
TEACHER_MODE="${TEACHER_MODE:-runs}"
PIPELINE_URL="${PIPELINE_URL:-http://127.0.0.1:10006}"
TEACHER_MODEL="${TEACHER_MODEL:-google/gemini-2.5-pro-preview}"

SCRIPTS="$TRAINING_ROOT/scripts"

echo "==> [1/6] Prepare splits (train=$TRAIN_N val=$VAL_N duel=$DUEL_N seed=$SEED)"
python "$SCRIPTS/prepare_splits.py" \
  --pool "$PROMPTS_POOL" \
  --train "$TRAIN_N" --val "$VAL_N" --duel "$DUEL_N" \
  --seed "$SEED" \
  --out-dir "$TRAINING_ROOT/data/splits"

echo "==> [2/6] Download images for train+val (duel images downloaded later for eval)"
cat "$TRAINING_ROOT/data/splits/train.txt" "$TRAINING_ROOT/data/splits/val.txt" > "$TRAINING_ROOT/data/splits/train_val.txt"
python "$SCRIPTS/download_images.py" \
  --list "$TRAINING_ROOT/data/splits/train_val.txt" \
  --out "$TRAINING_ROOT/data/images" \
  --workers "${DL_WORKERS:-16}"

echo "==> [3/6] Collect teacher JS (mode=$TEACHER_MODE)"
RAW="$TRAINING_ROOT/data/raw_js/teacher"
case "$TEACHER_MODE" in
  runs)
    RUN_DIRS=()
    for d in \
      "$PIPELINE_DIR/runs/eval/baseline" \
      "$PIPELINE_DIR/runs/eval/merged" \
      "$WORKSPACE_ROOT/local-eval/runs/duel/shiny-guide" \
      "$WORKSPACE_ROOT/local-eval/runs/pool"
    do
      [[ -d "$d" ]] && RUN_DIRS+=("$d")
    done
    if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
      echo "No eval run dirs found. Set TEACHER_MODE=pipeline or openai." >&2
      exit 1
    fi
    python "$SCRIPTS/collect_teacher_js.py" --from-runs --run-dirs "${RUN_DIRS[@]}" --out "$RAW"
    ;;
  pipeline)
    python "$SCRIPTS/collect_teacher_js.py" --from-pipeline \
      --list "$TRAINING_ROOT/data/splits/train.txt" \
      --base-url "$PIPELINE_URL" \
      --out "$RAW"
    ;;
  openai)
    : "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY for TEACHER_MODE=openai}"
    python "$SCRIPTS/collect_teacher_js.py" --from-openai \
      --list "$TRAINING_ROOT/data/splits/train.txt" \
      --images "$TRAINING_ROOT/data/images" \
      --base-url "${TEACHER_BASE_URL:-https://openrouter.ai/api/v1}" \
      --model "$TEACHER_MODEL" \
      --out "$RAW"
    ;;
  *)
    echo "Unknown TEACHER_MODE=$TEACHER_MODE" >&2
    exit 1
    ;;
esac

echo "==> [4/6] Filter with validate.js"
python "$SCRIPTS/filter_validate.py" \
  --js-dir "$RAW" \
  --out-dir "$TRAINING_ROOT/data/filtered_js" \
  --workers "${VAL_WORKERS:-8}" \
  --copy-fail

echo "==> [5/6] Pack SFT dataset"
python "$SCRIPTS/pack_sft_dataset.py" \
  --js-dir "$TRAINING_ROOT/data/filtered_js" \
  --images "$TRAINING_ROOT/data/images" \
  --list "$TRAINING_ROOT/data/splits/train.txt" \
  --out "$TRAINING_ROOT/data/hf/sft_train" \
  --source-tag "$TEACHER_MODE"

echo "==> [6/6] Pack GRPO prompt dataset"
python "$SCRIPTS/pack_grpo_prompts.py" \
  --list "$TRAINING_ROOT/data/splits/train.txt" \
  --images "$TRAINING_ROOT/data/images" \
  --out "$TRAINING_ROOT/data/hf/grpo_train"

echo "==> Data prep done."
echo "    SFT:  $TRAINING_ROOT/data/hf/sft_train"
echo "    GRPO: $TRAINING_ROOT/data/hf/grpo_train"
echo "    Duel holdout list: $TRAINING_ROOT/data/splits/duel.txt"
