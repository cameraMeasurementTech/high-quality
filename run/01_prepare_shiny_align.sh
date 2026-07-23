#!/usr/bin/env bash
# Prepare alignment data for shiny-guide (DPO and/or GRPO) — NO SFT.
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"
cd "$TRAINING_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

ALIGN="${ALIGN:-both}"
TRAIN_N="${TRAIN_N:-5000}"
VAL_N="${VAL_N:-300}"
DUEL_N="${DUEL_N:-200}"
SEED="${SEED:-7}"
PIPELINE_URL="${PIPELINE_URL:-http://127.0.0.1:10006}"
DPO_SAMPLES="${DPO_SAMPLES:-4}"
PREP_SFT="${PREP_SFT:-0}"
SCRIPTS="$TRAINING_ROOT/scripts"

echo "==> Alignment prep (ALIGN=$ALIGN, PROMPTS_POOL=$PROMPTS_POOL)"
require_prompts_pool

python "$SCRIPTS/prepare_splits.py" \
  --pool "$PROMPTS_POOL" \
  --train "$TRAIN_N" --val "$VAL_N" --duel "$DUEL_N" \
  --seed "$SEED" \
  --out-dir "$TRAINING_ROOT/data/splits"

cat "$TRAINING_ROOT/data/splits/train.txt" "$TRAINING_ROOT/data/splits/val.txt" > "$TRAINING_ROOT/data/splits/train_val.txt"
python "$SCRIPTS/download_images.py" \
  --list "$TRAINING_ROOT/data/splits/train_val.txt" \
  --out "$TRAINING_ROOT/data/images" \
  --workers "${DL_WORKERS:-16}"

if [[ "$PREP_SFT" == "1" ]]; then
  RAW="$TRAINING_ROOT/data/raw_js/shiny-guide"
  mkdir -p "$RAW"
  python "$SCRIPTS/collect_teacher_js.py" --from-pipeline \
    --list "$TRAINING_ROOT/data/splits/train.txt" \
    --base-url "$PIPELINE_URL" \
    --out "$RAW" \
    --batch-size "${BATCH_SIZE:-16}"
  python "$SCRIPTS/filter_validate.py" \
    --js-dir "$RAW" \
    --out-dir "$TRAINING_ROOT/data/filtered_js/shiny-guide" \
    --workers "${VAL_WORKERS:-8}" \
    --copy-fail
  python "$SCRIPTS/pack_sft_dataset.py" \
    --js-dir "$TRAINING_ROOT/data/filtered_js/shiny-guide" \
    --images "$TRAINING_ROOT/data/images" \
    --list "$TRAINING_ROOT/data/splits/train.txt" \
    --out "$TRAINING_ROOT/data/hf/sft_shiny" \
    --source-tag shiny-guide
fi

if [[ "$ALIGN" == "dpo" || "$ALIGN" == "both" ]]; then
  CAND="$TRAINING_ROOT/data/candidates/shiny_k${DPO_SAMPLES}"
  python "$SCRIPTS/collect_candidates.py" --from-pipeline \
    --list "$TRAINING_ROOT/data/splits/train.txt" \
    --base-url "$PIPELINE_URL" \
    --samples "$DPO_SAMPLES" \
    --out "$CAND"

  python "$SCRIPTS/pack_dpo_dataset.py" \
    --source candidates \
    --candidates-dir "$CAND" \
    --list "$TRAINING_ROOT/data/splits/train.txt" \
    --images "$TRAINING_ROOT/data/images" \
    --reward-mode "${REWARD_MODE:-cheap}" \
    --min-margin "${MIN_MARGIN:-0.15}" \
    --out "$TRAINING_ROOT/data/hf/dpo_shiny"
fi

if [[ "$ALIGN" == "grpo" || "$ALIGN" == "both" ]]; then
  python "$SCRIPTS/pack_grpo_prompts.py" \
    --list "$TRAINING_ROOT/data/splits/train.txt" \
    --images "$TRAINING_ROOT/data/images" \
    --out "$TRAINING_ROOT/data/hf/grpo_shiny"
fi

echo "==> Done."
[[ "$ALIGN" == "dpo" || "$ALIGN" == "both" ]] && echo "    DPO:  $TRAINING_ROOT/data/hf/dpo_shiny/dataset"
[[ "$ALIGN" == "grpo" || "$ALIGN" == "both" ]] && echo "    GRPO: $TRAINING_ROOT/data/hf/grpo_shiny/dataset"
