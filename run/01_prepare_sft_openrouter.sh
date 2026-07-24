#!/usr/bin/env bash
# Prepare SFT dataset starting from prompts.txt (~99k image URLs) — same pool as
# duel/DPO prep (PROMPTS_POOL → data/prompts.txt after bootstrap).
#
# Default: FULL_POOL=1 → train on (almost) the entire prompts.txt
#   hold out VAL_N + DUEL_N, remainder → train (~98.9k stems)
#
# Flow:
#   data/prompts.txt
#     → prepare_splits (train/val/duel)
#     → download PNGs
#     → OpenRouter GPT-5 with miner coder prompts
#     → validate.js → pack HF SFT
#
# Smoke (subsample from the same prompts.txt):
#   FULL_POOL=0 TRAIN_N=50 ./run/01_prepare_sft_openrouter.sh
#
# Train LoRA (4× H200):
#   CONFIG=configs/sft_shiny_27b_gpt_teacher.yaml NUM_PROCESSES=4 ./run/02_sft.sh
#
# Env:
#   FULL_POOL=1              # default — entire prompts.txt minus holdouts
#   TRAIN_N                  # only when FULL_POOL=0 (subsample for smoke)
#   VAL_N DUEL_N SEED FORCE_SPLITS
#   TEACHER_MODEL TEACHER_WORKERS …
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"
cd "$TRAINING_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY in $TRAINING_ROOT/.env}"

# Default: use the full prompts.txt pool (same as duel pipeline source).
FULL_POOL="${FULL_POOL:-1}"
VAL_N="${VAL_N:-500}"
DUEL_N="${DUEL_N:-200}"
SEED="${SEED:-7}"
TEACHER_MODEL="${TEACHER_MODEL:-openai/gpt-5-chat}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-https://openrouter.ai/api/v1}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.4}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-24576}"
TEACHER_WORKERS="${TEACHER_WORKERS:-16}"
DL_WORKERS="${DL_WORKERS:-64}"
VAL_WORKERS="${VAL_WORKERS:-16}"

RAW_JS="${RAW_JS:-$TRAINING_ROOT/data/raw_js/gpt_teacher}"
FILT_JS="${FILT_JS:-$TRAINING_ROOT/data/filtered_js/gpt_teacher}"
MINER_IN="${MINER_IN:-$TRAINING_ROOT/data/miner_inputs/sft}"
OUT_SFT="${OUT_SFT:-$TRAINING_ROOT/data/hf/sft_gpt_teacher}"
SCRIPTS="$TRAINING_ROOT/scripts"

require_prompts_pool
pool_lines="$(wc -l < "$PROMPTS_POOL" | tr -d ' ')"
echo "==> Starting from prompts pool (same as duel/DPO pipeline):"
echo "    PROMPTS_POOL=$PROMPTS_POOL"
echo "    lines=$pool_lines  (expect ~99610)"

if [[ "$pool_lines" -lt 50000 ]] && [[ "${ALLOW_SMALL_POOL:-0}" != "1" ]]; then
  echo "ERROR: prompts pool looks too small ($pool_lines lines)." >&2
  echo "  Expected data/prompts.txt from ./run/00_bootstrap_assets.sh (~99k)." >&2
  echo "  Or set ALLOW_SMALL_POOL=1 for intentional tiny pools." >&2
  exit 1
fi

# Full-pool: do not dump ~99k request JSON by default (disk bomb).
if [[ "$FULL_POOL" == "1" ]]; then
  SKIP_EXPORT="${SKIP_EXPORT:-1}"
  FORCE_SPLITS="${FORCE_SPLITS:-1}"
  TRAIN_N="${TRAIN_N:-0}"  # ignored when --train-remainder
else
  TRAIN_N="${TRAIN_N:-5000}"
  SKIP_EXPORT="${SKIP_EXPORT:-0}"
  echo "    FULL_POOL=0 → subsample TRAIN_N=$TRAIN_N from the same prompts.txt"
fi

echo "=============================================="
echo "  SFT prep — OpenRouter GPT teacher (miner prompts)"
echo "  pool=$PROMPTS_POOL ($pool_lines lines)"
echo "  TEACHER_MODEL=$TEACHER_MODEL"
echo "  FULL_POOL=$FULL_POOL  TRAIN_N=${TRAIN_N:-remainder}  VAL_N=$VAL_N  DUEL_N=$DUEL_N"
echo "  TEACHER_WORKERS=$TEACHER_WORKERS  DL_WORKERS=$DL_WORKERS"
echo "  OUT_SFT=$OUT_SFT"
echo "=============================================="

echo "==> [1/6] Splits from prompts.txt pool"
if [[ ! -f "$TRAINING_ROOT/data/splits/train.txt" ]] || [[ "${FORCE_SPLITS:-0}" == "1" ]]; then
  SPLIT_ARGS=(
    --pool "$PROMPTS_POOL"
    --val "$VAL_N" --duel "$DUEL_N"
    --seed "$SEED"
    --out-dir "$TRAINING_ROOT/data/splits"
  )
  if [[ "$FULL_POOL" == "1" ]]; then
    echo "    mode=train-remainder (all of prompts.txt except val+duel holdouts)"
    python "$SCRIPTS/prepare_splits.py" "${SPLIT_ARGS[@]}" --train-remainder
  else
    echo "    mode=subsample train=$TRAIN_N from prompts.txt"
    python "$SCRIPTS/prepare_splits.py" "${SPLIT_ARGS[@]}" --train "$TRAIN_N"
  fi
else
  echo "    using existing data/splits/train.txt (FORCE_SPLITS=1 to rebuild from prompts.txt)"
fi
train_lines="$(wc -l < "$TRAINING_ROOT/data/splits/train.txt" | tr -d ' ')"
echo "    train stems: $train_lines  (sourced from $PROMPTS_POOL)"

echo "==> [2/6] Download reference images (train+val)"
cat "$TRAINING_ROOT/data/splits/train.txt" "$TRAINING_ROOT/data/splits/val.txt" \
  > "$TRAINING_ROOT/data/splits/train_val.txt"
python "$SCRIPTS/download_images.py" \
  --list "$TRAINING_ROOT/data/splits/train_val.txt" \
  --out "$TRAINING_ROOT/data/images" \
  --workers "$DL_WORKERS" \
  --fail-ok

echo "==> [3/6] Export miner-identical coder inputs"
if [[ "${SKIP_EXPORT:-0}" != "1" ]]; then
  python "$SCRIPTS/export_miner_coder_inputs.py" \
    --list "$TRAINING_ROOT/data/splits/train.txt" \
    --images "$TRAINING_ROOT/data/images" \
    --out "$MINER_IN" \
    --model "$TEACHER_MODEL" \
    --temperature "$TEACHER_TEMPERATURE" \
    --max-tokens "$TEACHER_MAX_TOKENS" \
    --limit "${EXPORT_LIMIT:-20}"
  echo "    wrote sample inputs under $MINER_IN (limit=${EXPORT_LIMIT:-20})"
else
  echo "    SKIP_EXPORT=1 (recommended for FULL_POOL; prompts still used at teacher call time)"
  # Still write a tiny sample for parity checks
  if [[ ! -f "$MINER_IN/_prompt_meta.json" ]]; then
    python "$SCRIPTS/export_miner_coder_inputs.py" \
      --list "$TRAINING_ROOT/data/splits/train.txt" \
      --images "$TRAINING_ROOT/data/images" \
      --out "$MINER_IN" \
      --model "$TEACHER_MODEL" \
      --limit 5 || true
  fi
fi

echo "==> [4/6] Teacher JS via OpenRouter ($TEACHER_MODEL)"
if [[ "${SKIP_TEACHER:-0}" != "1" ]]; then
  python "$SCRIPTS/collect_teacher_js.py" --from-openai \
    --list "$TRAINING_ROOT/data/splits/train.txt" \
    --images "$TRAINING_ROOT/data/images" \
    --base-url "$TEACHER_BASE_URL" \
    --model "$TEACHER_MODEL" \
    --temperature "$TEACHER_TEMPERATURE" \
    --max-tokens "$TEACHER_MAX_TOKENS" \
    --workers "$TEACHER_WORKERS" \
    --save-request \
    --out "$RAW_JS"
else
  echo "    SKIP_TEACHER=1"
fi

echo "==> [5/6] validate.js filter"
if [[ "${SKIP_VALIDATE:-0}" != "1" ]]; then
  python "$SCRIPTS/filter_validate.py" \
    --js-dir "$RAW_JS" \
    --out-dir "$FILT_JS" \
    --workers "$VAL_WORKERS" \
    --report "$FILT_JS/validate_report.json"
else
  echo "    SKIP_VALIDATE=1"
fi

echo "==> [6/6] Pack HF SFT dataset (miner prompts + teacher JS)"
if [[ "${SKIP_PACK:-0}" != "1" ]]; then
  python "$SCRIPTS/pack_sft_dataset.py" \
    --js-dir "$FILT_JS" \
    --images "$TRAINING_ROOT/data/images" \
    --list "$TRAINING_ROOT/data/splits/train.txt" \
    --out "$OUT_SFT" \
    --source-tag "openrouter:${TEACHER_MODEL}"
else
  echo "    SKIP_PACK=1"
fi

n_ds=0
if [[ -d "$OUT_SFT/dataset" ]]; then
  n_ds="$(python - <<PY
from datasets import load_from_disk
print(len(load_from_disk("$OUT_SFT/dataset")))
PY
)"
fi

filt_n="$(find "$FILT_JS" -maxdepth 1 -name '*.js' 2>/dev/null | wc -l | tr -d ' ')"
raw_n="$(find "$RAW_JS" -maxdepth 1 -name '*.js' 2>/dev/null | wc -l | tr -d ' ')"

echo ""
echo "==> Done"
echo "    train stems  : $train_lines"
echo "    raw teacher  : $RAW_JS  (n≈$raw_n)"
echo "    filtered JS  : $FILT_JS  (n≈$filt_n)"
echo "    HF dataset   : $OUT_SFT/dataset  (n=$n_ds)"
echo ""
echo "Train LoRA on AstroWolf (4× H200 recommended):"
echo "  CONFIG=configs/sft_shiny_27b_gpt_teacher.yaml NUM_PROCESSES=4 ./run/02_sft.sh"
echo "Docs: docs/SFT_GPT_TEACHER.md"
