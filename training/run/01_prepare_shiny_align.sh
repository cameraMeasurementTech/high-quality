#!/usr/bin/env bash
# Prepare alignment data for shiny-guide (DPO and/or GRPO) — NO SFT.
#
# AstroWolf is already specialized; skip SFT and go straight to DPO or GRPO.
#
# Env:
#   ALIGN=dpo|grpo|both          (default: both)
#   TRAIN_N=5000 VAL_N=300 DUEL_N=200 SEED=7
#   PIPELINE_URL=http://127.0.0.1:10006
#   DPO_SAMPLES=4
#   REWARD_MODE=cheap|s1
#   PREP_SFT=1                   optional: also pack SFT dataset (usually off)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

export PROMPTS_ROOT=shiny-guide
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

ALIGN="${ALIGN:-both}"
TRAIN_N="${TRAIN_N:-5000}"
VAL_N="${VAL_N:-300}"
DUEL_N="${DUEL_N:-200}"
SEED="${SEED:-7}"
PIPELINE_URL="${PIPELINE_URL:-http://127.0.0.1:10006}"
DPO_SAMPLES="${DPO_SAMPLES:-4}"
PREP_SFT="${PREP_SFT:-0}"

SCRIPTS="$ROOT/scripts"

echo "==> Alignment data prep (ALIGN=$ALIGN, PREP_SFT=$PREP_SFT, no SFT training step)"

echo "==> [1] Splits from validator pool"
python "$SCRIPTS/prepare_splits.py" \
  --pool "$REPO/prompts.txt" \
  --train "$TRAIN_N" --val "$VAL_N" --duel "$DUEL_N" \
  --seed "$SEED" \
  --out-dir "$ROOT/data/splits"

echo "==> [2] Download images"
cat "$ROOT/data/splits/train.txt" "$ROOT/data/splits/val.txt" > "$ROOT/data/splits/train_val.txt"
python "$SCRIPTS/download_images.py" \
  --list "$ROOT/data/splits/train_val.txt" \
  --out "$ROOT/data/images" \
  --workers "${DL_WORKERS:-16}"

if [[ "$PREP_SFT" == "1" ]]; then
  echo "==> [optional] SFT dataset pack (PREP_SFT=1)"
  RAW="$ROOT/data/raw_js/shiny-guide"
  mkdir -p "$RAW"
  python "$SCRIPTS/collect_teacher_js.py" --from-pipeline \
    --list "$ROOT/data/splits/train.txt" \
    --base-url "$PIPELINE_URL" \
    --out "$RAW" \
    --batch-size "${BATCH_SIZE:-16}"
  python "$SCRIPTS/filter_validate.py" \
    --js-dir "$RAW" \
    --out-dir "$ROOT/data/filtered_js/shiny-guide" \
    --workers "${VAL_WORKERS:-8}" \
    --copy-fail
  python "$SCRIPTS/pack_sft_dataset.py" \
    --js-dir "$ROOT/data/filtered_js/shiny-guide" \
    --images "$ROOT/data/images" \
    --list "$ROOT/data/splits/train.txt" \
    --out "$ROOT/data/hf/sft_shiny" \
    --source-tag shiny-guide
fi

if [[ "$ALIGN" == "dpo" || "$ALIGN" == "both" ]]; then
  echo "==> [DPO] Collect K=$DPO_SAMPLES candidates/stem (shiny-guide on $PIPELINE_URL)"
  CAND="$ROOT/data/candidates/shiny_k${DPO_SAMPLES}"
  python "$SCRIPTS/collect_candidates.py" --from-pipeline \
    --list "$ROOT/data/splits/train.txt" \
    --base-url "$PIPELINE_URL" \
    --samples "$DPO_SAMPLES" \
    --out "$CAND"

  echo "==> [DPO] Pack preference pairs"
  python "$SCRIPTS/pack_dpo_dataset.py" \
    --source candidates \
    --candidates-dir "$CAND" \
    --list "$ROOT/data/splits/train.txt" \
    --images "$ROOT/data/images" \
    --reward-mode "${REWARD_MODE:-cheap}" \
    --min-margin "${MIN_MARGIN:-0.15}" \
    --out "$ROOT/data/hf/dpo_shiny"
fi

if [[ "$ALIGN" == "grpo" || "$ALIGN" == "both" ]]; then
  echo "==> [GRPO] Pack prompt-only dataset (completions sampled during training)"
  python "$SCRIPTS/pack_grpo_prompts.py" \
    --list "$ROOT/data/splits/train.txt" \
    --images "$ROOT/data/images" \
    --out "$ROOT/data/hf/grpo_shiny"
fi

echo "==> Done."
[[ "$ALIGN" == "dpo" || "$ALIGN" == "both" ]] && echo "    DPO:  $ROOT/data/hf/dpo_shiny/dataset"
[[ "$ALIGN" == "grpo" || "$ALIGN" == "both" ]] && echo "    GRPO: $ROOT/data/hf/grpo_shiny/dataset"
echo ""
echo "Train (no SFT — base = Tooony133/Qwen-3.6-27B-AstroWolf):"
[[ "$ALIGN" == "dpo" || "$ALIGN" == "both" ]] && echo "  CONFIG=configs/dpo_shiny_27b.yaml ./run/03_dpo.sh"
[[ "$ALIGN" == "grpo" || "$ALIGN" == "both" ]] && echo "  CONFIG=configs/grpo_shiny_27b.yaml ./run/03_grpo.sh"
