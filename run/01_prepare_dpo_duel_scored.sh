#!/usr/bin/env bash
# DPO dataset with production multiview duel scoring (S1–S4 + DINO + AB/BA judge).
#
# Flow:
#   1. splits + images (if missing)
#   2. Generate 2 JS candidates per stem via shiny-guide pipeline (:10006)
#   3. Score sample_0 vs sample_1 with OpenRouter judge + multiview render
#   4. Pack DPO pairs (winner=chosen, loser=rejected)
#
# Requires:
#   - Pipeline up for step 2 only (OPENROUTER + GPU AstroWolf)
#   - Step 3: OPENROUTER + Node/Chromium sidecars (stop pipeline first on small boxes)
#
# Env:
#   TRAIN_N VAL_N DUEL_N SEED
#   DPO_SAMPLES=2              (default: 2 codes per prompt)
#   PIPELINE_URL=http://127.0.0.1:10006
#   DUEL_JSON=data/duel_scores/candidate_duels.json
#   JUDGE_CONFIG=pipeline/configuration.duel-judge.yaml
#   SKIP_COLLECT=1 SKIP_DUEL_SCORE=1   run subsets (split across GPU phases)
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"
cd "$TRAINING_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

TRAIN_N="${TRAIN_N:-3000}"
VAL_N="${VAL_N:-200}"
DUEL_N="${DUEL_N:-200}"
SEED="${SEED:-7}"
DPO_SAMPLES="${DPO_SAMPLES:-2}"
PIPELINE_URL="${PIPELINE_URL:-http://127.0.0.1:10006}"
CAND_DIR="${CAND_DIR:-$TRAINING_ROOT/data/candidates/shiny_k${DPO_SAMPLES}}"
DUEL_JSON="${DUEL_JSON:-$TRAINING_ROOT/data/duel_scores/candidate_duels.json}"
JUDGE_CONFIG="${JUDGE_CONFIG:-$PIPELINE_DIR/configuration.duel-judge.yaml}"
DUEL_LIMIT="${DUEL_LIMIT:-0}"
SCRIPTS="$TRAINING_ROOT/scripts"

export SHINY_GUIDE_ROOT
export CONFIG_FILE="$JUDGE_CONFIG"
export PYTHONPATH="$SHINY_GUIDE_ROOT/pipeline_service:$TRAINING_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY for multiview judge scoring}"

echo "==> [1/4] Splits + images (TRAIN_N=$TRAIN_N)"
if [[ ! -f "$TRAINING_ROOT/data/splits/train.txt" ]]; then
  python "$SCRIPTS/prepare_splits.py" \
    --pool "$PROMPTS_POOL" \
    --train "$TRAIN_N" --val "$VAL_N" --duel "$DUEL_N" \
    --seed "$SEED" \
    --out-dir "$TRAINING_ROOT/data/splits"
fi
if [[ ! -d "$TRAINING_ROOT/data/images" ]] || [[ -z "$(ls -A "$TRAINING_ROOT/data/images"/*.png 2>/dev/null)" ]]; then
  cat "$TRAINING_ROOT/data/splits/train.txt" "$TRAINING_ROOT/data/splits/val.txt" > "$TRAINING_ROOT/data/splits/train_val.txt"
  python "$SCRIPTS/download_images.py" \
    --list "$TRAINING_ROOT/data/splits/train_val.txt" \
    --out "$TRAINING_ROOT/data/images" \
    --workers "${DL_WORKERS:-16}"
fi

echo "==> [2/4] Collect $DPO_SAMPLES JS candidates/stem via pipeline ($PIPELINE_URL)"
if [[ "${SKIP_COLLECT:-0}" != "1" ]]; then
  echo "    (Pipeline must be running — ./pipeline/start-native-bg.sh)"
  python "$SCRIPTS/collect_candidates.py" --from-pipeline \
    --list "$TRAINING_ROOT/data/splits/train.txt" \
    --base-url "$PIPELINE_URL" \
    --samples "$DPO_SAMPLES" \
    --out "$CAND_DIR"
else
  echo "    SKIP_COLLECT=1"
fi

echo "==> [3/4] Multiview duel score (OpenRouter judge S1–S4)"
if [[ "${SKIP_DUEL_SCORE:-0}" != "1" ]]; then
  echo "    Stop pipeline on 2× GPU boxes before this step (./pipeline/stop-native.sh)"
  LIMIT_ARGS=()
  [[ -n "$DUEL_LIMIT" && "$DUEL_LIMIT" != "0" ]] && LIMIT_ARGS=(--limit "$DUEL_LIMIT")
  python "$SCRIPTS/duel_score_candidates.py" \
    --candidates-dir "$CAND_DIR" \
    --images "$TRAINING_ROOT/data/images" \
    --list "$TRAINING_ROOT/data/splits/train.txt" \
    --out "$DUEL_JSON" \
    "${LIMIT_ARGS[@]}"
else
  echo "    SKIP_DUEL_SCORE=1"
fi

echo "==> [4/4] Pack DPO dataset (duel-scored pairs)"
if [[ "${SKIP_PACK:-0}" != "1" ]]; then
  python "$SCRIPTS/pack_dpo_dataset.py" \
    --source duel-scored \
    --duel-json "$DUEL_JSON" \
    --list "$TRAINING_ROOT/data/splits/train.txt" \
    --images "$TRAINING_ROOT/data/images" \
    --out "$TRAINING_ROOT/data/hf/dpo_shiny_duel"
else
  echo "    SKIP_PACK=1"
fi

echo ""
echo "==> Done: $TRAINING_ROOT/data/hf/dpo_shiny_duel/dataset"
echo "    Train: CONFIG=configs/dpo_shiny_27b.yaml \\"
echo "           # edit dataset_path to data/hf/dpo_shiny_duel/dataset"
echo "           ./run/03_dpo.sh"
