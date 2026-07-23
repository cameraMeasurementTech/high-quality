#!/usr/bin/env bash
# DPO dataset with production multiview duel scoring (S1–S4 + DINO + AB/BA judge).
#
# Diversity for the 2 JS codes (same coder model):
#   SAME image + SAME coder prompts + SAME temperature (pipeline yaml, ~0.6)
#   DIFFERENT seeds only (sample_0 seed=42…, sample_1 seed=1042…)
#   → mirrors production multigen (seed+k at fixed ensemble_temperature)
#
# Flow:
#   1. splits + images
#   2. Generate DPO_SAMPLES=2 JS/stem via AstroWolf pipeline (:10006)
#   3. Score sample_0 vs sample_1 (validator-like multiview duel)
#   4. Pack winner=chosen, loser=rejected → HF DPO dataset
#
# 4× H200 (recommended):
#   ./run/00_configure_profile.sh h200x4-dpo-duel
#   Phase A (GPUs 0–3 = vLLM TP=4):
#     ./pipeline/start-native-bg.sh && ./pipeline/wait-ready.sh
#     SKIP_DUEL_SCORE=1 SKIP_PACK=1 ./run/01_prepare_dpo_duel_scored.sh
#     ./pipeline/stop-native.sh
#   Phase B (OpenRouter + Chromium; DINO on cuda:0):
#     SKIP_COLLECT=1 ./run/01_prepare_dpo_duel_scored.sh
#   Phase C (train, NUM_PROCESSES=4):
#     CONFIG=configs/dpo_shiny_27b_duel.yaml ./run/03_dpo.sh
#
# Env:
#   TRAIN_N VAL_N DUEL_N SEED DPO_SAMPLES=2
#   BATCH_SIZE=96              (match max_num_seqs; Phase A is JS-only / skip_render)
#   SIDECAR_COUNT=16           (Chromium render farm for Phase B scoring)
#   DUEL_CONCURRENCY=8         (parallel stems while scoring; A∥B render per stem)
#   PIPELINE_URL JUDGE_CONFIG
#   SKIP_COLLECT=1 SKIP_DUEL_SCORE=1 SKIP_PACK=1
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"
cd "$TRAINING_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

TRAIN_N="${TRAIN_N:-5000}"
VAL_N="${VAL_N:-300}"
DUEL_N="${DUEL_N:-200}"
SEED="${SEED:-7}"
DPO_SAMPLES="${DPO_SAMPLES:-2}"
PIPELINE_URL="${PIPELINE_URL:-http://127.0.0.1:10006}"
CAND_DIR="${CAND_DIR:-$TRAINING_ROOT/data/candidates/shiny_k${DPO_SAMPLES}}"
DUEL_JSON="${DUEL_JSON:-$TRAINING_ROOT/data/duel_scores/candidate_duels.json}"
JUDGE_CONFIG="${JUDGE_CONFIG:-$PIPELINE_DIR/configuration.duel-judge.yaml}"
DUEL_LIMIT="${DUEL_LIMIT:-0}"
BATCH_SIZE="${BATCH_SIZE:-96}"
SIDECAR_COUNT="${SIDECAR_COUNT:-16}"
DUEL_CONCURRENCY="${DUEL_CONCURRENCY:-8}"
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
if [[ ! -d "$TRAINING_ROOT/data/images" ]] || [[ -z "$(ls -A "$TRAINING_ROOT/data/images"/*.png 2>/dev/null || true)" ]]; then
  cat "$TRAINING_ROOT/data/splits/train.txt" "$TRAINING_ROOT/data/splits/val.txt" > "$TRAINING_ROOT/data/splits/train_val.txt"
  python "$SCRIPTS/download_images.py" \
    --list "$TRAINING_ROOT/data/splits/train_val.txt" \
    --out "$TRAINING_ROOT/data/images" \
    --workers "${DL_WORKERS:-32}"
fi

echo "==> [2/4] Collect $DPO_SAMPLES JS/stem (diversity=seed, batch=$BATCH_SIZE) via $PIPELINE_URL"
if [[ "${SKIP_COLLECT:-0}" != "1" ]]; then
  echo "    Same prompt/temp; different seeds (sample_0 vs sample_1)."
  echo "    Pipeline should use skip_render=true (JS only; render in step 3)."
  echo "    Pipeline must be running: ./pipeline/start-native-bg.sh"
  python "$SCRIPTS/collect_candidates.py" --from-pipeline \
    --list "$TRAINING_ROOT/data/splits/train.txt" \
    --base-url "$PIPELINE_URL" \
    --samples "$DPO_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --out "$CAND_DIR"
else
  echo "    SKIP_COLLECT=1"
fi

echo "==> [3/4] Multiview duel score (S1–S4 + DINO + OpenRouter)"
if [[ "${SKIP_DUEL_SCORE:-0}" != "1" ]]; then
  echo "    Stop coder vLLM first on shared boxes: ./pipeline/stop-native.sh"
  LIMIT_ARGS=()
  [[ -n "$DUEL_LIMIT" && "$DUEL_LIMIT" != "0" ]] && LIMIT_ARGS=(--limit "$DUEL_LIMIT")
  python "$SCRIPTS/duel_score_candidates.py" \
    --candidates-dir "$CAND_DIR" \
    --images "$TRAINING_ROOT/data/images" \
    --list "$TRAINING_ROOT/data/splits/train.txt" \
    --out "$DUEL_JSON" \
    --sidecar-count "$SIDECAR_COUNT" \
    --concurrency "$DUEL_CONCURRENCY" \
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
echo "    Train: CONFIG=configs/dpo_shiny_27b_duel.yaml NUM_PROCESSES=\${NUM_PROCESSES:-4} ./run/03_dpo.sh"
echo "    Docs:  docs/DPO_DUEL_SCORING.md"
