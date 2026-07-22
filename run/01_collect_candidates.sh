#!/usr/bin/env bash
# Collect K JS candidates per stem for score-based DPO pair mining.
#
# Env:
#   MODE=pipeline|openai          (default: openai)
#   SAMPLES=4
#   TEMPERATURES=0.5,0.7,0.9,1.0
#   PIPELINE_URL=http://127.0.0.1:10006
#   TEACHER_MODEL=google/gemini-2.5-pro-preview
#   OUT=data/candidates/openai_k4
#   LIMIT=500                     (optional: subsample train list)
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"
cd "$TRAINING_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

MODE="${MODE:-openai}"
SAMPLES="${SAMPLES:-4}"
TEMPERATURES="${TEMPERATURES:-0.5,0.7,0.9,1.0}"
PIPELINE_URL="${PIPELINE_URL:-http://127.0.0.1:10006}"
TEACHER_MODEL="${TEACHER_MODEL:-google/gemini-2.5-pro-preview}"
OUT="${OUT:-$TRAINING_ROOT/data/candidates/openai_k4}"
LIMIT="${LIMIT:-}"

SCRIPTS="$TRAINING_ROOT/scripts"

LIST="$TRAINING_ROOT/data/splits/train.txt"
if [[ ! -f "$LIST" ]]; then
  echo "Missing $LIST — run ./run/01_prepare_data.sh first." >&2
  exit 1
fi

WORK_LIST="$LIST"
if [[ -n "$LIMIT" ]]; then
  WORK_LIST="$TRAINING_ROOT/data/splits/train_limit_${LIMIT}.txt"
  head -n "$LIMIT" "$LIST" > "$WORK_LIST"
fi

mkdir -p "$(dirname "$OUT")"

echo "==> Collecting $SAMPLES candidates/stem (mode=$MODE) -> $OUT"

case "$MODE" in
  pipeline)
    python "$SCRIPTS/collect_candidates.py" --from-pipeline \
      --list "$WORK_LIST" \
      --base-url "$PIPELINE_URL" \
      --samples "$SAMPLES" \
      --out "$OUT"
    ;;
  openai)
    : "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY for MODE=openai}"
    python "$SCRIPTS/collect_candidates.py" --from-openai \
      --list "$WORK_LIST" \
      --images "$TRAINING_ROOT/data/images" \
      --base-url "${TEACHER_BASE_URL:-https://openrouter.ai/api/v1}" \
      --model "$TEACHER_MODEL" \
      --samples "$SAMPLES" \
      --temperatures "$TEMPERATURES" \
      --out "$OUT"
    ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 1
    ;;
esac

echo "==> Candidates collected. Next:"
echo "    SOURCE=candidates ./run/01_prepare_dpo_data.sh"
