#!/usr/bin/env bash
# Build DPO preference dataset (offline pairs).
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"
cd "$TRAINING_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

SOURCE="${SOURCE:-candidates}"
DUEL_JSON="${DUEL_JSON:-$PIPELINE_DIR/runs/duel/duel_detailed.json}"
PREFER_LABEL="${PREFER_LABEL:-shiny-guide}"
ONLY_LOSSES="${ONLY_LOSSES:-1}"
CANDIDATES_DIR="${CANDIDATES_DIR:-$TRAINING_ROOT/data/candidates/shiny_k4}"
CHOSEN_DIR="${CHOSEN_DIR:-$TRAINING_ROOT/data/filtered_js}"
REJECTED_DIR="${REJECTED_DIR:-$TRAINING_ROOT/data/raw_js/teacher}"
REWARD_MODE="${REWARD_MODE:-cheap}"
MIN_MARGIN="${MIN_MARGIN:-0.15}"
OUT="${OUT:-$TRAINING_ROOT/data/hf/dpo_train}"

SCRIPTS="$TRAINING_ROOT/scripts"
LIST="$TRAINING_ROOT/data/splits/train.txt"

if [[ ! -f "$LIST" ]]; then
  echo "Missing $LIST — run ./run/01_prepare_shiny_align.sh first." >&2
  exit 1
fi

echo "==> DPO dataset prep (source=$SOURCE -> $OUT)"

case "$SOURCE" in
  duel)
    if [[ ! -f "$DUEL_JSON" ]]; then
      echo "Missing duel JSON: $DUEL_JSON" >&2
      exit 1
    fi
    DUEL_LIST="$TRAINING_ROOT/data/splits/duel.txt"
    LIST_ARG=()
    [[ -f "$DUEL_LIST" ]] && LIST_ARG=(--list "$DUEL_LIST")
    EXTRA=()
    [[ -n "$PREFER_LABEL" ]] && EXTRA+=(--prefer-label "$PREFER_LABEL")
    [[ "$ONLY_LOSSES" == "1" ]] && EXTRA+=(--only-losses)
    python "$SCRIPTS/pack_dpo_dataset.py" \
      --source duel \
      --duel-json "$DUEL_JSON" \
      "${LIST_ARG[@]}" \
      --images "$TRAINING_ROOT/data/images" \
      --out "$OUT" \
      "${EXTRA[@]}"
    ;;
  candidates)
    python "$SCRIPTS/pack_dpo_dataset.py" \
      --source candidates \
      --candidates-dir "$CANDIDATES_DIR" \
      --list "$LIST" \
      --images "$TRAINING_ROOT/data/images" \
      --out "$OUT" \
      --reward-mode "$REWARD_MODE" \
      --min-margin "$MIN_MARGIN"
    ;;
  dirs)
    python "$SCRIPTS/pack_dpo_dataset.py" \
      --source dirs \
      --chosen-dir "$CHOSEN_DIR" \
      --rejected-dir "$REJECTED_DIR" \
      --list "$LIST" \
      --images "$TRAINING_ROOT/data/images" \
      --out "$OUT" \
      --reward-mode "$REWARD_MODE" \
      --min-margin "$MIN_MARGIN"
    ;;
  *)
    echo "Unknown SOURCE=$SOURCE" >&2
    exit 1
    ;;
esac

echo "==> DPO dataset ready: $OUT/dataset"
