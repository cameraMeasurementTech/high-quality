#!/usr/bin/env bash
# Step 1b — build DPO preference dataset (offline pairs).
#
# Three common paths (pick one SOURCE):
#   SOURCE=duel      Harvest winner/loser pairs from local-eval duel JSON (free, subnet-aligned)
#   SOURCE=candidates Score K samples per stem; best vs worst (needs collect_candidates first)
#   SOURCE=dirs      filtered_js (chosen) vs raw fails (rejected)
#
# Env knobs:
#   SOURCE=duel|candidates|dirs
#   DUEL_JSON=../../local-eval/runs/duel/duel_detailed.json
#   PREFER_LABEL=shiny-guide   # duel: use leader JS as chosen when my-agent lost
#   ONLY_LOSSES=1              # duel: only stems where my-agent lost
#   CANDIDATES_DIR=data/candidates/openai_k4
#   CHOSEN_DIR=data/filtered_js  REJECTED_DIR=data/raw_js/teacher
#   REWARD_MODE=cheap|s1
#   MIN_MARGIN=0.15
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

SOURCE="${SOURCE:-duel}"
DUEL_JSON="${DUEL_JSON:-$REPO/local-eval/runs/duel/duel_detailed.json}"
PREFER_LABEL="${PREFER_LABEL:-shiny-guide}"
ONLY_LOSSES="${ONLY_LOSSES:-1}"
CANDIDATES_DIR="${CANDIDATES_DIR:-$ROOT/data/candidates/openai_k4}"
CHOSEN_DIR="${CHOSEN_DIR:-$ROOT/data/filtered_js}"
REJECTED_DIR="${REJECTED_DIR:-$ROOT/data/raw_js/teacher}"
REWARD_MODE="${REWARD_MODE:-cheap}"
MIN_MARGIN="${MIN_MARGIN:-0.15}"
OUT="${OUT:-$ROOT/data/hf/dpo_train}"

SCRIPTS="$ROOT/scripts"
export PYTHONPATH="$SCRIPTS${PYTHONPATH:+:$PYTHONPATH}"

LIST="$ROOT/data/splits/train.txt"
if [[ ! -f "$LIST" ]]; then
  echo "Missing $LIST — run ./run/01_prepare_data.sh first (or at least prepare_splits + download_images)." >&2
  exit 1
fi

echo "==> DPO dataset prep (source=$SOURCE -> $OUT)"

case "$SOURCE" in
  duel)
    if [[ ! -f "$DUEL_JSON" ]]; then
      echo "Missing duel JSON: $DUEL_JSON" >&2
      echo "Run a local duel first:" >&2
      echo "  $REPO/local-eval/run-duel-seq.sh $ROOT/data/splits/duel.txt --limit 50 --fresh" >&2
      exit 1
    fi
    # Prefer duel holdout list (stems that appear in duel JSON), not train.txt.
    DUEL_LIST="$ROOT/data/splits/duel.txt"
    LIST_ARG=()
    if [[ -f "$DUEL_LIST" ]]; then
      LIST_ARG=(--list "$DUEL_LIST")
    fi
    EXTRA=()
    [[ -n "$PREFER_LABEL" ]] && EXTRA+=(--prefer-label "$PREFER_LABEL")
    [[ "$ONLY_LOSSES" == "1" ]] && EXTRA+=(--only-losses)
    python "$SCRIPTS/pack_dpo_dataset.py" \
      --source duel \
      --duel-json "$DUEL_JSON" \
      "${LIST_ARG[@]}" \
      --images "$ROOT/data/images" \
      --out "$OUT" \
      "${EXTRA[@]}"
    ;;
  candidates)
    if [[ ! -d "$CANDIDATES_DIR" ]]; then
      echo "Missing candidates dir: $CANDIDATES_DIR" >&2
      echo "Collect candidates first, e.g.:" >&2
      echo "  TEACHER_MODE=openai SAMPLES=4 ./run/01_collect_candidates.sh" >&2
      exit 1
    fi
    python "$SCRIPTS/pack_dpo_dataset.py" \
      --source candidates \
      --candidates-dir "$CANDIDATES_DIR" \
      --list "$LIST" \
      --images "$ROOT/data/images" \
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
      --images "$ROOT/data/images" \
      --out "$OUT" \
      --reward-mode "$REWARD_MODE" \
      --min-margin "$MIN_MARGIN"
    ;;
  *)
    echo "Unknown SOURCE=$SOURCE (use duel|candidates|dirs)" >&2
    exit 1
    ;;
esac

echo "==> DPO dataset ready: $OUT/dataset"
echo "    Next: CONFIG=configs/dpo_8b.yaml ./run/03_dpo.sh"
