#!/usr/bin/env bash
# Eval shiny-guide pipeline against a prompt list (standalone — no local-eval monorepo).
#
# Usage:
#   ./run-eval.sh data/splits/duel.txt --limit 50
#   ./run-eval.sh data/splits/val.txt --limit 10 --name post-train
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$TRAINING_ROOT/run/env.sh"

PROMPTS="${1:?usage: run-eval.sh <prompts.txt> [--limit N] [--name RUN]}"
shift

LIMIT=""
RUN_NAME="eval"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --name) RUN_NAME="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ ! -f "$PROMPTS" ]]; then
  if [[ -f "$TRAINING_ROOT/$PROMPTS" ]]; then
    PROMPTS="$TRAINING_ROOT/$PROMPTS"
  else
    echo "ERROR: prompts file not found: $PROMPTS" >&2
    exit 1
  fi
fi

OUT_DIR="$PIPELINE_DIR/runs/eval/$RUN_NAME"
mkdir -p "$OUT_DIR"

LIMIT_ARGS=()
[[ -n "$LIMIT" ]] && LIMIT_ARGS=(--limit "$LIMIT")

cd "$SHINY_GUIDE_ROOT"
python3 tests/test_pipeline.py \
  --prompts "$PROMPTS" \
  --base-url "${PIPELINE_URL:-http://127.0.0.1:10006}" \
  --out "$OUT_DIR" \
  "${LIMIT_ARGS[@]}" \
  "${EXTRA[@]}"

echo "Results -> $OUT_DIR"
