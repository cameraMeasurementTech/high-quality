#!/usr/bin/env bash
# One command: bootstrap → install → pipeline → dataset prep → train.
#
# Usage:
#   cp .env.template .env
#   ./run/00_configure_profile.sh h200x4-dpo-duel   # or h200x4-dpo / h200x2-dpo
#   # set HF_TOKEN (+ OPENROUTER_API_KEY for duel profiles)
#   INSTALL_SYSTEM=1 ./run/run_all.sh
#
# Duel profiles (MACHINE_PROFILE=*dpo-duel* or CONFIG=*dpo_shiny_27b_duel*):
#   Phase A generate → stop pipeline → Phase B score+pack → Phase C train
#
# Env knobs:
#   TRAIN=dpo|grpo|both|skip
#   SKIP_BOOTSTRAP=1 SKIP_INSTALL=1 SKIP_PIPELINE=1 SKIP_PREP=1 SKIP_TRAIN=1
#   INSTALL_SYSTEM=1
#   SMOKE=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$TRAINING_ROOT"

if [[ ! -f .env ]]; then
  cp .env.template .env
  echo "Created .env — run ./run/00_configure_profile.sh <profile> then set API keys."
  echo "  Example: ./run/00_configure_profile.sh h200x4-dpo-duel"
  exit 1
fi

if [[ -z "${MACHINE_PROFILE:-}" ]]; then
  echo "TIP: Run ./run/00_configure_profile.sh h200x4-dpo-duel (see MACHINE_PROFILES.md)"
fi

# shellcheck disable=SC1091
source .env

if [[ "${SMOKE:-0}" == "1" ]]; then
  export TRAIN_N="${TRAIN_N:-100}"
  export VAL_N="${VAL_N:-20}"
  export DUEL_N="${DUEL_N:-20}"
  export DPO_SAMPLES="${DPO_SAMPLES:-2}"
  export DUEL_LIMIT="${DUEL_LIMIT:-50}"
fi

TRAIN="${TRAIN:-dpo}"
ALIGN="${ALIGN:-$TRAIN}"
[[ "$ALIGN" == "skip" ]] && ALIGN="dpo"

is_duel_profile=0
case "${MACHINE_PROFILE:-}" in
  *dpo-duel*) is_duel_profile=1 ;;
esac
case "${CONFIG:-}" in
  *dpo_shiny_27b_duel*) is_duel_profile=1 ;;
esac

echo "=============================================="
echo "  standalone training — run_all"
echo "  profile=${MACHINE_PROFILE:-<unset>}  TRAIN=$TRAIN  ALIGN=$ALIGN  TRAIN_N=${TRAIN_N:-5000}"
echo "  duel_scored=${is_duel_profile}"
echo "=============================================="

if [[ "${SKIP_BOOTSTRAP:-0}" != "1" ]]; then
  "$SCRIPT_DIR/00_bootstrap_assets.sh"
fi

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  INSTALL_SYSTEM="${INSTALL_SYSTEM:-0}" "$SCRIPT_DIR/00_install_all.sh"
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
# shellcheck disable=SC1091
source .venv/bin/activate

if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  if [[ "$is_duel_profile" == "1" ]]; then
    : "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY in .env for duel-scored DPO}"

    if [[ "${SKIP_PIPELINE:-0}" != "1" ]]; then
      "$PIPELINE_DIR/start-native-bg.sh"
      "$PIPELINE_DIR/wait-ready.sh"
    fi

    echo "==> [duel] Phase A — collect 2 JS/stem (seed diversity)"
    SKIP_DUEL_SCORE=1 SKIP_PACK=1 \
      TRAIN_N="${TRAIN_N:-5000}" VAL_N="${VAL_N:-300}" DUEL_N="${DUEL_N:-200}" \
      DPO_SAMPLES="${DPO_SAMPLES:-2}" BATCH_SIZE="${BATCH_SIZE:-96}" \
      "$SCRIPT_DIR/01_prepare_dpo_duel_scored.sh"

    echo "==> [duel] Stopping pipeline before scoring / training"
    "$PIPELINE_DIR/stop-native.sh" || true

    echo "==> [duel] Phase B — multiview S1–S4 score + pack"
    SKIP_COLLECT=1 \
      SIDECAR_COUNT="${SIDECAR_COUNT:-8}" DUEL_CONCURRENCY="${DUEL_CONCURRENCY:-4}" \
      DUEL_LIMIT="${DUEL_LIMIT:-0}" \
      "$SCRIPT_DIR/01_prepare_dpo_duel_scored.sh"
  else
    if [[ "${SKIP_PIPELINE:-0}" != "1" ]]; then
      "$PIPELINE_DIR/start-native-bg.sh"
      "$PIPELINE_DIR/wait-ready.sh"
    fi

    echo "==> Prepare cheap DPO/GRPO datasets"
    TRAIN_N="${TRAIN_N:-5000}" VAL_N="${VAL_N:-300}" DUEL_N="${DUEL_N:-200}" \
      DPO_SAMPLES="${DPO_SAMPLES:-4}" ALIGN="$ALIGN" \
      "$SCRIPT_DIR/01_prepare_shiny_align.sh"

    echo "==> Stopping pipeline before training"
    "$PIPELINE_DIR/stop-native.sh" || true
  fi
fi

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  echo "==> SKIP_TRAIN=1 — done after data prep"
  exit 0
fi

case "$TRAIN" in
  dpo)
    CONFIG="${CONFIG:-configs/dpo_shiny_27b.yaml}" "$SCRIPT_DIR/03_dpo.sh"
    ;;
  grpo)
    CONFIG="${CONFIG:-configs/grpo_shiny_27b.yaml}" "$SCRIPT_DIR/03_grpo.sh"
    ;;
  both)
    CONFIG="${CONFIG:-configs/dpo_shiny_27b.yaml}" "$SCRIPT_DIR/03_dpo.sh"
    CONFIG="${CONFIG:-configs/grpo_shiny_27b.yaml}" "$SCRIPT_DIR/03_grpo.sh"
    ;;
  skip)
    echo "TRAIN=skip — no training step"
    ;;
  *)
    echo "Unknown TRAIN=$TRAIN (use dpo|grpo|both|skip)" >&2
    exit 1
    ;;
esac

echo ""
echo "==> run_all complete"
echo "    Merge:  ADAPTER=data/checkpoints/dpo_shiny_27b_duel/final ./run/04_merge_and_eval.sh"
echo "    Eval:   ./pipeline/run-eval.sh data/splits/duel.txt --limit 50 --name merged"
