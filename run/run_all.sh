#!/usr/bin/env bash
# One command: bootstrap → install → pipeline → dataset prep → train.
#
# Usage:
#   cp .env.template .env    # set OPENROUTER_API_KEY + HF_TOKEN
#   ./run/run_all.sh
#
# Env knobs:
#   TRAIN=dpo|grpo|both|skip     (default: dpo)
#   ALIGN=dpo|grpo|both          (default: matches TRAIN)
#   TRAIN_N=500 VAL_N=50 DUEL_N=50 DPO_SAMPLES=4
#   SKIP_BOOTSTRAP=1 SKIP_INSTALL=1 SKIP_PIPELINE=1 SKIP_PREP=1 SKIP_TRAIN=1
#   INSTALL_SYSTEM=1             apt install Chromium deps
#   SMOKE=1                      TRAIN_N=100, small smoke run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$TRAINING_ROOT"

if [[ ! -f .env ]]; then
  cp .env.template .env
  echo "Created .env — run ./run/00_configure_profile.sh <profile> then set API keys."
  echo "  Example: ./run/00_configure_profile.sh h200x2-dpo"
  exit 1
fi

if [[ -z "${MACHINE_PROFILE:-}" ]]; then
  echo "TIP: Run ./run/00_configure_profile.sh h200x2-dpo to match your GPU box (see MACHINE_PROFILES.md)"
fi

# shellcheck disable=SC1091
source .env

if [[ "${SMOKE:-0}" == "1" ]]; then
  export TRAIN_N="${TRAIN_N:-100}"
  export VAL_N="${VAL_N:-20}"
  export DUEL_N="${DUEL_N:-20}"
  export DPO_SAMPLES="${DPO_SAMPLES:-2}"
fi

TRAIN="${TRAIN:-dpo}"
ALIGN="${ALIGN:-$TRAIN}"
[[ "$ALIGN" == "skip" ]] && ALIGN="dpo"

echo "=============================================="
echo "  shiny-guide standalone training — run_all"
echo "  profile=${MACHINE_PROFILE:-<unset>}  TRAIN=$TRAIN  ALIGN=$ALIGN  TRAIN_N=${TRAIN_N:-5000}"
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

if [[ "${SKIP_PIPELINE:-0}" != "1" ]]; then
  "$PIPELINE_DIR/start-native-bg.sh"
  "$PIPELINE_DIR/wait-ready.sh"
fi

if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  echo "==> Prepare DPO/GRPO datasets"
  TRAIN_N="${TRAIN_N:-5000}" VAL_N="${VAL_N:-300}" DUEL_N="${DUEL_N:-200}" \
    DPO_SAMPLES="${DPO_SAMPLES:-4}" ALIGN="$ALIGN" \
    "$SCRIPT_DIR/01_prepare_shiny_align.sh"
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
echo "    Merge:  ADAPTER=data/checkpoints/dpo_shiny_27b/final ./run/04_merge_and_eval.sh"
echo "    Deploy: ./run/05_deploy_merged.sh"
echo "    Eval:   ./pipeline/run-eval.sh data/splits/duel.txt --limit 50 --name merged"
