#!/usr/bin/env bash
# Step 3 — GRPO with validator-shaped rewards.
#   CONFIG=configs/grpo_8b.yaml ./run/03_grpo.sh
#   REWARD_MODE=s1 JUDGE_BASE_URL=... JUDGE_MODEL=... ./run/03_grpo.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

CONFIG="${CONFIG:-configs/grpo_8b.yaml}"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROOT/data/hf_cache}"

# Optional overrides (also readable from yaml)
export RENDER_URL="${RENDER_URL:-}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
export JUDGE_MODEL="${JUDGE_MODEL:-}"

echo "==> GRPO with $CONFIG"
echo "    RENDER_URL=${RENDER_URL:-<unset>} JUDGE_BASE_URL=${JUDGE_BASE_URL:-<unset>}"

if command -v accelerate >/dev/null 2>&1 && [[ "${USE_ACCELERATE:-1}" == "1" ]]; then
  accelerate launch --num_processes "${NUM_PROCESSES:-1}" \
    "$ROOT/scripts/train_grpo.py" --config "$ROOT/$CONFIG"
else
  python "$ROOT/scripts/train_grpo.py" --config "$ROOT/$CONFIG"
fi

echo "==> GRPO finished. Next: ./run/04_merge_and_eval.sh"
