#!/usr/bin/env bash
# Step 3b — DPO preference optimization (offline pairs, no rollouts).
#   CONFIG=configs/dpo_8b.yaml ./run/03_dpo.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

CONFIG="${CONFIG:-configs/dpo_8b.yaml}"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROOT/data/hf_cache}"

echo "==> DPO with $CONFIG"
echo "    Dataset expected at: $(grep dataset_path "$ROOT/$CONFIG" | awk '{print $2}' | tr -d '"')"

if command -v accelerate >/dev/null 2>&1 && [[ "${USE_ACCELERATE:-1}" == "1" ]]; then
  accelerate launch --num_processes "${NUM_PROCESSES:-1}" \
    "$ROOT/scripts/train_dpo.py" --config "$ROOT/$CONFIG"
else
  python "$ROOT/scripts/train_dpo.py" --config "$ROOT/$CONFIG"
fi

echo "==> DPO finished. Next: ADAPTER=data/checkpoints/dpo_shiny_27b/final ./run/04_merge_and_eval.sh"
echo "    (No SFT step — base model in yaml is already production AstroWolf.)"
