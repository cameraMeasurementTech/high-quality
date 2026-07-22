#!/usr/bin/env bash
# Step 3b — DPO preference optimization (offline pairs, no rollouts).
#   CONFIG=configs/dpo_8b.yaml ./run/03_dpo.sh
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"
cd "$TRAINING_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

# CONFIG may be set by MACHINE_PROFILE in .env
CONFIG="${CONFIG:-configs/dpo_8b.yaml}"

echo "==> DPO with $CONFIG"
echo "    Dataset expected at: $(grep dataset_path "$TRAINING_ROOT/$CONFIG" | awk '{print $2}' | tr -d '"')"

if command -v accelerate >/dev/null 2>&1 && [[ "${USE_ACCELERATE:-1}" == "1" ]]; then
  accelerate launch --num_processes "${NUM_PROCESSES:-1}" \
    "$TRAINING_ROOT/scripts/train_dpo.py" --config "$TRAINING_ROOT/$CONFIG"
else
  python "$TRAINING_ROOT/scripts/train_dpo.py" --config "$TRAINING_ROOT/$CONFIG"
fi

echo "==> DPO finished. Next: ADAPTER=data/checkpoints/dpo_shiny_27b/final ./run/04_merge_and_eval.sh"
echo "    (No SFT step — base model in yaml is already production AstroWolf.)"
