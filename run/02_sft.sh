#!/usr/bin/env bash
# Step 2 — SFT warm-start.
#   CONFIG=configs/sft_8b.yaml ./run/02_sft.sh
#   CONFIG=configs/sft_27b.yaml ./run/02_sft.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

CONFIG="${CONFIG:-configs/sft_8b.yaml}"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

# Resolve relative dataset/output paths against training root
export HF_HOME="${HF_HOME:-$ROOT/data/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

echo "==> SFT with $CONFIG"
if command -v accelerate >/dev/null 2>&1 && [[ "${USE_ACCELERATE:-1}" == "1" ]]; then
  accelerate launch --num_processes "${NUM_PROCESSES:-1}" \
    "$ROOT/scripts/train_sft.py" --config "$ROOT/$CONFIG"
else
  python "$ROOT/scripts/train_sft.py" --config "$ROOT/$CONFIG"
fi

echo "==> SFT finished. Next: CONFIG=configs/grpo_8b.yaml ./run/03_grpo.sh"
