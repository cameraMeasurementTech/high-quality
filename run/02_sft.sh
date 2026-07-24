#!/usr/bin/env bash
# Step 2 — SFT LoRA warm-start (AstroWolf or other VLM).
#
# GPT-5 teacher distill:
#   CONFIG=configs/sft_shiny_27b_gpt_teacher.yaml NUM_PROCESSES=4 ./run/02_sft.sh
#
# See docs/SFT_GPT_TEACHER.md
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"
cd "$TRAINING_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

CONFIG="${CONFIG:-configs/sft_shiny_27b_gpt_teacher.yaml}"
export PYTHONPATH="$TRAINING_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$TRAINING_ROOT/data/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

echo "==> SFT with $CONFIG"
echo "    MODEL_PATH=${MODEL_PATH:-<yaml / HF id>}"
if command -v accelerate >/dev/null 2>&1 && [[ "${USE_ACCELERATE:-1}" == "1" ]]; then
  accelerate launch --num_processes "${NUM_PROCESSES:-1}" \
    "$TRAINING_ROOT/scripts/train_sft.py" --config "$TRAINING_ROOT/$CONFIG"
else
  python "$TRAINING_ROOT/scripts/train_sft.py" --config "$TRAINING_ROOT/$CONFIG"
fi

echo "==> SFT finished."
echo "    Adapter: data/checkpoints/.../final  (see yaml output_dir)"
echo "    Next: merge (optional) then DPO/GRPO on AstroWolf self-samples"
