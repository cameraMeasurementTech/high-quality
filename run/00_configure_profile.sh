#!/usr/bin/env bash
# Apply a machine-capacity profile — sets .env + pipeline GPU layout for your hardware.
#
# Usage:
#   ./run/00_configure_profile.sh                  # list profiles
#   ./run/00_configure_profile.sh h200x2-dpo       # apply profile
#   MACHINE_PROFILE=h200x2-dpo ./run/00_configure_profile.sh
#
# Run this BEFORE 00_bootstrap_assets / 00_install_all on a new machine.
# Re-run anytime you move to different hardware.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$TRAINING_ROOT/.env"
PROFILE="${1:-${MACHINE_PROFILE:-}}"

if [[ -z "$PROFILE" ]]; then
  cat <<'EOF'
Machine profiles — pick one matching your GPU box:

  smoke          1× GPU smoke test (100 prompts, DPO, fast sanity)
  h100x2-dpo     2× H100 80GB — DPO bf16 LoRA (recommended minimum)
  h200x2-dpo     2× H200 141GB — DPO bf16 LoRA (recommended default)
  h200x4-dpo     4× H200 — cheap DPO data prep (TP=4, max_num_seqs=96)
  h200x4-dpo-duel 4× H200 — duel-scored DPO (2 JS + S1–S4 judge) ⭐
  h100x4-grpo    4× H100 80GB — GRPO bf16 LoRA + rollouts
  h200x2-grpo    2× H200 — GRPO bf16 LoRA (tight; num_generations=2)
  h200x8-fullft  8× H200 — full fine-tune SFT (use_lora: false, 8 GPU)
  train-only     Skip pipeline — train on pre-built dataset only

Apply:
  ./run/00_configure_profile.sh h200x2-dpo

Details: MACHINE_PROFILES.md
EOF
  exit 0
fi

[[ -f "$ENV_FILE" ]] || cp "$TRAINING_ROOT/.env.template" "$ENV_FILE"

upsert_env() {
  local key="$1" val="$2" file="$3"
  if grep -q "^${key}=" "$file" 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${val}|" "$file"
  else
    echo "${key}=${val}" >> "$file"
  fi
}

BASE_CFG="$TRAINING_ROOT/pipeline/configuration.gpu-native.yaml"
LOCAL_CFG="$TRAINING_ROOT/pipeline/configuration.local.yaml"

apply_pipeline_gpus() {
  local gpu_ids="$1" tp="$2"
  cp "$BASE_CFG" "$LOCAL_CFG"
  sed -i "s|gpu_ids: \"[^\"]*\"|gpu_ids: \"${gpu_ids}\"|" "$LOCAL_CFG"
  sed -i "s|tensor_parallel_size: [0-9]*|tensor_parallel_size: ${tp}|" "$LOCAL_CFG"
  upsert_env CONFIG_FILE "pipeline/configuration.local.yaml" "$ENV_FILE"
}

apply_pipeline_template() {
  local template="$1"
  cp "$template" "$LOCAL_CFG"
  upsert_env CONFIG_FILE "pipeline/configuration.local.yaml" "$ENV_FILE"
}

case "$PROFILE" in
  smoke)
    upsert_env MACHINE_PROFILE smoke "$ENV_FILE"
    upsert_env TRAIN dpo "$ENV_FILE"
    upsert_env TRAIN_N 100 "$ENV_FILE"
    upsert_env VAL_N 20 "$ENV_FILE"
    upsert_env DUEL_N 20 "$ENV_FILE"
    upsert_env DPO_SAMPLES 2 "$ENV_FILE"
    upsert_env CONFIG configs/dpo_shiny_27b.yaml "$ENV_FILE"
    upsert_env SMOKE 1 "$ENV_FILE"
    apply_pipeline_gpus "0" 1
    TRAIN_METHOD="DPO bf16 LoRA"
    DATA_NOTE="~100 prompts; sanity check only"
    GPU_NOTE="1× GPU for pipeline; training shares same GPU (run phases sequentially)"
    ;;
  h100x2-dpo)
    upsert_env MACHINE_PROFILE h100x2-dpo "$ENV_FILE"
    upsert_env TRAIN dpo "$ENV_FILE"
    upsert_env TRAIN_N 5000 "$ENV_FILE"
    upsert_env VAL_N 300 "$ENV_FILE"
    upsert_env DUEL_N 200 "$ENV_FILE"
    upsert_env DPO_SAMPLES 4 "$ENV_FILE"
    upsert_env CONFIG configs/dpo_shiny_27b.yaml "$ENV_FILE"
    upsert_env ALIGN dpo "$ENV_FILE"
    apply_pipeline_gpus "0,1" 2
    TRAIN_METHOD="DPO bf16 LoRA"
    DATA_NOTE="TRAIN_N=5000, DPO_SAMPLES=4 → aim ≥2000 pairs after mining"
    GPU_NOTE="2× GPU: pipeline tp=2; stop pipeline before training on same box"
    ;;
  h200x2-dpo)
    upsert_env MACHINE_PROFILE h200x2-dpo "$ENV_FILE"
    upsert_env TRAIN dpo "$ENV_FILE"
    upsert_env TRAIN_N 6000 "$ENV_FILE"
    upsert_env VAL_N 400 "$ENV_FILE"
    upsert_env DUEL_N 200 "$ENV_FILE"
    upsert_env DPO_SAMPLES 4 "$ENV_FILE"
    upsert_env CONFIG configs/dpo_shiny_27b.yaml "$ENV_FILE"
    upsert_env ALIGN dpo "$ENV_FILE"
    apply_pipeline_gpus "0,1" 2
    TRAIN_METHOD="DPO bf16 LoRA"
    DATA_NOTE="TRAIN_N=6000 recommended for 2× H200; aim ≥2500 DPO pairs"
    GPU_NOTE="2× H200: comfortable DPO; pipeline tp=2 on same box"
    ;;
  h200x4-dpo)
    upsert_env MACHINE_PROFILE h200x4-dpo "$ENV_FILE"
    upsert_env TRAIN dpo "$ENV_FILE"
    upsert_env TRAIN_N 5000 "$ENV_FILE"
    upsert_env VAL_N 300 "$ENV_FILE"
    upsert_env DUEL_N 200 "$ENV_FILE"
    upsert_env DPO_SAMPLES 4 "$ENV_FILE"
    upsert_env CONFIG configs/dpo_shiny_27b.yaml "$ENV_FILE"
    upsert_env ALIGN dpo "$ENV_FILE"
    upsert_env NUM_PROCESSES 2 "$ENV_FILE"
    apply_pipeline_template "$TRAINING_ROOT/pipeline/configuration.h200x4-dpo.yaml"
    TRAIN_METHOD="DPO bf16 LoRA (cheap validate.js pairs)"
    DATA_NOTE="TRAIN_N=5000, DPO_SAMPLES=4 → aim ≥2000 pairs; pipeline TP=4"
    GPU_NOTE="4× H200 data prep (vLLM TP=4); stop pipeline before DPO train"
    ;;
  h200x4-dpo-duel)
    upsert_env MACHINE_PROFILE h200x4-dpo-duel "$ENV_FILE"
    upsert_env TRAIN dpo "$ENV_FILE"
    upsert_env TRAIN_N 5000 "$ENV_FILE"
    upsert_env VAL_N 300 "$ENV_FILE"
    upsert_env DUEL_N 200 "$ENV_FILE"
    upsert_env DPO_SAMPLES 2 "$ENV_FILE"
    upsert_env BATCH_SIZE 48 "$ENV_FILE"
    upsert_env SIDECAR_COUNT 8 "$ENV_FILE"
    upsert_env DUEL_CONCURRENCY 4 "$ENV_FILE"
    upsert_env CONFIG configs/dpo_shiny_27b_duel.yaml "$ENV_FILE"
    upsert_env ALIGN dpo "$ENV_FILE"
    upsert_env NUM_PROCESSES 4 "$ENV_FILE"
    upsert_env JUDGE_CONFIG "$TRAINING_ROOT/pipeline/configuration.duel-judge.yaml" "$ENV_FILE"
    apply_pipeline_template "$TRAINING_ROOT/pipeline/configuration.h200x4-dpo-duel.yaml"
    TRAIN_METHOD="DPO bf16 LoRA on duel-scored pairs (S1–S4)"
    DATA_NOTE="TRAIN_N=5000 × 2 JS (seed diversity) → duel score → aim ≥3500–4500 pairs"
    GPU_NOTE="Phase A: TP=4 generate; stop vLLM; Phase B: Chromium+OpenRouter; Phase C: train ×4"
    ;;
  h100x4-grpo)
    upsert_env MACHINE_PROFILE h100x4-grpo "$ENV_FILE"
    upsert_env TRAIN grpo "$ENV_FILE"
    upsert_env TRAIN_N 5000 "$ENV_FILE"
    upsert_env VAL_N 300 "$ENV_FILE"
    upsert_env DUEL_N 200 "$ENV_FILE"
    upsert_env CONFIG configs/grpo_shiny_27b.yaml "$ENV_FILE"
    upsert_env ALIGN grpo "$ENV_FILE"
    apply_pipeline_gpus "0,1" 2
    TRAIN_METHOD="GRPO bf16 LoRA (num_generations=4 in yaml)"
    DATA_NOTE="TRAIN_N=5000 prompt-only; reward_mode=cheap default"
    GPU_NOTE="4× H100 for training; pipeline uses 2 GPU (split box or time-slice)"
    ;;
  h200x2-grpo)
    upsert_env MACHINE_PROFILE h200x2-grpo "$ENV_FILE"
    upsert_env TRAIN grpo "$ENV_FILE"
    upsert_env TRAIN_N 4000 "$ENV_FILE"
    upsert_env VAL_N 300 "$ENV_FILE"
    upsert_env DUEL_N 200 "$ENV_FILE"
    upsert_env CONFIG configs/grpo_shiny_27b.yaml "$ENV_FILE"
    upsert_env ALIGN grpo "$ENV_FILE"
    apply_pipeline_gpus "0,1" 2
    TRAIN_METHOD="GRPO bf16 LoRA — edit yaml: num_generations: 2 if OOM"
    DATA_NOTE="TRAIN_N=4000; tighter than 4× H100 GRPO profile"
    GPU_NOTE="2× H200 tight for GRPO; prefer DPO on this box"
    ;;
  h200x8-fullft)
    upsert_env MACHINE_PROFILE h200x8-fullft "$ENV_FILE"
    upsert_env TRAIN skip "$ENV_FILE"
    upsert_env PREP_SFT 1 "$ENV_FILE"
    upsert_env TRAIN_N 10000 "$ENV_FILE"
    upsert_env VAL_N 500 "$ENV_FILE"
    upsert_env DUEL_N 300 "$ENV_FILE"
    upsert_env CONFIG configs/sft_shiny_27b.yaml "$ENV_FILE"
    upsert_env ALIGN dpo "$ENV_FILE"
    upsert_env NUM_PROCESSES 8 "$ENV_FILE"
    apply_pipeline_gpus "0,1,2,3" 4
    TRAIN_METHOD="Full fine-tune SFT (use_lora: false in sft_shiny_27b.yaml)"
    DATA_NOTE="TRAIN_N=10000 validated teacher JS; PREP_SFT=1 during data prep"
    GPU_NOTE="8× H200 + DeepSpeed ZeRO-3; NOT supported on 2× H200"
    cat >> "$ENV_FILE" <<'NOTE'

# FULL FT: edit configs/sft_shiny_27b.yaml:
#   use_lora: false
#   load_in_4bit: false
# Launch: NUM_PROCESSES=8 USE_ACCELERATE=1 ./run/02_sft.sh
NOTE
    ;;
  train-only)
    upsert_env MACHINE_PROFILE train-only "$ENV_FILE"
    upsert_env SKIP_BOOTSTRAP 1 "$ENV_FILE"
    upsert_env SKIP_PIPELINE 1 "$ENV_FILE"
    upsert_env TRAIN dpo "$ENV_FILE"
    upsert_env CONFIG configs/dpo_shiny_27b.yaml "$ENV_FILE"
    TRAIN_METHOD="DPO bf16 LoRA (pre-built dataset)"
    DATA_NOTE="Place dataset at data/hf/dpo_shiny/dataset; no OPENROUTER needed"
    GPU_NOTE="Training GPUs only — 2× H200 / 2× H100 sufficient"
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    "$SCRIPT_DIR/00_configure_profile.sh"
    exit 1
    ;;
esac

cat <<EOF

Applied profile: ${PROFILE}
  Method:     ${TRAIN_METHOD}
  Dataset:    ${DATA_NOTE}
  GPUs:       ${GPU_NOTE}

Updated:
  ${ENV_FILE}
  ${LOCAL_CFG:-pipeline/configuration.local.yaml (if pipeline profile)}

Next on a new machine:
  1. Edit ${ENV_FILE} — set HF_TOKEN (OPENROUTER only for duel-scored / critic paths)
  2. ./run/00_bootstrap_assets.sh
  3. INSTALL_SYSTEM=1 ./run/00_install_all.sh
  4. source .env && ./run/run_all.sh

See MACHINE_PROFILES.md for full matrix.
EOF
