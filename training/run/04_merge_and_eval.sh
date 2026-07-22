#!/usr/bin/env bash
# Step 4 — merge LoRA → point miner config → local duel vs shiny-guide.
#
# Env:
#   BASE_MODEL=handsometiger0202/Qwen-3.6-27B-AstroWolf
#   ADAPTER=data/checkpoints/grpo_8b/final
#   MERGED=data/checkpoints/merged_coder
#   DUEL_LIMIT=100
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
ADAPTER="${ADAPTER:-data/checkpoints/grpo_8b/final}"
MERGED="${MERGED:-data/checkpoints/merged_coder}"
DUEL_LIMIT="${DUEL_LIMIT:-50}"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

echo "==> [1/3] Merge LoRA"
python "$ROOT/scripts/merge_lora.py" \
  --base "$BASE_MODEL" \
  --adapter "$ROOT/$ADAPTER" \
  --out "$ROOT/$MERGED"

echo "==> [2/3] Wire merged weights into my-agent configuration.yaml"
echo "    Edit my-agent/configuration.yaml:"
echo "      llm_clients.coder-instance.vllm.model: $ROOT/$MERGED"
echo "      actors.coder.model: $ROOT/$MERGED"
echo "      actors.planner.model: $ROOT/$MERGED"
echo "    Then rebuild Docker: cd $REPO/my-agent && docker build -f docker/Dockerfile -t my-404-miner ."

echo "==> [3/3] Local duel vs shiny-guide (requires both pipelines up)"
echo "    # Terminal A"
echo "    source $REPO/local-eval/.env && $REPO/local-eval/run-pipeline-cpu.sh"
echo "    # Terminal B"
echo "    source $REPO/local-eval/.env && $REPO/local-eval/run-pipeline-my-agent.sh"
echo "    # Terminal C — use the held-out duel split"
echo "    $REPO/local-eval/run-duel-seq.sh $ROOT/data/splits/duel.txt --limit $DUEL_LIMIT --fresh"
echo ""
echo "Ship gate: decisive win-rate vs shiny-guide > 55–60% on held-out duel,"
echo "then Docker audit rehearsal on 128 prompts (submitted must not beat regenerated)."
