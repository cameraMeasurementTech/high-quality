#!/usr/bin/env bash
# Step 0 — machine setup for the training cookbook.
#
# On a fresh GPU box, install CUDA torch BEFORE requirements.txt:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#   pip install -r requirements.txt
#
# Full walkthrough: SHINY_GUIDE_TRAINING.md (Phase 0–7)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
cd "$ROOT"

echo "==> Repo: $REPO"
echo "==> Training root: $ROOT"

# Node validator (required for reward + filter)
if [[ ! -d "$REPO/miner-reference/validator/node_modules" ]]; then
  echo "==> Installing miner-reference/validator deps"
  (cd "$REPO/miner-reference/validator" && npm install)
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip wheel

# Install CUDA torch FIRST if you are on a GPU box (edit cu124/cu121 to match driver).
# Example (CUDA 12.4):
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
#
# Then:
pip install -r requirements.txt

mkdir -p data/{splits,images,raw_js,filtered_js,candidates,hf,checkpoints,logs}
echo "==> Setup complete. Activate with: source $ROOT/.venv/bin/activate"
