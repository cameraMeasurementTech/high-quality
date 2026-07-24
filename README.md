# Standalone DPO / GRPO training for the 404-GEN coder VLM

**Self-contained:** copy or clone only this `training/` directory to a GPU box.  
Bootstrap fetches shiny-guide, prompts, and AstroWolf into `vendor/` and `data/`.

## Start here

**Full step-by-step (clone → base model → dataset → train):**  
[`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)

**GPT-5 → LoRA SFT on full ~99k pool:**  
[`docs/SFT_GPT_TEACHER.md`](docs/SFT_GPT_TEACHER.md)

```bash
cd training
cp .env.template .env
# Set HF_TOKEN=... and OPENROUTER_API_KEY=... in .env

./run/00_configure_profile.sh h200x4-dpo-duel
./run/00_bootstrap_assets.sh          # downloads AstroWolf → data/models/...
INSTALL_SYSTEM=1 ./run/00_install_all.sh
source .env && source .venv/bin/activate

# Phase A — generate 2 JS/stem (4× GPU, skip_render)
./pipeline/start-native-bg.sh && ./pipeline/wait-ready.sh
SKIP_DUEL_SCORE=1 SKIP_PACK=1 ./run/01_prepare_dpo_duel_scored.sh
./pipeline/stop-native.sh

# Phase B — duel score + pack DPO dataset
SKIP_COLLECT=1 ./run/01_prepare_dpo_duel_scored.sh

# Phase C — train
CONFIG=configs/dpo_shiny_27b_duel.yaml NUM_PROCESSES=4 ./run/03_dpo.sh
```

| Doc | Topic |
|-----|--------|
| [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) | **Clone → model → dataset → train** |
| [`docs/SFT_GPT_TEACHER.md`](docs/SFT_GPT_TEACHER.md) | **GPT-5 teacher → SFT LoRA (miner prompts)** |
| [`STANDALONE.md`](STANDALONE.md) | Layout + self-check |
| [`docs/CODER_MODEL.md`](docs/CODER_MODEL.md) | Where AstroWolf is downloaded/saved |
| [`docs/DPO_DUEL_SCORING.md`](docs/DPO_DUEL_SCORING.md) | 2 JS + validator duel scoring |
| [`MACHINE_PROFILES.md`](MACHINE_PROFILES.md) | GPU profiles |

| Hardware | Profile |
|----------|---------|
| **4× H200 duel-scored DPO** | **`h200x4-dpo-duel`** ⭐ |
| **4× H200 GPT-5 → SFT LoRA** | **`h200x4-sft-gpt`** |
| 4× H200 cheap DPO | `h200x4-dpo` |
| 2× H200 DPO | `h200x2-dpo` |
| 1 GPU smoke | `smoke` |
