# Standalone DPO / GRPO training for the 404-GEN coder VLM

**Self-contained:** copy only this `training/` directory to a GPU box.  
Bootstrap fetches shiny-guide, prompts, and AstroWolf into `vendor/` and `data/`.

```bash
cd training
cp .env.template .env          # HF_TOKEN + OPENROUTER_API_KEY for duel path
./run/00_configure_profile.sh h200x4-dpo-duel
./run/00_bootstrap_assets.sh
INSTALL_SYSTEM=1 ./run/00_install_all.sh
# then follow STANDALONE.md Phase A/B/C
```

| Doc | Topic |
|-----|--------|
| [`STANDALONE.md`](STANDALONE.md) | Install + run loop |
| [`docs/DPO_DUEL_SCORING.md`](docs/DPO_DUEL_SCORING.md) | **2 JS + validator duel scoring on 4× H200** |
| [`docs/CODER_MODEL.md`](docs/CODER_MODEL.md) | Where to download AstroWolf |
| [`MACHINE_PROFILES.md`](MACHINE_PROFILES.md) | GPU profiles |

| Hardware | Profile |
|----------|---------|
| **4× H200 duel-scored DPO** | **`h200x4-dpo-duel`** ⭐ |
| 4× H200 cheap DPO | `h200x4-dpo` |
| 2× H200 DPO | `h200x2-dpo` |
| 1 GPU smoke | `smoke` |
