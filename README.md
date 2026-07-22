# Standalone DPO / GRPO training for the 404-GEN coder VLM

**Self-contained:** copy only this `training/` directory to a GPU box.  
Bootstrap fetches shiny-guide, prompts, and AstroWolf into `vendor/` and `data/`.

```bash
cd training
cp .env.template .env          # set HF_TOKEN
./run/00_configure_profile.sh h200x4-dpo
./run/00_bootstrap_assets.sh
INSTALL_SYSTEM=1 ./run/00_install_all.sh
source .env && ./run/run_all.sh
```

Full guide: [`STANDALONE.md`](STANDALONE.md)

| Hardware | Profile |
|----------|---------|
| 4× H200 DPO prep | `h200x4-dpo` |
| 2× H200 DPO | `h200x2-dpo` |
| 1 GPU smoke | `smoke` |

Docs: [`MACHINE_PROFILES.md`](MACHINE_PROFILES.md) · [`docs/DPO_DUEL_SCORING.md`](docs/DPO_DUEL_SCORING.md)
