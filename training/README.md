# my-agent training

**Fresh GPU machine walkthrough:** [`SHINY_GUIDE_TRAINING.md`](SHINY_GUIDE_TRAINING.md) — start with **Phase 0**.

Train `Tooony133/Qwen-3.6-27B-AstroWolf` on validator prompts (shiny-guide prompts + skip SFT).

## Quick start (after Phase 0–2 setup)

```bash
export REPO=/home/404-gen-subnet
export TRAINING=$REPO/my-agent/training
cd $TRAINING && source .venv/bin/activate
export PROMPTS_ROOT=shiny-guide PYTHONPATH=$TRAINING/scripts

# shiny-guide GPU docker on :10006, then:
ALIGN=dpo TRAIN_N=5000 ./run/01_prepare_shiny_align.sh
CONFIG=configs/dpo_shiny_27b.yaml ./run/03_dpo.sh

# OR GRPO:
# ALIGN=grpo TRAIN_N=5000 ./run/01_prepare_shiny_align.sh
# CONFIG=configs/grpo_shiny_27b.yaml ./run/03_grpo.sh
```

General cookbook (all methods): [`COOKBOOK.md`](COOKBOOK.md)

Strategy docs:

- [`docs/TRAINING_DPO_STRATEGY.md`](../../docs/TRAINING_DPO_STRATEGY.md)
- [`docs/TRAINING_GRPO_STRATEGY.md`](../../docs/TRAINING_GRPO_STRATEGY.md)
