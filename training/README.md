# Training (shiny-guide focused)

**Start here:** [`SHINY_GUIDE_TRAINING.md`](SHINY_GUIDE_TRAINING.md)

AstroWolf is **already SFT-trained** in production — skip `./run/02_sft.sh` and use **DPO or GRPO** directly.

```bash
cd /home/404-gen-subnet/my-agent/training
./run/00_setup.sh && source .venv/bin/activate
export PROMPTS_ROOT=shiny-guide

# shiny-guide GPU on :10006, then prep data:
ALIGN=dpo TRAIN_N=5000 ./run/01_prepare_shiny_align.sh   # or ALIGN=grpo

# Train (pick one):
CONFIG=configs/dpo_shiny_27b.yaml ./run/03_dpo.sh
CONFIG=configs/grpo_shiny_27b.yaml ./run/03_grpo.sh

ADAPTER=data/checkpoints/dpo_shiny_27b/final ./run/04_merge_and_eval.sh
```

General cookbook: [`COOKBOOK.md`](COOKBOOK.md)
