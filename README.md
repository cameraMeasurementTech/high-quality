# Shiny-guide training — standalone

Everything in this folder: **profile → bootstrap → install → data prep → train → eval**.

**New machine:** [`MACHINE_PROFILES.md`](MACHINE_PROFILES.md) — pick hardware profile first.

## Quick start

```bash
cd training
cp .env.template .env
./run/00_configure_profile.sh h200x2-dpo   # match your GPU box
# edit .env: OPENROUTER_API_KEY + HF_TOKEN
chmod +x run/*.sh pipeline/*.sh
./run/run_all.sh
```

| GPUs | Profile |
|------|---------|
| 2× H200 (DPO) | `h200x2-dpo` |
| 2× H100 (DPO) | `h100x2-dpo` |
| 4× H100 (GRPO) | `h100x4-grpo` |
| 8× H200 (full FT) | `h200x8-fullft` |

Full guide: [`STANDALONE.md`](STANDALONE.md) · Phases: [`SHINY_GUIDE_TRAINING.md`](SHINY_GUIDE_TRAINING.md)
