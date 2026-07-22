# Standalone training — everything in `training/`

Run the full loop from a **single folder**. No monorepo, no Docker, no `my-agent` / `local-eval`.

## New machine — start here

```bash
cd training
cp .env.template .env

# 1. Match config to your GPU box
./run/00_configure_profile.sh              # list options
./run/00_configure_profile.sh h200x2-dpo     # example: 2× H200 DPO

# 2. API keys (edit .env)
#    OPENROUTER_API_KEY — pipeline data gen only (critic/judge)
#    HF_TOKEN           — model download + training

# 3. Full automated run
chmod +x run/*.sh pipeline/*.sh
./run/run_all.sh
```

**Profile guide:** [`MACHINE_PROFILES.md`](MACHINE_PROFILES.md)

| Your hardware | Profile |
|---------------|---------|
| 2× H200 | `h200x2-dpo` ⭐ |
| 2× H100 80GB | `h100x2-dpo` |
| 4× H100 GRPO | `h100x4-grpo` |
| 8× H200 full FT | `h200x8-fullft` |
| 1 GPU test | `smoke` |
| Dataset ready | `train-only` |

---

## One command

```bash
./run/run_all.sh
```

Smoke: `SMOKE=1 ./run/run_all.sh` (or apply `smoke` profile first)

---

## What `run_all.sh` does

| Step | Script | Action |
|------|--------|--------|
| 0 | `00_configure_profile.sh` | **You run this first** — sets `.env` + pipeline GPUs |
| 1 | `00_bootstrap_assets.sh` | Clone shiny-guide, prompts.txt, AstroWolf |
| 2 | `00_install_all.sh` | Validator npm, training venv, pipeline venv + vLLM |
| 3 | `pipeline/start-native-bg.sh` | King pipeline on `:10006` |
| 4 | `01_prepare_shiny_align.sh` | Datasets (sizes from profile / `.env`) |
| 5 | `03_dpo.sh` / `03_grpo.sh` | bf16 LoRA training (`CONFIG` from profile) |

Skip flags: `SKIP_BOOTSTRAP=1`, `SKIP_INSTALL=1`, `SKIP_PIPELINE=1`, `SKIP_PREP=1`, `SKIP_TRAIN=1`

---

## Step-by-step (manual control)

### 0 — Configure for your machine

```bash
cp .env.template .env
./run/00_configure_profile.sh h200x2-dpo
# edit .env: OPENROUTER_API_KEY, HF_TOKEN
```

Profile writes to `.env`:

| Variable | Example (h200x2-dpo) |
|----------|----------------------|
| `TRAIN_N` | 6000 |
| `TRAIN` | dpo |
| `CONFIG` | configs/dpo_shiny_27b.yaml |
| `CONFIG_FILE` | pipeline/configuration.local.yaml |

And creates `pipeline/configuration.local.yaml` with `gpu_ids: "0,1"`, `tensor_parallel_size: 2`.

### 1 — Bootstrap assets

```bash
./run/00_bootstrap_assets.sh
```

### 2 — Install packages

```bash
INSTALL_SYSTEM=1 ./run/00_install_all.sh   # Ubuntu: Chromium libs
```

### 3 — Data generation (needs OPENROUTER)

```bash
source .env
./pipeline/start-native-bg.sh && ./pipeline/wait-ready.sh
source .venv/bin/activate
./run/01_prepare_shiny_align.sh    # uses TRAIN_N from .env
./pipeline/stop-native.sh         # free GPUs before training
```

### 4 — Train (OPENROUTER not needed)

```bash
source .venv/bin/activate
source .env
./run/03_dpo.sh                   # uses CONFIG from .env
```

### 5 — Merge + eval

```bash
./run/04_merge_and_eval.sh
export MODEL_PATH=$PWD/data/checkpoints/merged_shiny_coder
./pipeline/stop-native.sh && ./pipeline/start-native-bg.sh && ./pipeline/wait-ready.sh
./pipeline/run-eval.sh data/splits/duel.txt --limit 100 --name merged
```

---

## Capacity quick reference

See [`MACHINE_PROFILES.md`](MACHINE_PROFILES.md) for the full matrix.

| Method | Min GPUs | Recommended | Dataset |
|--------|----------|-------------|---------|
| DPO bf16 LoRA | 2× 80 GB | 2× H200 | 5k–6k prompts → 2k+ pairs |
| GRPO bf16 LoRA | 2× 80 GB (tight) | 4× H100 | 4k–5k prompts |
| Full SFT 27B | 8× H100 + ZeRO-3 | 8× H200 | 8k–12k validated JS |

Default yaml: `load_in_4bit: false`, `use_lora: true` (bf16 LoRA).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Wrong GPU layout | Re-run `./run/00_configure_profile.sh <profile>` |
| OOM training | See profile notes in `MACHINE_PROFILES.md`; lower `max_completion_length` |
| OOM same box | Stop pipeline before `./run/03_dpo.sh` |
| No OPENROUTER for train-only | Use `train-only` profile |

See also [`SHINY_GUIDE_TRAINING.md`](SHINY_GUIDE_TRAINING.md).
