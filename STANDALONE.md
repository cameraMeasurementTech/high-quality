# Standalone training — self-contained in `training/`

Copy **only this folder** to a GPU machine. No monorepo, no `my-agent`, no `local-eval`, no sibling repos required.

Bootstrap clones everything else **into `training/`**:

```text
training/
├── vendor/shiny-guide/          ← git clone (pipeline + vLLM)
├── vendor/pipeline_prompts/     ← coder prompt snapshot
├── data/prompts.txt             ← ~99k validator URLs
├── data/models/Qwen-3.6-27B-AstroWolf/  ← coder base model (see docs/CODER_MODEL.md)
├── third_party/miner-reference/ ← bundled validate.js
├── pipeline/                    ← native launcher scripts
├── run/                         ← bootstrap, install, prep, train
└── scripts/                     ← Python tooling
```

## Standalone self-check

```bash
./run/00_verify_standalone.sh
```

Expect PASS on bundled `validate.js` + configs. WARN for missing `vendor/` / model until bootstrap.

## Quick start — 4× H200 duel-scored DPO (recommended)

**Full numbered guide:** [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)

Generate **2 JS** per prompt (same prompts, **different seeds**), score with **validator-like S1–S4**, then DPO:

```bash
cd training
cp .env.template .env
# HF_TOKEN=...  OPENROUTER_API_KEY=...

./run/00_configure_profile.sh h200x4-dpo-duel
./run/00_bootstrap_assets.sh
INSTALL_SYSTEM=1 ./run/00_install_all.sh
source .env && source .venv/bin/activate

# Phase A — all 4 GPUs generate JS
./pipeline/start-native-bg.sh && ./pipeline/wait-ready.sh
SKIP_DUEL_SCORE=1 SKIP_PACK=1 ./run/01_prepare_dpo_duel_scored.sh
./pipeline/stop-native.sh

# Phase B — duel score (OpenRouter + Chromium)
SKIP_COLLECT=1 ./run/01_prepare_dpo_duel_scored.sh

# Phase C — train on 4 GPUs
CONFIG=configs/dpo_shiny_27b_duel.yaml NUM_PROCESSES=4 ./run/03_dpo.sh
```

Full detail: [`docs/DPO_DUEL_SCORING.md`](docs/DPO_DUEL_SCORING.md)

### How the 2 JS codes differ

Same image + same coder prompts + same temperature; **only the RNG seed changes** (production multigen style). Do **not** use different temperatures as the primary diversity method.

## GPT-5 teacher → SFT LoRA (full ~99k pool)

```bash
./run/00_configure_profile.sh h200x4-sft-gpt
# OPENROUTER_API_KEY + HF_TOKEN in .env
./run/00_bootstrap_assets.sh && INSTALL_SYSTEM=1 ./run/00_install_all.sh
source .env && source .venv/bin/activate

# Prep — always starts from data/prompts.txt (~99k); FULL_POOL=1 is default
./run/01_prepare_sft_openrouter.sh

# Train LoRA on 4× H200
CONFIG=configs/sft_shiny_27b_gpt_teacher.yaml NUM_PROCESSES=4 ./run/02_sft.sh
```

Full guide: [`docs/SFT_GPT_TEACHER.md`](docs/SFT_GPT_TEACHER.md)

```bash
./run/00_configure_profile.sh h200x4-dpo
./run/00_bootstrap_assets.sh
INSTALL_SYSTEM=1 ./run/00_install_all.sh

source .env && source .venv/bin/activate
./pipeline/start-native-bg.sh && ./pipeline/wait-ready.sh
ALIGN=dpo ./run/01_prepare_shiny_align.sh
./pipeline/stop-native.sh
./run/03_dpo.sh
```

## What runs where

| Phase | Directory | Command | Needs |
|-------|-----------|---------|-------|
| Bootstrap | `training/` | `./run/00_bootstrap_assets.sh` | git, HF_TOKEN, network |
| Install | `training/` | `./run/00_install_all.sh` | Node ≥20, CUDA, sudo (optional) |
| Pipeline | `training/` | `./pipeline/start-native-bg.sh` | 4× GPU for h200x4 profile |
| Dataset | `training/` | `./run/01_prepare_shiny_align.sh` | pipeline on `:10006` |
| Train | `training/` | `./run/03_dpo.sh` | stop pipeline first |
| Eval | `training/` | `./pipeline/run-eval.sh data/splits/duel.txt` | pipeline running |

## Coder base model

Download and save path for **Tooony133/Qwen-3.6-27B-AstroWolf**:

→ **[`docs/CODER_MODEL.md`](docs/CODER_MODEL.md)** (HF token, default path, manual download, verify)

Quick:

```bash
# in .env: HF_TOKEN=hf_...
./run/00_bootstrap_assets.sh
# saves to: training/data/models/Qwen-3.6-27B-AstroWolf/
```

## API keys

| Key | Required? |
|-----|-----------|
| `HF_TOKEN` | **Yes** — model download + training |
| `OPENROUTER_API_KEY` | **Only** for duel-scored DPO or refinement/critic paths |

Cheap DPO prep (`refinement_enabled: false`, `openrouter.enabled: false`) does **not** call OpenRouter.

## Profiles

| Hardware | Profile |
|----------|---------|
| **4× H200 duel-scored DPO** | **`h200x4-dpo-duel`** ⭐ |
| 4× H200 cheap DPO | `h200x4-dpo` |
| 2× H200 DPO | `h200x2-dpo` |
| 1 GPU smoke | `smoke` |
| Pre-built dataset | `train-only` |

See [`MACHINE_PROFILES.md`](MACHINE_PROFILES.md).

## Duel-scored DPO

→ [`docs/DPO_DUEL_SCORING.md`](docs/DPO_DUEL_SCORING.md)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `shiny-guide not found` | `./run/00_bootstrap_assets.sh` |
| `prompts.txt not found` | same |
| `validate.js not found` | `./run/00_install_all.sh` |
| Model download fails | See [`docs/CODER_MODEL.md`](docs/CODER_MODEL.md) — `HF_TOKEN`, disk ~60GB |
| OOM on 4× H200 | lower `max_num_seqs` in `configuration.h200x4-dpo.yaml` |
| Train OOM | `./pipeline/stop-native.sh` before `./run/03_dpo.sh` |

See also: [`SHINY_GUIDE_TRAINING.md`](SHINY_GUIDE_TRAINING.md) · [`docs/DPO_DUEL_SCORING.md`](docs/DPO_DUEL_SCORING.md)
