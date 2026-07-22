# Standalone training — self-contained in `training/`

Copy **only this folder** to a GPU machine. No monorepo, no `my-agent`, no `local-eval`, no sibling repos required.

Bootstrap clones everything else **into `training/`**:

```text
training/
├── vendor/shiny-guide/          ← git clone (pipeline + vLLM)
├── vendor/pipeline_prompts/     ← coder prompt snapshot
├── data/prompts.txt             ← ~99k validator URLs
├── data/models/...              ← AstroWolf weights
├── third_party/miner-reference/ ← bundled validate.js
├── pipeline/                    ← native launcher scripts
├── run/                         ← bootstrap, install, prep, train
└── scripts/                     ← Python tooling
```

## Quick start (new 4× H200 box)

```bash
cd training
cp .env.template .env
# edit .env: HF_TOKEN=hf_...  (OPENROUTER optional for cheap DPO)

./run/00_configure_profile.sh h200x4-dpo   # or h200x2-dpo
chmod +x run/*.sh pipeline/*.sh
INSTALL_SYSTEM=1 ./run/run_all.sh
```

Or step-by-step:

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

## API keys

| Key | Required? |
|-----|-----------|
| `HF_TOKEN` | **Yes** — model download + training |
| `OPENROUTER_API_KEY` | **Only** for duel-scored DPO or refinement/critic paths |

Cheap DPO prep (`refinement_enabled: false`, `openrouter.enabled: false`) does **not** call OpenRouter.

## Profiles

| Hardware | Profile |
|----------|---------|
| 4× H200 DPO prep | `h200x4-dpo` |
| 2× H200 DPO | `h200x2-dpo` |
| 1 GPU smoke | `smoke` |
| Pre-built dataset | `train-only` |

See [`MACHINE_PROFILES.md`](MACHINE_PROFILES.md).

## Duel-scored DPO (optional, needs OpenRouter)

```bash
# enable openrouter in pipeline/configuration.duel-judge.yaml
./run/01_prepare_dpo_duel_scored.sh
CONFIG=configs/dpo_shiny_27b_duel.yaml ./run/03_dpo.sh
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `shiny-guide not found` | `./run/00_bootstrap_assets.sh` |
| `prompts.txt not found` | same |
| `validate.js not found` | `./run/00_install_all.sh` |
| OOM on 4× H200 | lower `max_num_seqs` in `configuration.h200x4-dpo.yaml` |
| Train OOM | `./pipeline/stop-native.sh` before `./run/03_dpo.sh` |

See also: [`SHINY_GUIDE_TRAINING.md`](SHINY_GUIDE_TRAINING.md) · [`docs/DPO_DUEL_SCORING.md`](docs/DPO_DUEL_SCORING.md)
