# Standalone training — everything in `training/`

Run the full loop from a **single folder**. No monorepo, no Docker, no `my-agent` / `local-eval`.

## One command

```bash
cd training
cp .env.template .env
# Edit: OPENROUTER_API_KEY, HF_TOKEN

chmod +x run/*.sh pipeline/*.sh
./run/run_all.sh
```

Smoke test (small dataset, fast sanity check):

```bash
SMOKE=1 ./run/run_all.sh
```

---

## What `run_all.sh` does

| Step | Script | Action |
|------|--------|--------|
| 1 | `00_bootstrap_assets.sh` | Clone **shiny-guide**, download **prompts.txt**, download **AstroWolf** locally |
| 2 | `00_install_all.sh` | Validator npm, training venv + CUDA torch, pipeline venv + vLLM |
| 3 | `pipeline/start-native-bg.sh` | Start king pipeline on `:10006` |
| 4 | `01_prepare_shiny_align.sh` | Splits, images, DPO candidates, pack DPO + GRPO datasets |
| 5 | `03_dpo.sh` / `03_grpo.sh` | QLoRA training |

Skip flags: `SKIP_BOOTSTRAP=1`, `SKIP_INSTALL=1`, `SKIP_PIPELINE=1`, `SKIP_PREP=1`, `SKIP_TRAIN=1`

Train mode: `TRAIN=dpo` (default), `TRAIN=grpo`, `TRAIN=both`, `TRAIN=skip`

---

## Workspace layout (auto-created)

```
workspace/                          # WORKSPACE_ROOT (parent or monorepo root)
├── shiny-guide/                    # cloned from GitHub
├── prompts.txt                     # ~99k validator URLs (or data/prompts.txt)
├── models/
│   └── Qwen-3.6-27B-AstroWolf/     # local coder weights
└── training/                       # this folder
    ├── .env
    ├── .venv
    ├── pipeline/                   # native GPU runner
    ├── third_party/                # bundled validate.js
    └── data/                       # splits, images, hf datasets, checkpoints
```

Inside the monorepo (`404-gen-subnet/my-agent/training`), paths auto-detect sibling `shiny-guide/` and repo `prompts.txt`.

---

## Step-by-step (manual control)

### 0 — Configure

```bash
cp .env.template .env
```

Required keys: `OPENROUTER_API_KEY`, `HF_TOKEN` (for AstroWolf + DINOv3).

Optional bootstrap overrides in `.env`:

| Variable | Default |
|----------|---------|
| `SHINY_GUIDE_REPO` | `https://github.com/mokabetrade/shiny-guide.git` |
| `PROMPTS_URL` | raw URL to `prompts.txt` |
| `CODER_MODEL_ID` | `Tooony133/Qwen-3.6-27B-AstroWolf` |
| `MODEL_DIR` | `$WORKSPACE/models/Qwen-3.6-27B-AstroWolf` |

### 1 — Bootstrap assets

```bash
./run/00_bootstrap_assets.sh
```

Clones shiny-guide, fetches prompts, downloads model, writes paths to `.env`.

### 2 — Install packages

```bash
INSTALL_SYSTEM=1 ./run/00_install_all.sh   # + apt Chromium libs on Ubuntu
```

Or on a machine that already has Node 20+ and GPU drivers:

```bash
./run/00_install_all.sh
```

### 3 — Start pipeline (Terminal A)

```bash
source .env
./pipeline/start-native-bg.sh
./pipeline/wait-ready.sh
# curl -s http://127.0.0.1:10006/health
```

Stop: `./pipeline/stop-native.sh`

### 4 — Prepare datasets

```bash
source .venv/bin/activate
source .env

ALIGN=dpo TRAIN_N=5000 VAL_N=300 DUEL_N=200 DPO_SAMPLES=4 \
  ./run/01_prepare_shiny_align.sh
```

| ALIGN | Output |
|-------|--------|
| `dpo` | `data/hf/dpo_shiny/dataset` |
| `grpo` | `data/hf/grpo_shiny/dataset` |
| `both` | both |

### 5 — Train

```bash
CONFIG=configs/dpo_shiny_27b.yaml ./run/03_dpo.sh
# or
CONFIG=configs/grpo_shiny_27b.yaml ./run/03_grpo.sh
```

Uses `MODEL_PATH` from `.env` when set (local AstroWolf).

### 6 — Merge, deploy, beat the king

```bash
ADAPTER=data/checkpoints/dpo_shiny_27b/final \
  MERGED=data/checkpoints/merged_shiny_coder \
  ./run/04_merge_and_eval.sh

export MODEL_PATH=$PWD/data/checkpoints/merged_shiny_coder
./pipeline/stop-native.sh
./pipeline/start-native-bg.sh && ./pipeline/wait-ready.sh

./pipeline/run-eval.sh data/splits/duel.txt --limit 100 --name merged
```

Compare with baseline eval (pipeline on original AstroWolf). Target: **>55–60%** win rate on held-out `duel.txt`.

Optional: DPO → GRPO stack — set `sft_adapter_path: data/checkpoints/dpo_shiny_27b/final` in `grpo_shiny_27b.yaml`.

---

## GPU requirements (27B QLoRA)

| Stage | Minimum | Recommended |
|-------|---------|-------------|
| Pipeline (vLLM) | 1× 80 GB | 2× 80 GB |
| DPO | 1× 80 GB | 2× H100 |
| GRPO | 2× 80 GB | 4× H100 |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `prompts.txt` download fails | Copy manually to `data/prompts.txt`, set `PROMPTS_POOL` |
| Model download 401 | Set `HF_TOKEN`, accept model license on HuggingFace |
| Pipeline not ready | `tail -f pipeline/runs/pipeline-server.log` |
| Chromium fails | `INSTALL_SYSTEM=1 ./run/00_install_all.sh` |
| OOM training | Lower `max_completion_length`; GRPO: `num_generations: 2` |

See also [`SHINY_GUIDE_TRAINING.md`](SHINY_GUIDE_TRAINING.md) for detailed phases.

---

## Monorepo

Still works at `404-gen-subnet/my-agent/training`. Bootstrap skips clone/download when assets already exist.
