# Getting started — clone → base model → dataset → train

Step-by-step guide after you clone **only this `training/` repo** onto a GPU machine.
No monorepo siblings required.

**Recommended path (subnet-aligned):** 4× H200, duel-scored DPO  
Profile: `h200x4-dpo-duel`

---

## What you will end up with

```text
training/
├── .env                                      ← secrets + profile
├── vendor/shiny-guide/                       ← generation pipeline
├── data/
│   ├── prompts.txt                           ← image URL pool (~99k)
│   ├── models/Qwen-3.6-27B-AstroWolf/        ← base coder weights
│   ├── images/                               ← downloaded references
│   ├── candidates/shiny_k2/                  ← 2 JS per stem
│   ├── duel_scores/candidate_duels.json      ← S1–S4 scores
│   ├── hf/dpo_shiny_duel/dataset/            ← HF DPO dataset
│   └── checkpoints/dpo_shiny_27b_duel/       ← LoRA adapter (after train)
└── ...
```

---

## Prerequisites

| Need | Notes |
|------|--------|
| GPU | **4× H200** recommended for duel path (or see [`MACHINE_PROFILES.md`](../MACHINE_PROFILES.md)) |
| Disk | ≥ **100 GB** free (model ~54 GB + cache + candidates) |
| OS | Linux, CUDA drivers matching your torch wheel |
| Node | **≥ 20** (Chromium sidecars for Phase B scoring) |
| Keys | `HF_TOKEN` (always); `OPENROUTER_API_KEY` (**required** for duel-scored path) |

---

## Step 0 — Clone and enter the repo

```bash
git clone <YOUR_TRAINING_REPO_URL> training
cd training
```

Confirm the tree looks standalone:

```bash
ls run/ pipeline/ scripts/ configs/ docs/ third_party/
./run/00_verify_standalone.sh
```

Expect **PASS** on bundled files; **WARN** for missing `vendor/` / model until bootstrap (normal).

---

## Step 1 — Create `.env` and pick a machine profile

```bash
cp .env.template .env
```

Edit `.env` and set at least:

```bash
HF_TOKEN=hf_...                    # https://huggingface.co/settings/tokens
OPENROUTER_API_KEY=sk-or-v1-...    # required for duel-scored dataset
```

Apply the 4× H200 duel profile (writes paths, batch sizes, train config into `.env`):

```bash
./run/00_configure_profile.sh h200x4-dpo-duel
```

This sets things like:

| Variable | Typical value |
|----------|----------------|
| `CONFIG` | `configs/dpo_shiny_27b_duel.yaml` |
| `DPO_SAMPLES` | `2` |
| `BATCH_SIZE` | `96` |
| `NUM_PROCESSES` | `4` |
| `CONFIG_FILE` | `pipeline/configuration.local.yaml` (copied from duel template) |

Other profiles: `h200x4-dpo` (cheap, no OpenRouter judge), `h200x2-dpo`, `smoke`.

---

## Step 2 — Locate / download the base model

### What the base model is

| Field | Value |
|-------|--------|
| HuggingFace | [`Tooony133/Qwen-3.6-27B-AstroWolf`](https://huggingface.co/Tooony133/Qwen-3.6-27B-AstroWolf) |
| Role | Image → Three.js coder (27B VLM) |
| Training | **Skip SFT** — start DPO/GRPO from this checkpoint |
| Size | ~54 GB bf16 |

Full detail: [`docs/CODER_MODEL.md`](CODER_MODEL.md)

### Default location (after bootstrap)

```text
training/data/models/Qwen-3.6-27B-AstroWolf/
├── config.json
├── tokenizer.json
├── model.safetensors.index.json   # (or *.safetensors)
└── ...
```

`.env` will contain:

```bash
MODEL_PATH=.../training/data/models/Qwen-3.6-27B-AstroWolf
CODER_MODEL_PATH=$MODEL_PATH
CODER_MODEL_ID=Tooony133/Qwen-3.6-27B-AstroWolf
```

### Download automatically (recommended)

```bash
# HF_TOKEN must be in .env or the environment
./run/00_bootstrap_assets.sh
```

Bootstrap also fetches:

- `vendor/shiny-guide/` — generation pipeline
- `data/prompts.txt` — **~99k image URLs** (one `https://…png` per line)
- coder prompt snapshot under `vendor/pipeline_prompts/`

How the prompt pool is used:

```text
data/prompts.txt
    → prepare_splits.py  →  data/splits/{train,val,duel}.txt   (stem\\turl)
    → download_images.py →  data/images/{stem}.png
    → collect / duel score / pack DPO
```

Re-download prompts only:

```bash
FORCE_PROMPTS_DOWNLOAD=1 ./run/00_bootstrap_assets.sh
```

First model download often takes **1–3 hours**.

### Or use an existing local copy

```bash
# in .env
MODEL_PATH=/mnt/nvme/models/Qwen-3.6-27B-AstroWolf
CODER_MODEL_PATH=$MODEL_PATH
```

Or symlink:

```bash
mkdir -p data/models
ln -s /mnt/nvme/models/Qwen-3.6-27B-AstroWolf data/models/Qwen-3.6-27B-AstroWolf
```

### Verify the model is present

```bash
source .env
ls "$MODEL_PATH/config.json"
ls "$MODEL_PATH"/*.safetensors* 2>/dev/null | head
./run/00_verify_standalone.sh
```

---

## Step 3 — Install software dependencies

```bash
# Chromium + system packages for render sidecars (Phase B)
INSTALL_SYSTEM=1 ./run/00_install_all.sh

source .env
source .venv/bin/activate
```

This installs the Python venv, Node deps for shiny-guide sidecars, and (with `INSTALL_SYSTEM=1`) OS packages.

---

## Step 4 — Prepare the training dataset (duel-scored DPO)

You generate **2 JS** per prompt (same image/prompts/temperature, **different seeds**), then score them with validator-like S1–S4 duels, then pack chosen/rejected pairs.

### Phase A — Generate JS (uses all 4 GPUs, TP=4)

Pipeline config uses `skip_render: true` here: **JS only**. Multiview render happens in Phase B.

**Seed-only diversity (default):** same temperature (~0.6), different seeds.

```bash
source .env && source .venv/bin/activate

./pipeline/start-native-bg.sh
./pipeline/wait-ready.sh

SKIP_DUEL_SCORE=1 SKIP_PACK=1 \
  TRAIN_N=5000 DPO_SAMPLES=2 BATCH_SIZE=96 \
  ./run/01_prepare_dpo_duel_scored.sh

./pipeline/stop-native.sh    # free GPUs before scoring / training
```

**Temperature diversity** (`sample_0` @ 0.5, `sample_1` @ 0.7):

```bash
SKIP_DUEL_SCORE=1 SKIP_PACK=1 \
  TRAIN_N=5000 DPO_SAMPLES=2 BATCH_SIZE=96 \
  DPO_TEMPERATURES=0.5,0.7 \
  SEED_STRIDE=0 \
  ./run/01_prepare_dpo_duel_scored.sh
```

| Setting | Effect |
|---------|--------|
| `DPO_TEMPERATURES=0.5,0.7` | `sample_0` temp 0.5, `sample_1` temp 0.7 |
| `SEED_STRIDE=0` | same seed → **only** temperature differs |
| `SEED_STRIDE=1000` (default) | different seed **and** temperature |

Keep temps ≤ ~0.7; higher values increase invalid JS.
**Outputs:**

```text
data/splits/train.txt
data/images/<stem>.png
data/candidates/shiny_k2/<stem>/sample_0.js
data/candidates/shiny_k2/<stem>/sample_1.js
```

Smoke with fewer prompts first:

```bash
SKIP_DUEL_SCORE=1 SKIP_PACK=1 TRAIN_N=50 ./run/01_prepare_dpo_duel_scored.sh
```

### Phase B — Duel score + pack (OpenRouter + Chromium + DINO)

```bash
source .env && source .venv/bin/activate
: "${OPENROUTER_API_KEY:?set in .env}"

SKIP_COLLECT=1 \
  SIDECAR_COUNT=16 DUEL_CONCURRENCY=8 \
  ./run/01_prepare_dpo_duel_scored.sh
```

Optional smoke:

```bash
SKIP_COLLECT=1 DUEL_LIMIT=100 ./run/01_prepare_dpo_duel_scored.sh
```

**Outputs:**

```text
data/duel_scores/candidate_duels.json
data/hf/dpo_shiny_duel/dataset/          ← train with this
```

Draws and identical JS pairs are skipped when packing (quality filter).

### Dataset quality checklist

```bash
# Candidates exist
find data/candidates/shiny_k2 -name 'sample_0.js' | wc -l
find data/candidates/shiny_k2 -name 'sample_1.js' | wc -l

# Decisive duel pairs
python3 - <<'PY'
import json
from pathlib import Path
p = Path("data/duel_scores/candidate_duels.json")
d = json.loads(p.read_text())
print("n_scored", d.get("n_scored"))
print("n_pairs_for_dpo", d.get("n_pairs_for_dpo"))
print("skipped", d.get("skipped"))
PY

# HF dataset on disk
ls data/hf/dpo_shiny_duel/dataset
```

More detail: [`docs/DPO_DUEL_SCORING.md`](DPO_DUEL_SCORING.md)

---

## Step 5 — Run DPO training

**Do not** run Phase A pipeline and training at the same time on the same GPUs.

```bash
source .env && source .venv/bin/activate

# Profile already set CONFIG + NUM_PROCESSES; override if needed:
CONFIG=configs/dpo_shiny_27b_duel.yaml \
NUM_PROCESSES=4 \
  ./run/03_dpo.sh
```

Training loads the base model from `MODEL_PATH` (local AstroWolf) and the dataset from the path in the yaml (`data/hf/dpo_shiny_duel/dataset`).

**Checkpoint output (typical):**

```text
data/checkpoints/dpo_shiny_27b_duel/
```

Exact output dir is in `configs/dpo_shiny_27b_duel.yaml` (`output_dir`).

### After training (optional merge)

```bash
ADAPTER=data/checkpoints/dpo_shiny_27b_duel/final \
  ./run/04_merge_and_eval.sh
```

(Adjust `ADAPTER` to whatever `output_dir` / final folder your run produced.)

---

## One-command alternative

After `.env` + profile + keys are set:

```bash
INSTALL_SYSTEM=1 ./run/run_all.sh
```

This runs bootstrap (if needed) → install → Phase A/B prep → train. Prefer the **phased** steps above the first time so you can stop the pipeline between generate and score.

---

## Cheap DPO (no OpenRouter judge)

If you only want format/`validate.js` pairs (faster, less subnet-aligned):

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

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `shiny-guide not found` | `./run/00_bootstrap_assets.sh` |
| `config.json` missing under models | Set `HF_TOKEN`, re-run bootstrap, or set `MODEL_PATH` |
| Pipeline never ready | `tail -f pipeline/runs/pipeline-server.log`; check GPUs / `vllm` |
| Phase B fails on OpenRouter | Export `OPENROUTER_API_KEY`; check judge model access |
| OOM during train | Lower `NUM_PROCESSES` or check LoRA yaml batch size |
| Want to re-score only | `SKIP_COLLECT=1 ./run/01_prepare_dpo_duel_scored.sh` |
| Want to re-pack only | `SKIP_COLLECT=1 SKIP_DUEL_SCORE=1 ./run/01_prepare_dpo_duel_scored.sh` |

---

## Doc map

| Doc | When to read |
|-----|----------------|
| **This file** | First run after clone |
| [`CODER_MODEL.md`](CODER_MODEL.md) | Model download / paths only |
| [`DPO_DUEL_SCORING.md`](DPO_DUEL_SCORING.md) | Duel scoring details |
| [`../STANDALONE.md`](../STANDALONE.md) | Standalone layout + self-check |
| [`../MACHINE_PROFILES.md`](../MACHINE_PROFILES.md) | Other GPU profiles |
