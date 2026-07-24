# SFT LoRA from OpenRouter GPT-5 (miner-identical coder inputs) — full guide

**Source of truth:** `data/prompts.txt` (~99k `https://…png` URLs) — the **same**
pool used by duel/DPO prep (`PROMPTS_POOL` after `./run/00_bootstrap_assets.sh`).

```text
data/prompts.txt          ← bootstrap (~99610 image URLs)
       │
       ▼  prepare_splits --train-remainder  (default FULL_POOL=1)
data/splits/train.txt     ← ~98.9k stems (minus val+duel holdouts)
       │
       ▼  download PNGs → GPT-5 teacher (miner prompts) → validate.js
data/hf/sft_gpt_teacher/dataset
       │
       ▼  LoRA SFT on AstroWolf (4× H200)
```

**Why LoRA:** AstroWolf is already a strong coder; GPT-5 is a teacher distill.
LoRA (`r=32`) fits ~50–80k validated samples on 4× H200 without overwriting the base.

Profile: `h200x4-sft-gpt` (sets `FULL_POOL=1` by default).

---

## Machine requirements

### A) Dataset prep (OpenRouter teacher) — no GPU required

| Resource | Recommendation |
|----------|----------------|
| CPU / RAM | ≥16 cores, ≥64 GB RAM |
| Disk | **≥500 GB free** (≈99k PNGs + raw/filtered JS + HF dataset) |
| Network | Fast / stable (image CDN + OpenRouter) |
| GPU | **Not needed** for prep |
| Keys | `OPENROUTER_API_KEY`, `HF_TOKEN` (for later train / model download) |

Prep can run on a CPU box; copy `data/hf/sft_gpt_teacher/` + `data/models/` to the GPU box for training, or do everything on the 4× H200 machine.

### B) LoRA training — 4× H200 (recommended)

| Resource | Recommendation |
|----------|----------------|
| GPUs | **4× H200** (bf16 LoRA 27B) |
| Disk | Model ~54 GB + dataset + checkpoints |
| Config | `configs/sft_shiny_27b_gpt_teacher.yaml` |
| Processes | `NUM_PROCESSES=4` |

---

## One-time setup

```bash
cd training
cp .env.template .env
# Edit .env:
#   HF_TOKEN=hf_...
#   OPENROUTER_API_KEY=sk-or-v1-...

./run/00_configure_profile.sh h200x4-sft-gpt
./run/00_bootstrap_assets.sh          # prompts.txt (~99k) + AstroWolf + shiny-guide
INSTALL_SYSTEM=1 ./run/00_install_all.sh
source .env && source .venv/bin/activate
./run/00_verify_standalone.sh
```

Confirm pool:

```bash
wc -l data/prompts.txt          # expect ~99610
grep -E '^PROMPTS_POOL=' .env   # absolute path under training/
```

---

## Phase 1 — Prepare dataset from prompts.txt (~99k)

Always starts from **`PROMPTS_POOL`** (= `data/prompts.txt` after bootstrap), not a
hand-built list.

### Smoke (optional) — subsample from the same prompts.txt

```bash
source .env && source .venv/bin/activate
FULL_POOL=0 FORCE_SPLITS=1 TRAIN_N=50 TEACHER_WORKERS=4 \
  ./run/01_prepare_sft_openrouter.sh
```

### Full pool (default) — almost all of prompts.txt

```bash
source .env && source .venv/bin/activate

# FULL_POOL defaults to 1 — train = prompts.txt minus VAL_N+DUEL_N
./run/01_prepare_sft_openrouter.sh

# explicit equivalent:
FULL_POOL=1 VAL_N=500 DUEL_N=200 \
TEACHER_MODEL=openai/gpt-5-chat \
TEACHER_WORKERS=16 DL_WORKERS=64 \
FORCE_SPLITS=1 \
  ./run/01_prepare_sft_openrouter.sh
```

What default / `FULL_POOL=1` does:

- Reads **`data/prompts.txt`** via `PROMPTS_POOL` (same file as duel prep)
- `--train-remainder` → **train ≈ 99k − 500 − 200**
- Skips dumping 99k `*.request.json` (disk); tiny miner-input sample still written
- Downloads train+val images with `--fail-ok`
- Teacher collect **resumes** (`--skip-existing`)

| Step | Time (rough) | Notes |
|------|----------------|-------|
| Splits | minutes | from `data/prompts.txt` |
| Image download | hours | ~99k PNGs; resume-safe |
| GPT-5 teacher | **days** | rate limits / cost dominate; raise `TEACHER_WORKERS` carefully |
| validate.js | hours | Node workers |
| Pack HF | tens of minutes | |

**Resume after interrupt**

```bash
# Same command — skips existing JS / images
FULL_POOL=1 TEACHER_WORKERS=16 ./run/01_prepare_sft_openrouter.sh

# Or skip finished stages:
SKIP_TEACHER=1 ./run/01_prepare_sft_openrouter.sh   # only validate+pack
SKIP_VALIDATE=1 SKIP_TEACHER=1 ./run/01_prepare_sft_openrouter.sh  # only pack
```

### Outputs

| Path | Content |
|------|---------|
| `data/splits/train.txt` | ~98.9k `stem\turl` |
| `data/splits/val.txt` / `duel.txt` | Holdouts — **do not train on duel** |
| `data/images/{stem}.png` | Reference images |
| `data/raw_js/gpt_teacher/` | Raw GPT-5 JS |
| `data/filtered_js/gpt_teacher/` | `validate.js` pass |
| `data/hf/sft_gpt_teacher/dataset` | HF SFT for LoRA |
| `data/miner_inputs/sft/_prompt_meta.json` | Miner prompt parity meta |

Expect **filtered N ≪ 99k** (often ~40–80% pass). Training uses filtered rows only.

### Sanity checks

```bash
python - <<'PY'
import json
from pathlib import Path
from datasets import load_from_disk
print("manifest", json.loads(Path("data/splits/manifest.json").read_text())["counts"])
ds = load_from_disk("data/hf/sft_gpt_teacher/dataset")
print("sft_n", len(ds))
print("sample keys", ds[0].keys())
PY
```

---

## Phase 2 — Train LoRA on AstroWolf (4× H200)

LoRA config (`configs/sft_shiny_27b_gpt_teacher.yaml`) is tuned for large SFT:

| Knob | Value | Why |
|------|-------|-----|
| `use_lora` | `true` | Distill without full FT |
| `lora_r` / `lora_alpha` | 32 / 64 | Capacity for tens of thousands of examples |
| `mask_prompt` | `true` | Loss on assistant JS only |
| `learning_rate` | `5e-6` | Stable for large LoRA SFT |
| `num_train_epochs` | `1` | One pass over ~50–80k is usually enough |
| `gradient_accumulation_steps` | `8` | With 4 GPUs → global batch ≈ 32 |
| `save_steps` | `500` | Fewer checkpoints on long runs |

```bash
source .env && source .venv/bin/activate

# Uses MODEL_PATH from bootstrap when set
CONFIG=configs/sft_shiny_27b_gpt_teacher.yaml \
NUM_PROCESSES=4 \
  ./run/02_sft.sh
```

Adapter output:

```text
data/checkpoints/sft_shiny_27b_gpt/final
```

**Do not** run the OpenRouter teacher job and LoRA train on the same GPUs at once if prep is also on the GPU box (prep is CPU/API-bound; train needs all 4 GPUs).

### Optional second epoch

If metrics still climb after 1 epoch, set in the yaml:

```yaml
num_train_epochs: 2
```

or re-run with a lower LR (`learning_rate: 2.0e-6`).

---

## Phase 3 — (Optional) merge + subnet DPO

```bash
ADAPTER=data/checkpoints/sft_shiny_27b_gpt/final ./run/04_merge_and_eval.sh
```

Then align to validators with **AstroWolf self-sample duel DPO** (not GPT pairs):

```bash
./run/00_configure_profile.sh h200x4-dpo-duel
# Phase A/B/C as in docs/DPO_DUEL_SCORING.md
```

---

## Miner parity (what the teacher sees)

Same as `SceneCoderAgent.code()` with `use_planner=false`:

1. **system:** `CODER_SYSTEM_PROMPT`
2. **user:** `[ image_url(data:image/png;base64,…), text(CODER_USER_TEMPLATE_IMAGE_ONLY) ]`
3. **assistant (train target):** validated teacher JS

SFT packing uses the same system/user strings; the trainer injects the PNG and masks prompt tokens.

---

## Env reference

| Variable | Default | Meaning |
|----------|---------|---------|
| `FULL_POOL` | **`1`** | Use entire `prompts.txt` minus val/duel (default) |
| `TRAIN_N` | — | Only when `FULL_POOL=0` (subsample for smoke) |
| `VAL_N` / `DUEL_N` | `500` / `200` | Holdouts |
| `TEACHER_MODEL` | `openai/gpt-5-chat` | OpenRouter vision model |
| `TEACHER_WORKERS` | `16` | Parallel teacher calls |
| `TEACHER_TEMPERATURE` | `0.4` | Keep moderate for valid JS |
| `DL_WORKERS` | `64` | Image download threads |
| `SKIP_EXPORT` | `1` if `FULL_POOL` | Avoid writing 99k JSON dumps |
| `FORCE_SPLITS` | `1` if `FULL_POOL` | Rebuild splits |
| `NUM_PROCESSES` | `4` | LoRA train GPUs |

---

## Cost / ops tips

- OpenRouter GPT-5 on ~99k vision completions is **expensive** — smoke 50 first, then ramp.
- Re-run teacher with the same `RAW_JS` dir; existing `*.js` are skipped.
- Keep `data/splits/duel.txt` forever out of SFT and later DPO train lists.
- Prefer **LoRA → duel DPO**; do not full-FT AstroWolf unless LoRA plateaus on 8× H200.
