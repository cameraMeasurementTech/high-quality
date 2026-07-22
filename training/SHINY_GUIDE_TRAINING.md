# Shiny-guide training guide (ignore my-agent)

Train the **404-GEN king coder** (`shiny-guide`) on validator prompts. All prompts, teacher JS, and base weights come from **shiny-guide only**.

Training scripts live in `my-agent/training/` (reusable harness). Set `PROMPTS_ROOT=shiny-guide` (now the default).

---

## What you are training

| Item | shiny-guide value |
|------|-------------------|
| Coder model (GPU) | `Tooony133/Qwen-3.6-27B-AstroWolf` |
| Coder prompts | `shiny-guide/pipeline_service/modules/scene_coder/prompts.py` |
| Validator prompts | `prompts.txt` (~99.6k) → `https://sn12domain.org/procgen/{stem}.png` |
| Serve API | `POST /generate` batch on `:10006` |

**Do not use CPU OpenRouter eval config for training data** — that runs Gemini, not AstroWolf. For production-faithful JS, use **GPU Docker** with `shiny-guide/configuration.yaml`.

---

**AstroWolf is already SFT-specialized in production.** You can skip SFT and go straight to **DPO** or **GRPO** with a fresh LoRA on `Tooony133/Qwen-3.6-27B-AstroWolf` (no `sft_adapter_path` in config).

---

## Skip SFT — DPO or GRPO only (recommended for you)

### Choose DPO vs GRPO

| | **DPO** | **GRPO** |
|---|---------|----------|
| Data | Offline chosen/rejected JS pairs | Prompt-only; samples completions during training |
| GPU cost | Lower | Higher (rollouts + reward each step) |
| When | You can run shiny-guide K× per prompt | You have reward infra (validate + optional GLM judge) |
| Config | `configs/dpo_shiny_27b.yaml` | `configs/grpo_shiny_27b.yaml` |

Both configs load **base AstroWolf directly** — no SFT checkpoint required.

### 1. Data prep (no SFT pack)

```bash
cd /home/404-gen-subnet/my-agent/training
source .venv/bin/activate
export PROMPTS_ROOT=shiny-guide

# shiny-guide GPU docker on :10006 first

# DPO only:
ALIGN=dpo TRAIN_N=5000 ./run/01_prepare_shiny_align.sh

# GRPO only:
ALIGN=grpo TRAIN_N=5000 ./run/01_prepare_shiny_align.sh

# Both datasets (pick one training method later):
ALIGN=both TRAIN_N=5000 ./run/01_prepare_shiny_align.sh
```

### 2. Train

```bash
# DPO — offline preference pairs
CONFIG=configs/dpo_shiny_27b.yaml ./run/03_dpo.sh

# OR GRPO — online validator rewards
CONFIG=configs/grpo_shiny_27b.yaml ./run/03_grpo.sh
# Phase 1: reward_mode: cheap in yaml
# Phase 2: set reward_mode: s1 + JUDGE_BASE_URL / GLM judge
```

### 3. Merge & deploy into shiny-guide

```bash
BASE_MODEL=Tooony133/Qwen-3.6-27B-AstroWolf \
ADAPTER=data/checkpoints/dpo_shiny_27b/final \
MERGED=data/checkpoints/merged_shiny_coder \
./run/04_merge_and_eval.sh
# (or grpo_shiny_27b/final if you ran GRPO)
```

Point `shiny-guide/configuration.yaml` coder model at `MERGED`, rebuild Docker.

---

## Pipeline overview (full path, if you ever need SFT)

```text
validator prompts (prompts.txt)
  → download images
  → K samples/prompt       →  score → DPO pairs (chosen vs rejected)
  → OR prompt-only pack    →  GRPO online rollouts
  → LoRA on AstroWolf (skip SFT — already trained)
  → merge → deploy back into shiny-guide Docker
```

---

## Step 0 — Setup

```bash
cd /home/404-gen-subnet/my-agent/training
./run/00_setup.sh
source .venv/bin/activate
export PROMPTS_ROOT=shiny-guide   # default; explicit for clarity
```

---

## Step 1 — Start shiny-guide (GPU recommended)

### Option A — GPU Docker (production models)

```bash
cd /home/404-gen-subnet/shiny-guide/docker
docker compose up --build
# serves :10006 with Tooony133/Qwen-3.6-27B-AstroWolf
```

### Option B — CPU OpenRouter (quick test only, NOT for final training data)

```bash
cd /home/404-gen-subnet
source local-eval/.env
./local-eval/run-pipeline-cpu.sh   # Gemini via OpenRouter — wrong model for SFT target
```

Use Option A for real training data.

---

## Step 2 — Build datasets (no SFT training)

```bash
cd /home/404-gen-subnet/my-agent/training
source .venv/bin/activate
export PROMPTS_ROOT=shiny-guide

# shiny-guide GPU on :10006
ALIGN=dpo TRAIN_N=5000 VAL_N=300 DUEL_N=200 \
PIPELINE_URL=http://127.0.0.1:10006 DPO_SAMPLES=4 \
./run/01_prepare_shiny_align.sh
```

For GRPO prompts instead: `ALIGN=grpo ./run/01_prepare_shiny_align.sh`

Outputs:

| Path | Contents |
|------|----------|
| `data/hf/dpo_shiny/dataset` | DPO: chosen=better JS, rejected=worse JS per prompt |
| `data/hf/grpo_shiny/dataset` | GRPO: validator image + prompt only (if `ALIGN=grpo`) |

### Alternative: harvest existing eval runs for DPO candidates

If you already have shiny-guide JS in run dirs, collect candidates manually — see manual steps below.

---

## Step 3 — Manual DPO dataset (if you want control)

### 3a. Validator stems + images

```bash
export PYTHONPATH=$PWD/scripts
export PROMPTS_ROOT=shiny-guide

python scripts/prepare_splits.py \
  --pool /home/404-gen-subnet/prompts.txt \
  --train 5000 --val 300 --duel 200 --seed 7 \
  --out-dir data/splits

python scripts/download_images.py \
  --list data/splits/train_val.txt \
  --out data/images --workers 16
```

### 3b. Collect K JS candidates from shiny-guide

```bash
python scripts/collect_candidates.py --from-pipeline \
  --list data/splits/train.txt \
  --base-url http://127.0.0.1:10006 \
  --samples 4 \
  --out data/candidates/shiny_k4
```

Each stem gets `data/candidates/shiny_k4/{stem}/sample_0.js … sample_3.js`.

### 3c. Pack DPO pairs (best vs worst by validator reward)

```bash
python scripts/pack_dpo_dataset.py --source candidates \
  --candidates-dir data/candidates/shiny_k4 \
  --list data/splits/train.txt \
  --images data/images \
  --reward-mode cheap \
  --min-margin 0.15 \
  --out data/hf/dpo_shiny
```

Check quality:

```bash
cat data/hf/dpo_shiny/meta.json
head -1 data/hf/dpo_shiny/dpo.jsonl | python -m json.tool
```

**DPO row shape:**

```json
{
  "prompt": [/* shiny-guide system + user + image */],
  "chosen": [{"role": "assistant", "content": [{"type": "text", "text": "export default function generate(THREE) { … better … }"}]}],
  "rejected": [{"role": "assistant", "content": [{"type": "text", "text": "export default function generate(THREE) { … worse … }"}]}]
}
```

### 3d. Alternative DPO source — pass vs fail (bootstrap)

```bash
python scripts/pack_dpo_dataset.py --source dirs \
  --chosen-dir data/filtered_js/shiny-guide \
  --rejected-dir data/raw_js/shiny-guide \
  --list data/splits/train.txt \
  --images data/images \
  --out data/hf/dpo_shiny_passfail
```

---

## Step 4 — Train (skip `./run/02_sft.sh` — AstroWolf already specialized)

```bash
# DPO
CONFIG=configs/dpo_shiny_27b.yaml ./run/03_dpo.sh

# OR GRPO
CONFIG=configs/grpo_shiny_27b.yaml ./run/03_grpo.sh
```

Only run SFT if you are training from a **non-specialized** base:

```bash
CONFIG=configs/sft_shiny_27b.yaml ./run/02_sft.sh   # usually skip
```

---

## Step 5 — Deploy improved weights into shiny-guide

```bash
BASE_MODEL=Tooony133/Qwen-3.6-27B-AstroWolf \
ADAPTER=data/checkpoints/dpo_shiny_27b/final \
MERGED=data/checkpoints/merged_shiny_coder \
./run/04_merge_and_eval.sh
```

Edit `shiny-guide/configuration.yaml`:

```yaml
llm_clients:
  coder-instance:
    vllm:
      model: "/path/to/data/checkpoints/merged_shiny_coder"
actors:
  coder:
    model: "/path/to/data/checkpoints/merged_shiny_coder"
  planner:
    model: "/path/to/data/checkpoints/merged_shiny_coder"
```

Rebuild Docker:

```bash
cd /home/404-gen-subnet/shiny-guide/docker
docker compose build && docker compose up
```

Eval on held-out validator stems (never used in train):

```bash
./local-eval/run-eval.sh \
  my-agent/training/data/splits/duel.txt --limit 100 --name shiny-eval
```

---

## Reward modes for DPO pair mining

| `REWARD_MODE` | What it scores | When to use |
|---------------|----------------|-------------|
| `cheap` | validate.js + format heuristics | Fast iteration |
| `s1` | GLM front-match proxy | Closer to validator S1 |
| `render` | HTTP render service | Needs render sidecar |

```bash
export JUDGE_BASE_URL=http://127.0.0.1:8002/v1
export JUDGE_MODEL=zai-org/GLM-4.6V-Flash
REWARD_MODE=s1 MIN_MARGIN=0.20 ./run/01_prepare_shiny_data.sh
```

---

## Important rules

1. **Train on validator pool, eval on `duel.txt`** — never pack DPO from eval stems.
2. **Exclude live rounds from train** — pass `--exclude rounds/N/prompts.txt` to `prepare_splits.py`.
3. **Prompt parity** — always `PROMPTS_ROOT=shiny-guide` when packing/training.
4. **GPU teacher for GPU student** — AstroWolf JS from AstroWolf pipeline, not Gemini CPU path.
5. **Audit** — after deploy, regenerated Docker output must match CDN upload (0% margin rule).

---

## Quick reference

```bash
# Full shiny-guide alignment prep (no SFT training)
PROMPTS_ROOT=shiny-guide ALIGN=dpo ./run/01_prepare_shiny_align.sh
CONFIG=configs/dpo_shiny_27b.yaml ./run/03_dpo.sh

# OR GRPO
PROMPTS_ROOT=shiny-guide ALIGN=grpo ./run/01_prepare_shiny_align.sh
CONFIG=configs/grpo_shiny_27b.yaml ./run/03_grpo.sh
```

See also: [`docs/TRAINING_DPO_STRATEGY.md`](../../docs/TRAINING_DPO_STRATEGY.md) for DPO theory and hyperparameters.
