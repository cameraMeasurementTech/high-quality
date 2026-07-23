# Training Cookbook: Beat the King (SFT → DPO → GRPO)

> **Standalone (copy only `training/`):** use [`STANDALONE.md`](STANDALONE.md) and
> [`docs/DPO_DUEL_SCORING.md`](docs/DPO_DUEL_SCORING.md).  
> This COOKBOOK still mentions monorepo paths (`local-eval/`, `/home/404-gen-subnet/…`)
> for historical context — **do not follow those paths** on a fresh GPU box.

> **All-in-one:** `./run/00_configure_profile.sh h200x4-dpo-duel` then `INSTALL_SYSTEM=1 ./run/run_all.sh`

End-to-end runbook to improve the **coder VLM** that emits validator-safe
`generate(THREE)` modules for 404-GEN Competition 2.

Companion docs (standalone):

- [`STANDALONE.md`](STANDALONE.md)
- [`docs/DPO_DUEL_SCORING.md`](docs/DPO_DUEL_SCORING.md)
- [`docs/CODER_MODEL.md`](docs/CODER_MODEL.md)
- [`MACHINE_PROFILES.md`](MACHINE_PROFILES.md)

---

## Verdict: SFT, DPO, or GRPO?

| Question | Answer |
|----------|--------|
| Which alone beats the king? | **None alone is enough.** |
| Cheapest first dollar? | **SFT warm-start** on validator-filtered `(image, js)` from your best pipeline / teacher. |
| Best cost per alignment lift? | **DPO** on offline duel pairs or scored candidates (no online rollouts). |
| What optimizes duel win-rate most? | **GRPO** with rewards = validate → render → S1/DINO/S4 (same logic as the subnet). |
| Cost-effective path | **Pipeline/prompt wins** → **SFT** → **DPO (duel pairs)** → optional **GRPO** → Docker audit → live duel. |

**Why not GRPO from cold start?** Cold base models emit invalid JS; group rewards collapse and you burn GPU hours. SFT gets validate+render rate ≥ ~90%, then GRPO has signal.

**Why not SFT forever?** SFT clones teacher style. The subnet scores **pairwise duels**, not teacher likelihood. GRPO’s within-group advantages match that comparative metric.

---

## Platform logic (what you are optimizing)

```
prompts.txt image
    → miner Docker generates {stem}.js
    → validators: validate → multi-view render → DINO
    → pairwise S1–S4 VLM duel vs leader
    → challenger needs margin ≥ ~5% over 128 prompts
    → Docker audit: submitted must NOT beat regenerated (0% margin)
    → winner becomes king (leader.json → on-chain weights)
```

Most production duels end at **S1 (front match)**. Missing/invalid JS is an **auto-loss**. Reliability + front silhouette dominate.

---

## Where the training dataset comes from

There is **no separate labeled train set**. You build it from:

| Asset | Location |
|-------|----------|
| Image pool | Repo `prompts.txt` (~99.6k URLs) or [404-active-competition](https://github.com/404-Repo/404-active-competition) |
| Images | `https://sn12domain.org/procgen/{64hex}.png` |
| Live rounds | `rounds/{N}/prompts.txt` — **exclude** these stems from train |
| Teacher JS | Your best pipeline outputs (`shiny-guide` / `my-agent`) or a frontier teacher, **filtered by `validate.js`** |

This cookbook’s scripts do: **split → download → collect JS → validate filter → pack HF datasets**.

---

## Choose a machine

| Tier | GPUs | What to run | Wall time (order of magnitude) | Ballpark GPU $ |
|------|------|-------------|-------------------------------|----------------|
| **A. Iterate** | 2× H100 80GB | 7–8B QLoRA SFT + GRPO (cheap reward) | SFT 0.5–1d + GRPO 2–4d | $200–1.5k |
| **B. Serious** | 4× H100 | 8B full loop or 27B SFT + light GRPO | 3–7d | $1k–5k |
| **C. Beat king** | 8× H100/H200 | 27B AstroWolf QLoRA SFT→GRPO + GLM judge | 5–12d | $3k–15k |
| **CPU only** | — | Data prep + teacher via OpenRouter; **no** local train | 1–3d data | API $ |

Also need:

- **CPU**: 16–64 cores for `validate.js` / Chromium render farm (Phase-3)
- **Disk**: 1–4 TB SSD (images + reward cache + checkpoints)
- **RAM**: ≥256 GB on the 27B box

**Recommended first run:** Tier A on `Qwen2.5-VL-7B-Instruct`, prove the pipeline, then move adapters / recipe to `handsometiger0202/Qwen-3.6-27B-AstroWolf` (production coder).

---

## End-to-end pipeline

```text
0 setup
1 prepare_data   splits → images → teacher JS → validate → HF pack (SFT + GRPO prompts)
1b prepare_dpo   duel pairs OR scored candidates → HF DPO pack
2 sft            LoRA/QLoRA warm-start
2b dpo           offline preference optimization (optional, recommended)
3 grpo           online rollouts + validator reward (optional polish)
4 merge + duel   merge LoRA → Docker → run-duel-seq vs shiny-guide
5 ship           audit rehearsal → live round CDN
```

All commands below assume:

```bash
cd /home/404-gen-subnet/my-agent/training
```

---

## Step 0 — Setup (30–90 min)

```bash
cd /home/404-gen-subnet/my-agent/training
./run/00_setup.sh
source .venv/bin/activate

# GPU box: install matching CUDA torch BEFORE / instead of the CPU wheel
# pip install torch --index-url https://download.pytorch.org/whl/cu124
# pip install -r requirements.txt

# One-time: miner validator node deps
cd /home/404-gen-subnet/miner-reference/validator && npm install && cd -
```

Optional: HuggingFace login for gated DINOv3 / private bases:

```bash
huggingface-cli login
export HF_TOKEN=...
```

---

## Step 1 — Prepare training dataset (hours → 1–2 days)

### 1a. Default path (harvest existing local-eval JS — cheapest)

If you already ran local duels / pool evals:

```bash
cd /home/404-gen-subnet/my-agent/training
source .venv/bin/activate

# Smaller first pass (recommended)
TRAIN_N=2000 VAL_N=200 DUEL_N=100 SEED=7 \
TEACHER_MODE=runs \
./run/01_prepare_data.sh
```

This writes:

| Path | Contents |
|------|----------|
| `data/splits/{train,val,duel}.txt` | Non-overlapping stem lists |
| `data/images/{stem}.png` | Cached prompts |
| `data/raw_js/teacher/*.js` | Harvested teacher code |
| `data/filtered_js/*.js` | validate.js **PASSED** only |
| `data/hf/sft_train/` | SFT HF dataset + jsonl |
| `data/hf/grpo_train/` | GRPO prompt-only dataset |

### 1b. Generate teacher JS from a running pipeline (best quality/cost)

```bash
# Terminal 1 — start best teacher pipeline (shiny-guide or tuned my-agent)
cd /home/404-gen-subnet
source local-eval/.env
./local-eval/run-pipeline-cpu.sh          # :10006
# OR GPU Docker miner with AstroWolf

# Terminal 2 — collect over train stems
cd /home/404-gen-subnet/my-agent/training
source .venv/bin/activate
TRAIN_N=5000 VAL_N=300 DUEL_N=150 \
TEACHER_MODE=pipeline PIPELINE_URL=http://127.0.0.1:10006 \
./run/01_prepare_data.sh
```

### 1c. Frontier teacher via OpenRouter (fast bootstrap, watch $)

```bash
export OPENROUTER_API_KEY=sk-or-...
TRAIN_N=3000 TEACHER_MODE=openai \
TEACHER_MODEL=google/gemini-2.5-pro-preview \
./run/01_prepare_data.sh
```

### 1d. Manual one-liners (if you prefer not to use the wrapper)

```bash
export PYTHONPATH=$PWD/scripts

python scripts/prepare_splits.py --train 10000 --val 500 --duel 200 --seed 7 \
  --out-dir data/splits

python scripts/download_images.py --list data/splits/train.txt --out data/images

python scripts/collect_teacher_js.py --from-pipeline \
  --list data/splits/train.txt --base-url http://127.0.0.1:10006 \
  --out data/raw_js/shiny-guide

python scripts/filter_validate.py --js-dir data/raw_js/shiny-guide \
  --out-dir data/filtered_js --copy-fail

python scripts/pack_sft_dataset.py --js-dir data/filtered_js \
  --images data/images --list data/splits/train.txt --out data/hf/sft_train

python scripts/pack_grpo_prompts.py --list data/splits/train.txt \
  --images data/images --out data/hf/grpo_train
```

**Quality bar before SFT:** `filtered_js` pass rate should be high on the teacher set (teacher itself should already pass). Aim for **≥5k** clean SFT pairs for 8B; **≥10k** for 27B.

Exclude recent live round stems:

```bash
curl -sL "https://raw.githubusercontent.com/404-Repo/404-active-competition/main/rounds/N/prompts.txt" \
  > /tmp/roundN.txt
python scripts/prepare_splits.py --train 10000 --val 500 --duel 200 \
  --exclude /tmp/roundN.txt --out-dir data/splits
```

---

## Step 2 — SFT warm-start (0.5–2 days)

### 8B iterate (Tier A)

```bash
cd /home/404-gen-subnet/my-agent/training
source .venv/bin/activate
CONFIG=configs/sft_8b.yaml ./run/02_sft.sh
```

### 27B production coder (Tier B/C)

```bash
CONFIG=configs/sft_27b.yaml NUM_PROCESSES=1 ./run/02_sft.sh
```

Or dry-run 64 samples: set `max_samples: 64` in the yaml.

**Exit criteria:** on `data/splits/val.txt`, validate+render rate ≥ **~90%** before spending on DPO/GRPO.

Smoke-check a single reward:

```bash
python scripts/reward.py --js data/filtered_js/<stem>.js --image data/images/<stem>.png
# expect reward > 0 for valid JS; -2.0 for fails
```

---

## Step 2b — DPO (0.5–1 day, cheaper than GRPO)

DPO learns from **offline chosen vs rejected** JS pairs. Full detail: [`docs/TRAINING_DPO_STRATEGY.md`](../../docs/TRAINING_DPO_STRATEGY.md).

### Path A — Duel pairs (recommended first)

Run a local duel first (see Step 4), then:

```bash
SOURCE=duel PREFER_LABEL=shiny-guide ONLY_LOSSES=1 \
  ./run/01_prepare_dpo_data.sh
```

### Path B — Score-mined pairs (K samples per stem)

```bash
export OPENROUTER_API_KEY=sk-or-...
MODE=openai SAMPLES=4 LIMIT=500 ./run/01_collect_candidates.sh

SOURCE=candidates REWARD_MODE=cheap MIN_MARGIN=0.15 \
  ./run/01_prepare_dpo_data.sh
```

### Train DPO

```bash
CONFIG=configs/dpo_8b.yaml ./run/03_dpo.sh
```

**Exit criteria:** held-out duel win-rate improves **+3–5 pp** vs SFT-only before spending on GRPO.

---

## Step 3 — GRPO (2–10 days)

### Phase 2 — cheap rewards (validate + format) — **start here**

```bash
CONFIG=configs/grpo_8b.yaml ./run/03_grpo.sh
```

### Phase 2b — add S1 proxy judge

Self-host GLM (matches subnet) or use OpenRouter for dry-runs only:

```bash
# Example: local GLM vLLM already used by my-agent critic
export JUDGE_BASE_URL=http://127.0.0.1:8002/v1
export JUDGE_MODEL=zai-org/GLM-4.6V-Flash
# edit configs/grpo_8b.yaml: reward_mode: s1
CONFIG=configs/grpo_8b.yaml ./run/03_grpo.sh
```

### Phase 3 — 27B + GLM rewards

```bash
CONFIG=configs/grpo_27b.yaml ./run/03_grpo.sh
```

**Knobs that matter:**

| Knob | Start value | Notes |
|------|-------------|-------|
| `num_generations` (G) | 4 | 8 if VRAM allows |
| `max_completion_length` | 12k–16k | Three.js modules are long |
| `beta` (KL) | 0.03–0.05 | Too low → invalid JS returns |
| `temperature` | 0.7–0.9 | Need within-group diversity |
| `learning_rate` | 5e-7–1e-6 | Lower than SFT |

---

## Step 4 — Merge, wire miner, duel (0.5–1 day)

```bash
BASE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
ADAPTER=data/checkpoints/grpo_8b/final \
MERGED=data/checkpoints/merged_coder \
./run/04_merge_and_eval.sh
```

For 27B:

```bash
BASE_MODEL=handsometiger0202/Qwen-3.6-27B-AstroWolf \
ADAPTER=data/checkpoints/grpo_27b/final \
MERGED=data/checkpoints/merged_coder_27b \
./run/04_merge_and_eval.sh
```

Edit `my-agent/configuration.yaml` so coder/planner vLLM points at the merged path, rebuild:

```bash
cd /home/404-gen-subnet/my-agent
docker build -f docker/Dockerfile -t my-404-miner .
```

Held-out duel vs leader baseline:

```bash
# Term A: shiny-guide :10006
source /home/404-gen-subnet/local-eval/.env
./local-eval/run-pipeline-cpu.sh

# Term B: my-agent :10007 (GPU Docker or CPU profile with new weights)
./local-eval/run-pipeline-my-agent.sh

# Term C
./local-eval/run-duel-seq.sh \
  /home/404-gen-subnet/my-agent/training/data/splits/duel.txt \
  --limit 100 --fresh
```

**Ship gate:** decisive win-rate vs `shiny-guide` **> 55–60%** on held-out duel, then audit rehearsal (Docker regen must not lose to CDN upload at 0% margin).

---

## Step 5 — Go live

1. Pin open commercial-use weights + commit hash in the public miner repo.
2. Serve on `:10006` per subnet rules.
3. Generate all 128 round stems; validate each locally:
   ```bash
   node miner-reference/tools/validate.js path/to/{stem}.js
   ```
4. Upload `{cdn_url}/{stem}.js` before deadline.
5. Watch [404-active-competition](https://github.com/404-Repo/404-active-competition) for duel outcomes.

---

## Cost-effective 14-day plan

| Day | Work | Spend focus |
|-----|------|-------------|
| 1 | Pipeline/prompt duel loop (`IMPROVEMENT_PLAN.md`) — free wins | $0 GPU |
| 2–3 | Data: 3–5k teacher JS via pipeline + validate filter | CPU + API |
| 4–5 | SFT 8B LoRA | 2× H100 |
| 6–8 | GRPO 8B cheap → s1 | 2–4× H100 |
| 9 | Merge + 100-prompt duel; decide if 27B worth it | — |
| 10–13 | SFT+GRPO on AstroWolf 27B if 8B shows lift | 4–8× H100 |
| 14 | Docker audit + ship | — |

**Do not** jump to pairwise group-duel rewards until Phase-2 moves the duel needle — they roughly double reward cost.

---

## Directory map

```
my-agent/training/
  COOKBOOK.md          ← this file
  requirements.txt
  configs/             ← sft_*.yaml, dpo_*.yaml, grpo_*.yaml
  scripts/             ← prepare / train / reward / merge / dpo pack
  run/                 ← 00–04 + 01_collect_candidates, 01_prepare_dpo_data, 03_dpo
  data/                ← created by scripts (gitignored ideally)
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `validate.js` fails to load | `cd miner-reference/validator && npm install` |
| SFT OOM | `load_in_4bit: true`, lower `max_seq_length`, batch=1 |
| GRPO all rewards ≈ -2 | Raise SFT quality; increase fail penalty already set; mix SFT replay |
| Reward hacking (white blobs) | Move `reward_mode` to `s1` / add render; never stay on format-only |
| Audit fail | Always generate CDN JS with the **same Docker weights** you ship |
| Judge drift | Final GRPO with self-hosted `GLM-4.6V-Flash`, not Gemini |

---

## Bottom line

1. **Dataset** = `prompts.txt` images + validator-filtered teacher JS (scripts in this folder).  
2. **Train** = **SFT first**, then **GRPO** with validator-shaped rewards.  
3. **Machines** = start **2× H100 / 8B**, promote to **27B AstroWolf on 4–8× H100**.  
4. **Measure** = held-out `run-duel-seq.sh` vs `shiny-guide`, then Docker audit.  
5. Beating the king is **reliability + S1 front match + side depth (S4)** — train the metric the subnet uses.
