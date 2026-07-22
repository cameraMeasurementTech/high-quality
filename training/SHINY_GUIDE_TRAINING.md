# Shiny-guide training guide

Train the **404-GEN king coder** (`shiny-guide`) on **validator prompts** using **DPO** or **GRPO**.

- Base model: `Tooony133/Qwen-3.6-27B-AstroWolf` (already SFT-specialized — **skip SFT**)
- Prompts: `shiny-guide/pipeline_service/modules/scene_coder/prompts.py`
- Scripts: `my-agent/training/` (set `PROMPTS_ROOT=shiny-guide`, default)

Strategy background: [`docs/TRAINING_DPO_STRATEGY.md`](../../docs/TRAINING_DPO_STRATEGY.md)

---

## Fresh GPU machine — step-by-step (start here)

Complete walkthrough from a **blank Ubuntu GPU box** to a trained LoRA checkpoint.

Replace `/home/404-gen-subnet` with your clone path if different.

### Timeline overview

| Phase | What | Wall time (order of magnitude) |
|-------|------|--------------------------------|
| 0 | Machine + drivers | 30–60 min |
| 1 | Clone repo + system deps | 20–40 min |
| 2 | Python training env (CUDA torch) | 20–60 min |
| 3 | Start shiny-guide (data generation) | 1–3 h first boot (model download) |
| 4 | Prepare dataset (DPO or GRPO) | 4–48 h (depends on `TRAIN_N`, pipeline speed) |
| 5 | Run DPO or GRPO training | 6–48 h (27B QLoRA) |
| 6 | Merge LoRA + deploy | 1–2 h |

---

### Phase 0 — Hardware and OS

**Minimum for 27B DPO/GRPO (QLoRA):**

| Resource | Recommendation |
|----------|----------------|
| GPU | 2× H100 80GB (train) **or** 4× A100 80GB |
| CPU RAM | ≥ 128 GB |
| Disk | ≥ 2 TB SSD (HF cache + datasets + checkpoints) |
| OS | Ubuntu 22.04 / 24.04 |

**For dataset generation (shiny-guide Docker on same or second box):**

| Resource | Recommendation |
|----------|----------------|
| GPU | 2× GPU for AstroWolf vLLM (matches shiny-guide `configuration.yaml`) |
| Port | `:10006` free for pipeline API |

Check GPU driver:

```bash
nvidia-smi
# Driver ≥ 535 recommended for CUDA 12.x
```

Install Docker + NVIDIA container toolkit (**only if you will use Docker** — skip on nested GPU containers):

```bash
# Optional — skip if you cannot run Docker
# curl -fsSL https://get.docker.com | sh
```

Install system packages (required for **native** shiny-guide — Chromium renderer sidecars):

```bash
sudo apt update
sudo apt install -y git curl wget build-essential python3 python3-venv python3-pip \
  nodejs npm \
  fonts-liberation libnss3 libatk-bridge2.0-0 libx11-xcb1 libxcomposite1 \
  libxdamage1 libxrandr2 libgbm1 libasound2 libpangocairo-1.0-0 libgtk-3-0 \
  libegl1 libgl1-mesa-dri libgles2 xvfb
node --version   # need ≥ 20 for validate.js + renderer sidecars
```

If `node --version` is below 20, install Node 20 from [nodejs.org](https://nodejs.org/) or NodeSource.

---

### Phase 1 — Clone the repo

```bash
cd /home
git clone https://github.com/cameraMeasurementTech/high-quality.git 404-gen-subnet
# Or clone your fork; ensure it contains:
#   404-gen-subnet/prompts.txt
#   404-gen-subnet/shiny-guide/
#   404-gen-subnet/my-agent/training/
#   404-gen-subnet/miner-reference/

cd /home/404-gen-subnet
ls prompts.txt shiny-guide my-agent/training
```

Set a convenience variable (add to `~/.bashrc`):

```bash
export REPO=/home/404-gen-subnet
export TRAINING=$REPO/my-agent/training
```

---

### Phase 2 — Training Python environment

```bash
cd $TRAINING

# Node deps for validate.js (DPO reward + GRPO reward)
cd $REPO/miner-reference/validator && npm install && cd $TRAINING

# Create venv
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

# IMPORTANT: install CUDA torch BEFORE requirements.txt
# Pick the index matching your driver (see https://pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Training stack (trl, peft, bitsandbytes, vllm for GRPO, …)
pip install -r requirements.txt
```

HuggingFace login (AstroWolf + gated models):

```bash
pip install huggingface_hub
huggingface-cli login
# paste HF token with read access to Tooony133/Qwen-3.6-27B-AstroWolf
export HF_HOME=$TRAINING/data/hf_cache
export HF_TOKEN=hf_...   # optional; CLI login is enough
```

Verify install:

```bash
source $TRAINING/.venv/bin/activate
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())"
python -c "import trl, peft, transformers; print('trl', trl.__version__)"
export PYTHONPATH=$TRAINING/scripts PROMPTS_ROOT=shiny-guide
python -c "from coder_prompts import load_coder_prompts; print('prompts OK', len(load_coder_prompts()[0]))"
node $REPO/miner-reference/tools/validate.js --help 2>&1 | head -3
```

Expected: `cuda: True`, imports OK, prompts load without error.

Run the bundled setup script (optional — does venv + npm + dirs):

```bash
cd $TRAINING
./run/00_setup.sh
source .venv/bin/activate
# Still install CUDA torch first if 00_setup.sh pulled CPU torch
```

Create data directories:

```bash
mkdir -p $TRAINING/data/{splits,images,candidates,hf,checkpoints,logs}
```

---

### Phase 3 — Start shiny-guide (for dataset generation)

Training data must use **AstroWolf on GPU** for the **coder**. Do **not** use `run-pipeline-cpu.sh` (Gemini via OpenRouter) for DPO pair mining — that is a different model.

Pick **one** path below.

---

#### Option A — Native GPU (recommended when Docker is unavailable)

Use this on **GPU cloud containers**, bare metal, or any host where `docker compose` is not available.

**Terminal 1 — keep running**

```bash
export REPO=/home/404-gen-subnet
cd $REPO

# One-time native setup (Node sidecars + vLLM venv + pipeline Python deps)
chmod +x local-eval/setup-gpu-native.sh local-eval/run-pipeline-gpu-native.sh
./local-eval/setup-gpu-native.sh

cp local-eval/.env.template local-eval/.env
# Edit .env: OPENROUTER_API_KEY=...  HF_TOKEN=...  (OpenRouter used for critic only)
source local-eval/.env

./local-eval/run-pipeline-gpu-native.sh
```

What this runs:

| Component | How |
|-----------|-----|
| Coder | **Local vLLM** — `Tooony133/Qwen-3.6-27B-AstroWolf` on `:8001` |
| Critic/judge | **OpenRouter** (saves GPU VRAM) |
| API | `serve.py` on **`:10006`** |
| Config | `local-eval/configuration.gpu-native.yaml` |

**Adjust GPUs** in `local-eval/configuration.gpu-native.yaml` before starting:

```yaml
llm_clients:
  coder-instance:
    vllm:
      gpu_ids: "0"              # 1 GPU
      tensor_parallel_size: 1
      # 2 GPUs: gpu_ids "0,1", tensor_parallel_size 2
```

Wait until ready:

```bash
curl -s http://127.0.0.1:10006/health
curl -s http://127.0.0.1:10006/status | python -m json.tool
```

First boot downloads AstroWolf into `$HF_HOME` — can take 1–3 hours.

Logs:

```bash
tail -f $REPO/local-eval/runs/pipeline-server-gpu-native.log
tail -f $REPO/shiny-guide/pipeline_service/logs/pipeline.log
```

---

#### Option B — Docker (full production stack)

Only if Docker works on your host (not inside a nested container without DinD):

```bash
cd $REPO/shiny-guide/docker
docker compose build
docker compose up -d
docker compose logs -f miner-pipeline
```

Uses `shiny-guide/configuration.yaml` (4-GPU layout: AstroWolf + GLM vLLM).

---

#### Option C — OpenRouter only (smoke test — NOT for training data)

```bash
./local-eval/run-pipeline-cpu.sh   # Gemini, not AstroWolf — wrong model for DPO mining
```

---

**Tip:** Dataset generation can run on **machine A** and training on **machine B**:

```bash
PIPELINE_URL=http://<machine-a-ip>:10006 ./run/01_prepare_shiny_align.sh
```

---

### Phase 4 — Prepare training dataset

Always:

```bash
cd $TRAINING
source .venv/bin/activate
export PROMPTS_ROOT=shiny-guide
export PYTHONPATH=$TRAINING/scripts
export HF_HOME=$TRAINING/data/hf_cache
```

#### Choose DPO or GRPO

| Method | Prep command | Output dataset |
|--------|--------------|----------------|
| **DPO** | `ALIGN=dpo` | `data/hf/dpo_shiny/dataset` |
| **GRPO** | `ALIGN=grpo` | `data/hf/grpo_shiny/dataset` |
| Both | `ALIGN=both` | both dirs (train one method) |

#### One-command prep (recommended)

**Smaller first run (smoke / iterate):**

```bash
ALIGN=dpo TRAIN_N=500 VAL_N=50 DUEL_N=50 DPO_SAMPLES=4 \
PIPELINE_URL=http://127.0.0.1:10006 \
./run/01_prepare_shiny_align.sh
```

**Production-sized run:**

```bash
ALIGN=dpo TRAIN_N=5000 VAL_N=300 DUEL_N=200 DPO_SAMPLES=4 \
PIPELINE_URL=http://127.0.0.1:10006 \
REWARD_MODE=cheap MIN_MARGIN=0.15 \
./run/01_prepare_shiny_align.sh
```

For **GRPO** (prompt-only — no candidate collection):

```bash
ALIGN=grpo TRAIN_N=5000 VAL_N=300 DUEL_N=200 \
./run/01_prepare_shiny_align.sh
```

What the script does:

1. Split `prompts.txt` → `data/splits/{train,val,duel}.txt`
2. Download PNGs → `data/images/{stem}.png`
3. **DPO only:** call shiny-guide `K=4` times per stem → `data/candidates/shiny_k4/`
4. **DPO only:** score pairs with `validate.js` → `data/hf/dpo_shiny/`
5. **GRPO only:** pack prompts → `data/hf/grpo_shiny/`

#### Verify dataset quality

```bash
# DPO
cat data/hf/dpo_shiny/meta.json
# expect n_pairs > 0 (aim ≥500 iterate, ≥2000 serious)

head -1 data/hf/dpo_shiny/dpo.jsonl | python -m json.tool | head -20

# GRPO
cat data/hf/grpo_shiny/meta.json
python -c "from datasets import load_from_disk; ds=load_from_disk('data/hf/grpo_shiny/dataset'); print(len(ds), ds.column_names)"
```

If `n_pairs: 0` for DPO:

- shiny-guide not reachable at `PIPELINE_URL`
- candidates all fail validate (raise `MIN_MARGIN` or check pipeline output)
- increase `DPO_SAMPLES` to 6–8

#### Exclude live competition round from train (recommended)

```bash
curl -sL "https://raw.githubusercontent.com/404-Repo/404-active-competition/main/rounds/N/prompts.txt" \
  > /tmp/roundN.txt

export PYTHONPATH=$TRAINING/scripts
python scripts/prepare_splits.py \
  --pool $REPO/prompts.txt \
  --train 5000 --val 300 --duel 200 --seed 7 \
  --exclude /tmp/roundN.txt \
  --out-dir data/splits
# then re-run download + candidate collection steps from 01_prepare_shiny_align.sh
# or run the full script after deleting data/splits and re-splitting manually
```

---

### Phase 5 — Run training (skip SFT)

AstroWolf is already specialized. **Do not run** `./run/02_sft.sh`.

#### DPO training

Dry-run (64 samples) — uncomment in yaml first:

```bash
# edit configs/dpo_shiny_27b.yaml → max_samples: 64
CONFIG=configs/dpo_shiny_27b.yaml ./run/03_dpo.sh
```

Full run:

```bash
CONFIG=configs/dpo_shiny_27b.yaml NUM_PROCESSES=1 ./run/03_dpo.sh
```

Checkpoint: `data/checkpoints/dpo_shiny_27b/final`

#### GRPO training

Phase 1 — cheap reward (validate + format):

```bash
# configs/grpo_shiny_27b.yaml → reward_mode: cheap
CONFIG=configs/grpo_shiny_27b.yaml ./run/03_grpo.sh
```

Phase 2 — S1 proxy (optional, needs GLM judge):

```bash
export JUDGE_BASE_URL=http://127.0.0.1:8002/v1
export JUDGE_MODEL=zai-org/GLM-4.6V-Flash
# edit yaml: reward_mode: s1
CONFIG=configs/grpo_shiny_27b.yaml ./run/03_grpo.sh
```

Checkpoint: `data/checkpoints/grpo_shiny_27b/final`

#### Monitor training

```bash
tail -f data/checkpoints/dpo_shiny_27b/trainer_log.jsonl 2>/dev/null || true
watch -n5 nvidia-smi
```

---

### Phase 6 — Merge LoRA and deploy

```bash
cd $TRAINING
source .venv/bin/activate
export PYTHONPATH=$TRAINING/scripts

# After DPO:
BASE_MODEL=Tooony133/Qwen-3.6-27B-AstroWolf \
ADAPTER=data/checkpoints/dpo_shiny_27b/final \
MERGED=data/checkpoints/merged_shiny_coder \
./run/04_merge_and_eval.sh

# After GRPO:
BASE_MODEL=Tooony133/Qwen-3.6-27B-AstroWolf \
ADAPTER=data/checkpoints/grpo_shiny_27b/final \
MERGED=data/checkpoints/merged_shiny_coder \
./run/04_merge_and_eval.sh
```

Wire into shiny-guide:

```yaml
# shiny-guide/configuration.yaml
llm_clients:
  coder-instance:
    vllm:
      model: "/absolute/path/to/my-agent/training/data/checkpoints/merged_shiny_coder"
actors:
  coder:
    model: "/absolute/path/to/my-agent/training/data/checkpoints/merged_shiny_coder"
  planner:
    model: "/absolute/path/to/my-agent/training/data/checkpoints/merged_shiny_coder"
```

Rebuild and serve:

```bash
cd $REPO/shiny-guide/docker
docker compose build && docker compose up -d
```

---

### Phase 7 — Evaluate (held-out stems)

Never train on `data/splits/duel.txt` — use it for eval only:

```bash
cd $REPO
source local-eval/.env 2>/dev/null || true
./local-eval/run-eval.sh $TRAINING/data/splits/duel.txt --limit 100 --name post-train-eval
```

---

### Copy-paste checklist (fresh machine)

```bash
# 0–2: env
export REPO=/home/404-gen-subnet
export TRAINING=$REPO/my-agent/training
cd $TRAINING && source .venv/bin/activate
export PROMPTS_ROOT=shiny-guide PYTHONPATH=$TRAINING/scripts HF_HOME=$TRAINING/data/hf_cache

# 3: shiny-guide native GPU (no Docker) — separate terminal, keep running
cd $REPO && source local-eval/.env
./local-eval/setup-gpu-native.sh          # once
./local-eval/run-pipeline-gpu-native.sh   # until curl :10006/health OK

# 4: data (DPO example)
cd $TRAINING
ALIGN=dpo TRAIN_N=5000 PIPELINE_URL=http://127.0.0.1:10006 ./run/01_prepare_shiny_align.sh
cat data/hf/dpo_shiny/meta.json

# 5: train
CONFIG=configs/dpo_shiny_27b.yaml ./run/03_dpo.sh

# 6: merge
ADAPTER=data/checkpoints/dpo_shiny_27b/final MERGED=data/checkpoints/merged_shiny_coder \
  ./run/04_merge_and_eval.sh
```

---

## DPO vs GRPO (which to pick)

| | **DPO** | **GRPO** |
|---|---------|----------|
| Data | Offline chosen/rejected JS pairs | Prompt-only; samples during training |
| GPU cost | Lower | Higher (rollouts + reward) |
| Config | `configs/dpo_shiny_27b.yaml` | `configs/grpo_shiny_27b.yaml` |
| Best when | shiny-guide running for K samples/prompt | Reward farm + vLLM rollouts available |

Both load **base AstroWolf** directly — no `sft_adapter_path`.

Optional stack: **DPO first** → then GRPO with `sft_adapter_path: data/checkpoints/dpo_shiny_27b/final` in `grpo_shiny_27b.yaml`.

---

## Manual dataset steps (if not using the wrapper script)

### Splits + images

```bash
export PYTHONPATH=$TRAINING/scripts PROMPTS_ROOT=shiny-guide

python scripts/prepare_splits.py \
  --pool $REPO/prompts.txt \
  --train 5000 --val 300 --duel 200 --seed 7 \
  --out-dir data/splits

cat data/splits/train.txt data/splits/val.txt > data/splits/train_val.txt
python scripts/download_images.py \
  --list data/splits/train_val.txt \
  --out data/images --workers 16
```

### DPO: candidates + pack

```bash
python scripts/collect_candidates.py --from-pipeline \
  --list data/splits/train.txt \
  --base-url http://127.0.0.1:10006 \
  --samples 4 --out data/candidates/shiny_k4

python scripts/pack_dpo_dataset.py --source candidates \
  --candidates-dir data/candidates/shiny_k4 \
  --list data/splits/train.txt \
  --images data/images \
  --reward-mode cheap --min-margin 0.15 \
  --out data/hf/dpo_shiny
```

### GRPO: prompt pack only

```bash
python scripts/pack_grpo_prompts.py \
  --list data/splits/train.txt \
  --images data/images \
  --out data/hf/grpo_shiny
```

---

## Reward modes (DPO pair mining)

| `REWARD_MODE` | Scores | When |
|---------------|--------|------|
| `cheap` | validate.js + format | Fast iteration (default) |
| `s1` | GLM front-match proxy | Closer to validator S1 |
| `render` | HTTP render service | Needs render sidecar |

```bash
export JUDGE_BASE_URL=http://127.0.0.1:8002/v1
export JUDGE_MODEL=zai-org/GLM-4.6V-Flash
REWARD_MODE=s1 MIN_MARGIN=0.20 ALIGN=dpo ./run/01_prepare_shiny_align.sh
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Cannot run Docker (nested container) | Use **native GPU**: `./local-eval/setup-gpu-native.sh` then `./local-eval/run-pipeline-gpu-native.sh` |
| Chromium / puppeteer launch fails | Install renderer libs (`libgbm1`, `libnss3`, …) — see Phase 0 apt list |
| vLLM not found | Run `setup-gpu-native.sh`; check `local-eval/.vllm-env/bin/vllm` |
| `cuda: False` | Reinstall torch with CUDA index URL matching driver |
| `ModuleNotFoundError: torch` | `source .venv/bin/activate`; install CUDA torch before requirements |
| `validate.js` fails | `cd $REPO/miner-reference/validator && npm install` |
| shiny-guide `:10006` connection refused | `docker compose up -d`; wait for model download |
| DPO `n_pairs: 0` | Check pipeline logs; lower `MIN_MARGIN`; increase `DPO_SAMPLES` |
| GRPO OOM | Lower `num_generations` to 2; enable `load_in_4bit`; use more GPUs |
| DPO OOM on 27B | Keep `load_in_4bit: true`; batch=1; increase `gradient_accumulation_steps` |
| HF 401 / gated model | `huggingface-cli login`; accept model license on HF website |
| Train/serve prompt drift | Always `PROMPTS_ROOT=shiny-guide` for pack + train |

---

## Important rules

1. **Train on `train.txt`, eval on `duel.txt`** — never leak duel stems into DPO/GRPO train sets.
2. **Exclude recent live rounds** from train splits.
3. **GPU teacher for GPU student** — do not build DPO pairs from CPU Gemini OpenRouter path.
4. **Skip SFT** unless starting from a non-specialized base model.
5. **Audit** — Docker-regenerated JS must match CDN upload (0% margin rule on subnet).

---

## File map

| Step | Script / config |
|------|-----------------|
| Setup | `run/00_setup.sh` |
| Data prep | `run/01_prepare_shiny_align.sh` |
| DPO train | `run/03_dpo.sh` + `configs/dpo_shiny_27b.yaml` |
| GRPO train | `run/03_grpo.sh` + `configs/grpo_shiny_27b.yaml` |
| Merge | `run/04_merge_and_eval.sh` + `scripts/merge_lora.py` |
| Strategy | [`docs/TRAINING_DPO_STRATEGY.md`](../../docs/TRAINING_DPO_STRATEGY.md) |
