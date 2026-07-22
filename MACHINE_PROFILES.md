# Machine capacity profiles

Pick a profile **before** bootstrap on a new GPU box. Each profile sets `.env`, dataset sizes, training config, and pipeline GPU layout.

```bash
cd training
cp .env.template .env
./run/00_configure_profile.sh              # list profiles
./run/00_configure_profile.sh h200x2-dpo   # apply for your hardware
# edit .env: OPENROUTER_API_KEY + HF_TOKEN
./run/run_all.sh
```

---

## Profile matrix

| Profile | GPUs | Method | Dataset (train) | Pipeline GPUs | Training GPUs |
|---------|------|--------|-----------------|---------------|---------------|
| **smoke** | 1× any | DPO LoRA | 100 prompts | 1 (`tp=1`) | same box, sequential |
| **h100x2-dpo** | 2× H100 80GB | DPO bf16 LoRA | 5k → ~2k pairs | 2 (`tp=2`) | 2 (stop pipeline first) |
| **h200x2-dpo-duel** ⭐ | 2× H200 | DPO duel-scored | 3k × 2 JS | 2 (`tp=2`) | 2 (sequential phases) |
| **h200x4-dpo** | 4× H200 | DPO bf16 LoRA prep | 5k → ~2k pairs | 4 (`tp=4`) | 2 (stop pipeline first) |
| **h100x4-grpo** | 4× H100 80GB | GRPO bf16 LoRA | 5k prompts | 2 (`tp=2`) | 4 |
| **h200x2-grpo** | 2× H200 | GRPO bf16 LoRA | 4k prompts | 2 (`tp=2`) | 2 (tight; `num_generations: 2`) |
| **h200x8-fullft** | 8× H200 | Full SFT | 10k validated JS | 4 (`tp=4`) | 8 + ZeRO-3 |
| **train-only** | 2× H200/H100 | DPO LoRA | pre-built | skip | 2 |

⭐ Default recommendation for beating the king on a single renter box: **`h200x2-dpo`**.

---

## What each profile changes

| Setting | Where | Purpose |
|---------|-------|---------|
| `TRAIN_N`, `VAL_N`, `DUEL_N` | `.env` | Dataset size vs VRAM/time |
| `DPO_SAMPLES` | `.env` | Candidates per prompt (DPO mining) |
| `TRAIN`, `ALIGN`, `CONFIG` | `.env` | DPO vs GRPO vs SFT full FT |
| `gpu_ids`, `tensor_parallel_size` | `pipeline/configuration.local.yaml` | vLLM AstroWolf layout |
| `CONFIG_FILE` | `.env` | Points pipeline at local yaml |
| `load_in_4bit`, `use_lora` | `configs/*.yaml` | bf16 LoRA default; full FT manual |

---

## Single-box workflow (2× H200 DPO)

One machine cannot run pipeline vLLM + training at full load simultaneously.

```
Phase A — data generation (needs OPENROUTER_API_KEY for critic/judge)
  ./pipeline/start-native-bg.sh
  ./pipeline/wait-ready.sh
  source .env && ALIGN=dpo ./run/01_prepare_shiny_align.sh
  ./pipeline/stop-native.sh

Phase B — training (OPENROUTER not needed)
  CONFIG=configs/dpo_shiny_27b.yaml ./run/03_dpo.sh
```

Or use `./run/run_all.sh` (runs both; stop pipeline manually if OOM).

---

## Dataset size guide

| Goal | SFT (full FT) | DPO pairs | GRPO prompts |
|------|---------------|-----------|--------------|
| Smoke / debug | 100–500 | 50–200 | 100 |
| Iterate | — | 500–2000 | 1000–3000 |
| **Serious (LoRA)** | — | **2000–5000** | **4000–6000** |
| **Serious (full FT)** | **8000–12000** | — | — |
| Diminishing returns | 30k+ | 10k+ | 10k+ |

All examples must pass **`validate.js`**. Teacher for data gen = **AstroWolf via shiny-guide**, not OpenRouter Gemini.

---

## Training method vs GPU

### bf16 LoRA (default configs)

| Method | Min GPUs | Recommended | Config |
|--------|----------|-------------|--------|
| DPO | 2× 80 GB | **2× H200** | `dpo_shiny_27b.yaml` |
| GRPO | 2× 80 GB (tight) | **4× H100** | `grpo_shiny_27b.yaml` |

```yaml
use_lora: true
load_in_4bit: false   # bf16 base + LoRA adapters
```

### Full fine-tune (all 27B weights)

| Min | Recommended | Dataset |
|-----|-------------|---------|
| 8× H100 80GB + ZeRO-3 | **8× H200** | 8k–12k SFT JS |

```yaml
# configs/sft_shiny_27b.yaml
use_lora: false
load_in_4bit: false
learning_rate: 1.0e-6   # lower than LoRA
```

**2× H200 cannot full fine-tune 27B** — use LoRA DPO instead.

---

## API keys by phase

| Phase | `HF_TOKEN` | `OPENROUTER_API_KEY` |
|-------|------------|----------------------|
| Bootstrap (model download) | ✅ ([`docs/CODER_MODEL.md`](docs/CODER_MODEL.md)) | ❌ |
| Pipeline data gen | ✅ | ✅ (critic/judge only) |
| **DPO duel-scored prep** | ✅ | ✅ (**multiview S1–S4 judge**) |
| DPO/GRPO training (`cheap` reward) | ✅ | ❌ |
| GRPO `reward_mode: s1` | ✅ | optional (judge) |
| Train-only profile | ✅ | ❌ |

---

## Manual overrides after applying a profile

### Pipeline — 1 GPU only

Edit `pipeline/configuration.local.yaml`:

```yaml
gpu_ids: "0"
tensor_parallel_size: 1
```

### GRPO OOM on 2× H200

Edit `configs/grpo_shiny_27b.yaml`:

```yaml
num_generations: 2
use_vllm: false
```

### DPO OOM

```yaml
max_completion_length: 12288
gradient_accumulation_steps: 32
```

### Split data gen and training across two machines

**Machine A** (pipeline):

```bash
./pipeline/run-native.sh
```

**Machine B** (train):

```bash
PIPELINE_URL=http://<machine-a>:10006 ./run/01_prepare_shiny_align.sh
SKIP_PIPELINE=1 ./run/run_all.sh
```

---

## Config file reference

| Profile | Training config | Notes |
|---------|-----------------|-------|
| h200x2-dpo, h100x2-dpo | `configs/dpo_shiny_27b.yaml` | Skip SFT |
| h100x4-grpo, h200x2-grpo | `configs/grpo_shiny_27b.yaml` | `reward_mode: cheap` |
| h200x8-fullft | `configs/sft_shiny_27b.yaml` | Set `use_lora: false` |
| smoke | `configs/dpo_shiny_27b.yaml` | `SMOKE=1` in .env |

---

## Quick decision tree

```
How many GPUs?
├─ 1 GPU        → smoke profile; DPO iterate only
├─ 2× H200/H100 → h200x2-dpo (best ROI)
│                 h200x2-grpo only if you accept tighter settings
├─ 4× H100      → h100x4-grpo
└─ 8× H200      → h200x8-fullft (full SFT) OR run larger LoRA DPO with TRAIN_N=10000

Already have dataset?
└─ train-only profile → skip pipeline + OPENROUTER
```

See also: [`STANDALONE.md`](STANDALONE.md), [`SHINY_GUIDE_TRAINING.md`](SHINY_GUIDE_TRAINING.md), [`docs/DPO_DUEL_SCORING.md`](docs/DPO_DUEL_SCORING.md).
