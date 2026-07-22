# Coder base model — download and save path

The **coder base model** for DPO/GRPO training and pipeline data generation is the subnet king weights:

| Field | Value |
|-------|--------|
| **HuggingFace repo** | [`Tooony133/Qwen-3.6-27B-AstroWolf`](https://huggingface.co/Tooony133/Qwen-3.6-27B-AstroWolf) |
| **Role** | Image → Three.js `generate(THREE)` (27B VLM, already specialized) |
| **Training** | **Skip SFT** — start from DPO or GRPO directly on this checkpoint |
| **Disk (bf16)** | ~54 GB weights + tokenizer/config |

---

## Default save location (standalone)

After `./run/00_bootstrap_assets.sh`, weights live here:

```text
training/data/models/Qwen-3.6-27B-AstroWolf/
├── config.json
├── tokenizer.json
├── model.safetensors.index.json   # or single .safetensors
└── ...
```

Environment variables (written to `.env` by bootstrap):

```bash
MODEL_PATH=/path/to/training/data/models/Qwen-3.6-27B-AstroWolf
CODER_MODEL_PATH=$MODEL_PATH          # alias
CODER_MODEL_ID=Tooony133/Qwen-3.6-27B-AstroWolf
HF_HOME=/path/to/training/data/hf_cache   # HF download cache (optional)
```

Override save directory before bootstrap:

```bash
# in .env
MODEL_DIR=/mnt/nvme/models/AstroWolf
```

---

## Automatic download (recommended)

```bash
cd training
cp .env.template .env
# Edit .env — required:
#   HF_TOKEN=hf_...

export HF_TOKEN=hf_...    # or: huggingface-cli login

./run/00_bootstrap_assets.sh
```

Bootstrap runs:

```bash
huggingface-cli download Tooony133/Qwen-3.6-27B-AstroWolf \
  --local-dir data/models/Qwen-3.6-27B-AstroWolf \
  --token "$HF_TOKEN"
```

First download often takes **1–3 hours** depending on bandwidth.

---

## Manual download

If you already have the weights elsewhere, **copy or symlink** into the default path, or set `MODEL_PATH` in `.env`:

```bash
cd training
mkdir -p data/models

# Option A — download yourself
pip install -U "huggingface_hub[cli]"
export HF_TOKEN=hf_...
huggingface-cli download Tooony133/Qwen-3.6-27B-AstroWolf \
  --local-dir data/models/Qwen-3.6-27B-AstroWolf \
  --token "$HF_TOKEN"

# Option B — use an existing local copy
export MODEL_PATH=/mnt/models/Qwen-3.6-27B-AstroWolf
echo "MODEL_PATH=$MODEL_PATH" >> .env
```

---

## Verify the download

```bash
ls training/data/models/Qwen-3.6-27B-AstroWolf/config.json
ls training/data/models/Qwen-3.6-27B-AstroWolf/*.safetensors* 2>/dev/null | head
```

Bootstrap skips re-download if `config.json` or `model.safetensors.index.json` already exists.

---

## Who uses `MODEL_PATH`?

| Component | How |
|-----------|-----|
| **Pipeline vLLM** (`pipeline/run-native.sh`) | Patches runtime yaml → local path for `:8001` coder |
| **DPO training** (`configs/dpo_shiny_27b.yaml`) | `model_name_or_path` — env `MODEL_PATH` overrides HF id in `train_dpo.py` |
| **GRPO training** | Same via `resolve_model_path()` in training scripts |
| **Merge LoRA** (`run/04_merge_and_eval.sh`) | `BASE_MODEL=$MODEL_PATH` or HF id |

Training yaml default (if `MODEL_PATH` unset):

```yaml
model_name_or_path: "Tooony133/Qwen-3.6-27B-AstroWolf"
```

With `MODEL_PATH` set, training loads **local files** (faster, no re-download).

---

## HuggingFace access

1. Create a token: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access).
2. Accept the model license on the model page if gated.
3. Put `HF_TOKEN=hf_...` in `training/.env`.

```bash
huggingface-cli login --token "$HF_TOKEN"
```

---

## Common issues

| Issue | Fix |
|-------|-----|
| `401 Unauthorized` | Set `HF_TOKEN`; accept model license on HF |
| `No space left on device` | Need ~60 GB free under `data/models/` |
| vLLM loads wrong model | Check `MODEL_PATH` in `.env`; restart pipeline after change |
| Re-download every time | Ensure `MODEL_DIR` is on persistent disk, not ephemeral container root |

---

## Do not use for DPO data generation

These are **different models** — do not use them to mine DPO pairs for AstroWolf:

- OpenRouter Gemini / GPT (CPU pipeline)
- Random Qwen-VL base without AstroWolf fine-tune
- Production `handsometiger0202/...` unless you intentionally switch `CODER_MODEL_ID`

Data generation must use **the same weights you train** (`Tooony133/Qwen-3.6-27B-AstroWolf` by default).

See also: [`STANDALONE.md`](../STANDALONE.md) · [`MACHINE_PROFILES.md`](../MACHINE_PROFILES.md)
