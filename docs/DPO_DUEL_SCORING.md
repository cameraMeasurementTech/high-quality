# DPO with production multiview duel scoring (4× H200)

Build DPO pairs where **chosen/rejected** come from scoring **two JS candidates** with the **same stack as subnet validators** (validate → multiview render → DINO → S1–S4 AB/BA judge).

---

## How to generate 2 JS codes from the same coder (your question)

Use **the same prompt and temperature; change only the seed**.

| Same across both samples | Different |
|--------------------------|-----------|
| Reference image URL | RNG **seed** |
| Coder system + user prompts | → `sample_0.js` vs `sample_1.js` |
| Model (AstroWolf) | |
| Temperature (pipeline `actors.coder.temperature`, typically **0.6**) | |

`collect_candidates.py` does:

```text
sample_0 → /generate with seed = 42 + batch_i
sample_1 → /generate with seed = 42 + 1000 + batch_i
```

This matches **production multigen**: `ensemble_size > 1` uses `seed + k` at a fixed `ensemble_temperature`.

### Why not different temperatures?

| Approach | Pros | Cons |
|----------|------|------|
| **Different seeds (recommended)** | Fair pairwise compare; matches subnet multigen | Samples can be similar (ok — judge still ranks) |
| Different temperatures (e.g. 0.5 vs 0.9) | More stylistic diversity | High temp → more invalid JS; preference becomes “valid vs broken”, not “better silhouette” |

For **duel-scored DPO** aimed at subnet win-rate, keep temperature fixed and vary seeds only.

---

## Why this path

| Method | Scoring | Aligns with subnet duels? |
|--------|---------|---------------------------|
| Cheap (`validate.js`) | Format / pass-fail | Partial |
| **`duel-scored`** | Multiview + DINO + S1–S4 | **Yes** |

---

## 4× H200 — max GPU use

| Phase | GPUs | What |
|-------|------|------|
| **A — Generate 2 JS/stem** | **0–3 TP=4** vLLM AstroWolf | `max_num_seqs=96`, `BATCH_SIZE=48` |
| **B — Duel score** | Stop vLLM; DINO on `cuda:0`; Chromium on CPU | OpenRouter judge |
| **C — DPO train** | **4 processes** LoRA | `NUM_PROCESSES=4` |

Do **not** run Phase A and C at the same time on one box.

Profile:

```bash
./run/00_configure_profile.sh h200x4-dpo-duel
```

Sets: `DPO_SAMPLES=2`, `BATCH_SIZE=48`, `SIDECAR_COUNT=8`, `DUEL_CONCURRENCY=4`, `CONFIG=configs/dpo_shiny_27b_duel.yaml`, pipeline TP=4.

---

## Full command sequence (4× H200)

```bash
cd training
cp .env.template .env
# HF_TOKEN=...   OPENROUTER_API_KEY=...   (both required for this path)

./run/00_configure_profile.sh h200x4-dpo-duel
./run/00_bootstrap_assets.sh
INSTALL_SYSTEM=1 ./run/00_install_all.sh
source .env && source .venv/bin/activate

# --- Phase A: generate 2 JS per prompt (all 4 GPUs) ---
./pipeline/start-native-bg.sh && ./pipeline/wait-ready.sh

SKIP_DUEL_SCORE=1 SKIP_PACK=1 \
  TRAIN_N=5000 DPO_SAMPLES=2 BATCH_SIZE=48 \
  ./run/01_prepare_dpo_duel_scored.sh

./pipeline/stop-native.sh   # free GPUs before scoring / training

# --- Phase B: validator-like duel score ---
SKIP_COLLECT=1 \
  SIDECAR_COUNT=8 DUEL_CONCURRENCY=4 \
  ./run/01_prepare_dpo_duel_scored.sh

# Optional smoke first: DUEL_LIMIT=100

# --- Phase C: train ---
CONFIG=configs/dpo_shiny_27b_duel.yaml NUM_PROCESSES=4 ./run/03_dpo.sh
```

Check pairs:

```bash
cat data/duel_scores/candidate_duels.json | python -c \
  "import json,sys; d=json.load(sys.stdin); print(d['n_pairs_for_dpo'], d['skipped'])"
cat data/hf/dpo_shiny_duel/meta.json
```

---

## Scoring stack (validator-aligned)

Per stem (`sample_0` vs `sample_1`):

1. `validate.js` gate (both invalid → skip)
2. Multiview Chromium render (white + gray views)
3. DINOv3 embeddings vs reference
4. OpenRouter judge **AB + BA** (S1–S4), same style as production
5. Winner → `chosen`, loser → `rejected` (draws skipped)

Config: `pipeline/configuration.duel-judge.yaml`

---

## Dataset size (4× H200)

| Goal | `TRAIN_N` | Expected pairs (after draws) | Notes |
|------|-----------|------------------------------|-------|
| Smoke | 100 | ~60–80 | `DUEL_LIMIT=100` |
| Iterate | 1000 | ~700–900 | moderate OpenRouter |
| Serious | 5000 | ~3500–4500 | profile default |

---

## Knobs for speed

| Knob | Default (profile) | Meaning |
|------|-------------------|---------|
| `BATCH_SIZE` | 48 | Stems per `/generate` (raise toward 64 if GPU util low) |
| `max_num_seqs` | 96 | vLLM concurrent coder slots |
| `coder.workers` | 96 | Pipeline semaphore |
| `SIDECAR_COUNT` | 8 | Chromium processes for scoring |
| `DUEL_CONCURRENCY` | 4 | Parallel stems while judging (watch OpenRouter rate limits) |
| `NUM_PROCESSES` | 4 | DPO training GPUs |

If GPU util is low during Phase A → raise `BATCH_SIZE`.  
If vLLM OOM → lower `max_num_seqs` to 64 in `configuration.h200x4-dpo.yaml`.

---

## Output

| Path | Content |
|------|---------|
| `data/candidates/shiny_k2/{stem}/sample_{0,1}.js` | Two JS + `.meta.json` (seed) |
| `data/duel_scores/candidate_duels.json` | Winner + S1–S4 detail |
| `data/hf/dpo_shiny_duel/dataset` | HF DPO dataset |
| `data/checkpoints/dpo_shiny_27b_duel/` | Trained LoRA |

---

## Cheap DPO vs duel-scored

| | Cheap | Duel-scored |
|-|-------|-------------|
| Profile | `h200x4-dpo` | **`h200x4-dpo-duel`** |
| Samples | 4 | **2** |
| Ranking | validate.js best vs worst | **S1–S4 duel winner/loser** |
| OpenRouter | No | **Yes** |
| Goal | Reliability | **Subnet duel win-rate** |

See also: [`CODER_MODEL.md`](CODER_MODEL.md) · [`STANDALONE.md`](../STANDALONE.md) · [`MACHINE_PROFILES.md`](../MACHINE_PROFILES.md)
