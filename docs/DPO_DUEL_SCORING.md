# DPO with production multiview duel scoring

Build DPO pairs aligned with **subnet validator logic** (S1–S4, DINO, AB/BA judge) — not cheap `validate.js` margin alone.

## Why this path

| Method | Scoring | Aligns with subnet duels? |
|--------|---------|---------------------------|
| `--reward-mode cheap` | validate.js + heuristics | Partial (reliability only) |
| **`duel-scored`** | render → DINO → OpenRouter S1–S4 | **Yes** (same stack as king workflow) |

Production duels use multiview renders + VLM judge. This path scores **two JS candidates per prompt** that way, then packs winner/loser as DPO chosen/rejected.

## Requirements

| Resource | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | **Required** for S1–S4 judge (Gemini/GLM via OpenRouter) |
| `HF_TOKEN` | DINO embedder download |
| GPU pipeline | Step 2 only — generate 2 AstroWolf JS per stem |
| CPU/GPU + Chromium | Step 3 — multiview renderer sidecars |
| Node ≥ 20 | validate.js + renderer |

**2× H200 workflow:** run steps **sequentially** — stop pipeline before duel scoring.

## Full command sequence

```bash
cd training
source .env   # OPENROUTER_API_KEY + HF_TOKEN

# Optional: profile for dataset size
./run/00_configure_profile.sh h200x2-dpo

# --- Phase A: generate 2 JS per prompt ---
./pipeline/start-native-bg.sh && ./pipeline/wait-ready.sh

DPO_SAMPLES=2 TRAIN_N=3000 DUEL_LIMIT=500 \
  ./run/01_prepare_dpo_duel_scored.sh
# (script collects candidates, then scores — stop pipeline before scoring if OOM)

./pipeline/stop-native.sh

# --- Phase B: train (no OpenRouter) ---
CONFIG=configs/dpo_shiny_27b_duel.yaml ./run/03_dpo.sh
```

Or run steps manually:

```bash
# 1. Collect 2 candidates
DPO_SAMPLES=2 python scripts/collect_candidates.py --from-pipeline \
  --list data/splits/train.txt --base-url http://127.0.0.1:10006 \
  --samples 2 --out data/candidates/shiny_k2

# 2. Multiview duel score
export CONFIG_FILE=pipeline/configuration.duel-judge.yaml
export PYTHONPATH=$SHINY_GUIDE_ROOT/pipeline_service:$PYTHONPATH
python scripts/duel_score_candidates.py \
  --candidates-dir data/candidates/shiny_k2 \
  --images data/images \
  --list data/splits/train.txt \
  --out data/duel_scores/candidate_duels.json \
  --limit 500

# 3. Pack DPO
python scripts/pack_dpo_dataset.py --source duel-scored \
  --duel-json data/duel_scores/candidate_duels.json \
  --list data/splits/train.txt \
  --images data/images \
  --out data/hf/dpo_shiny_duel
```

## Judge config

Edit `pipeline/configuration.duel-judge.yaml`:

```yaml
actors:
  judge:
    model: "google/gemini-2.5-pro-preview"   # or zai-org/GLM-4.6V-Flash on OpenRouter
```

Set `JUDGE_CONFIG` in `.env` if using a custom path.

## Dataset size guidance

| Goal | Stems scored | OpenRouter cost | Notes |
|------|--------------|-----------------|-------|
| Smoke | 50–100 | low | verify pipeline |
| Iterate | 500–1000 | moderate | ~500 DPO pairs after draws |
| Serious | 3000–6000 | **high** | budget OpenRouter; 2 renders + up to 4 judge stages per pair |

Each stem runs **2 multiview renders + AB/BA judge** — much slower and costlier than `reward_mode=cheap`.

Draws are skipped by default (`--include-draws` to keep).

## Output

| Path | Content |
|------|---------|
| `data/duel_scores/candidate_duels.json` | Per-stem winner, S1–S4 detail, chosen/rejected paths |
| `data/hf/dpo_shiny_duel/dataset` | HF dataset for TRL DPOTrainer |
| `configs/dpo_shiny_27b_duel.yaml` | Training config pointing at duel dataset |

## Compare with cheap DPO

Use **`01_prepare_shiny_align.sh`** + `reward_mode=cheap` when:

- Bootstrapping quickly
- OpenRouter budget is limited
- You only need validate.js reliability signal

Use **`01_prepare_dpo_duel_scored.sh`** when:

- Optimizing for **subnet duel win-rate** (S1 front match, S4 depth)
- You can afford OpenRouter judge passes
- You want DPO pairs ranked like the **top workflow**

See also: [`docs/DPO_DUEL_SCORING.md`](docs/DPO_DUEL_SCORING.md) in this folder.
