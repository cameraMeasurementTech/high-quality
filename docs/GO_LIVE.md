# Running the my-agent Miner in Live Mode (404-GEN Subnet 17)

This is the operational runbook for taking the `my-agent` pipeline **live** on the
404-GEN Subnet 17 "Procedural Image-to-3D" competition (Level 3 / go-live).

For the pipeline internals see [`PIPELINE_WORKFLOW.md`](PIPELINE_WORKFLOW.md); for
the full subnet rules see [`../../docs/MINER_VALIDATOR_GUIDE.md`](../../docs/MINER_VALIDATOR_GUIDE.md).
This file only covers **how you actually run and compete each round**.

---

## 0. Mental model: there is no long-running "miner neuron"

`my-agent` does **not** run a Bittensor axon/neuron loop that waits for validator
calls. Going live is an **operational loop you drive**:

```
  on-chain reveal (repo + commit + cdn_url)
        │
        ▼
  validators publish round inputs (seed.json + prompts.txt, 128 image URLs)
        │
        ▼
  you generate 128 {stem}.js with the my-agent Docker (batch API on :10006)
        │
        ▼
  you validate + upload each {stem}.js to YOUR CDN before the deadline
        │
        ▼
  if you win the timeline, validators redeploy YOUR Docker image and
  audit-regenerate with the same seed  →  must reproduce your CDN files (0% margin)
```

The container itself is just: **vLLM servers + FastAPI batch API** (`serve.py` on
port `10006`). You (or a small script) act as the "client" that submits prompts and
collects results — exactly like the validators do during audit.

---

## 1. Prerequisites (one-time)

### 1.1 Hardware — 4×H200 (or equivalent)

The production `configuration.yaml` runs **two vLLM servers**, each
`tensor_parallel_size: 2`, on disjoint GPUs:

| vLLM server | Model repo | GPUs | Port |
|-------------|-----------|------|------|
| `coder-instance` | `Tooony133/Qwen-3.6-27B-AstroWolf` | `0,1` | 8001 |
| `judge-critic-instance` | `zai-org/GLM-4.6V-Flash` | `2,3` | 8002 |

Plus the DINO embedder on CUDA, 8 Chromium render sidecars, and `ensemble_size: 40`
parallel coders. This needs **4 large GPUs** (~4×141 GB on H200). See
[`MINER_VALIDATOR_GUIDE.md` §11.11 / "Why H200"](../../docs/MINER_VALIDATOR_GUIDE.md)
for the VRAM math and the `4xRTX6000Pro` alternative.

> **The hardware you declare must match the audit hardware.** Whatever box you
> generate CDN files on, validators re-run your Docker on the **same declared
> hardware** — different hardware can change output and fail the 0% audit.

### 1.2 Secrets and accounts

- `HF_TOKEN` — HuggingFace token with access to the gated model repos above and the
  DINO weights (`Tooony133/dinov3-vits16-pretrain-lvd1689m`). vLLM downloads weights
  on first boot; they are **not** baked into the image.
- A Bittensor **coldkey + hotkey** registered on **netuid 17** (finney).
- Conviction (α) locked on the coldkey — **≥ 100 ρ per submitting hotkey**.
- A **public CDN** you can upload `.js` files to (e.g. an S3/R2 bucket behind a URL).
- A **public GitHub repo** of this `my-agent` code with the `docker/Dockerfile`, so
  validators can build and audit the exact commit you reveal.

### 1.3 Repo gaps to close before first go-live

The current `my-agent/` tree is missing two things required for Level 3:

1. **`hardware.json`** at the repo root declaring your audit hardware, e.g.:
   ```json
   { "hardware": ["4xH200"] }
   ```
2. A **GitHub Actions Docker build** workflow so the competition build tracker can
   verify your image builds from the revealed commit.

---

## 2. Build and run the production container

```bash
cd my-agent

# 1. Build
docker build -f docker/Dockerfile -t my-404-miner .

# 2. Run (all 4 GPUs, expose the batch API)
docker run --gpus all \
  -p 10006:10006 \
  -e HF_TOKEN=hf_xxx \
  -e CONFIG_PATH=/workspace/configuration.yaml \
  my-404-miner
```

What happens inside (`pipeline_service/run.sh`):

1. **Preflight** (`python -m modules.metrics.preflight`) benchmarks GPU TFLOPS/VRAM +
   network. On failure it starts the API in **REPLACE** mode and skips vLLM (signals
   the orchestrator to swap the pod).
2. Starts **`serve.py`** (FastAPI) in the background on `:10006`.
3. Builds the GLM vLLM env if missing (`scripts/setup_glm_vllm_env.sh`).
4. Spawns the two **vLLM** servers (`python -m llm.spawn`) on `:8001` / `:8002`.
5. Warmup generation runs, then `/status` flips to `ready` (log line `models up — ready`).

> First boot downloads ~tens of GB of weights from HuggingFace and can take a long
> time — this is expected (the batch budget is 4 h). Pre-warm the HF cache volume
> (`docker-compose.yml` mounts cache volumes) so restarts are fast.

### 2.1 Verify the container is healthy

```bash
curl http://localhost:10006/health            # -> 200
curl http://localhost:10006/status            # -> {"status":"ready",...} once warmed
curl http://localhost:8001/v1/models          # coder (Qwen) up
curl http://localhost:8002/v1/models          # judge/critic (GLM) up
```

Do not proceed to a live round until `/status` is `ready` and both `/v1/models`
respond.

---

## 3. The per-round live loop

### 3.1 Watch the competition state

The competition state lives in the public repo `404-Repo/404-active-competition`.
Poll `state.json` for the current stage:

```bash
curl -s https://raw.githubusercontent.com/404-Repo/404-active-competition/main/state.json
```

Stages: `OPEN → MINER_GENERATION → DOWNLOADING → DUELS → FINALIZING`.
Per-round block timing is in `rounds/{N}/schedule.json`; block time ≈ 12 s.

### 3.2 Reveal on-chain (during the reveal window)

During `OPEN`, between `earliest_reveal_block` and `latest_reveal_block`, commit your
submission pointer. **Revealing earlier gives higher tournament priority.**

```python
import json, bittensor as bt
wallet = bt.Wallet(name="my_coldkey", hotkey="my_hotkey")
subtensor = bt.Subtensor(network="finney")
subtensor.commit(wallet=wallet, netuid=17, data=json.dumps({
    "repo":    "YOUR_ORG/my-404-miner",       # public repo with docker/Dockerfile
    "commit":  "40_char_git_sha",             # EXACT code that makes your CDN files
    "cdn_url": "https://your-cdn.example.com/round-25",
}))
```

> The revealed `commit` **must** be the exact code that generated your CDN files.
> Mismatched code → failed audit.

### 3.3 Generate the round (when stage = MINER_GENERATION)

After `latest_reveal_block`, validators publish the round inputs. Fetch them:

```bash
BASE=https://raw.githubusercontent.com/404-Repo/404-active-competition/main/rounds/25
curl -s $BASE/seed.json      # -> {"seed": <int>}
curl -s $BASE/prompts.txt    # 128 image URLs, one per line
```

Then drive the batch API the **same way validators audit** — 128 prompts, and use the
**published seed**. You can submit all 128 at once or in 4 batches of 32 (audit uses
4×32). Each prompt item is `{stem, image_url}` where `stem` is the URL filename
without extension.

Easiest: reuse the bundled harness (it POSTs `/generate`, polls `/status`, saves
`{stem}.js`):

```bash
# turn the URL list into the local stem<TAB>url format, then run the harness
python - <<'PY'
from urllib.parse import urlparse; from pathlib import Path
urls=[u.strip() for u in open("prompts.txt") if u.strip()]
open("round.txt","w").write("".join(f"{Path(urlparse(u).path).stem}\t{u}\n" for u in urls))
PY

cd pipeline_service
python tests/test_pipeline.py ../round.txt \
  --host localhost --port 10006 \
  --seed "$ROUND_SEED" \
  --name round25 --out-dir ../runs --timeout 14400
```

Or hit the API directly:

```bash
curl -X POST http://localhost:10006/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompts":[{"stem":"<stem>","image_url":"<url>"}, ...], "seed": <ROUND_SEED>}'
curl http://localhost:10006/status                 # poll until "complete"
curl -o results.zip http://localhost:10006/results # ZIP of {stem}.js
```

### 3.4 Validate every file before upload

```bash
for f in runs/round25/*.js; do
  node ../miner-reference/tools/validate.js "$f" || echo "INVALID: $f"
done
```

Any file that fails validation will render as a failure for that prompt and auto-lose
its duels — fix or regenerate before uploading.

### 3.5 Upload to your CDN (before the deadline)

Upload each file so it is reachable at exactly `{cdn_url}/{stem}.js` — this is the URL
validators pull (`submission_collector/download.py`: `f"{cdn_url.rstrip('/')}/{prompt}.js"`).

```bash
# example: aws s3 sync to the bucket behind your cdn_url
aws s3 cp runs/round25/ s3://your-bucket/round-25/ --recursive --exclude "*" --include "*.js"
```

Everything must be uploaded before `generation_deadline_block`
(`generation_stage_minutes` ≈ 240 min / 4 h).

### 3.6 If you win the timeline → audit

If your submission is a winner candidate, validators redeploy **your Docker image**
(from the revealed `repo`+`commit`) and re-run the batch API with the **same seed**.
The regenerated files are compared to your CDN files with a **0% margin**: if your
submitted files are *better* than the Docker reproduces, you **fail**. So:

- Generate CDN files **with this Docker image on the declared hardware** — not via
  OpenRouter, and not hand-edited. (The `local-eval/run-generate-cdn.sh` OpenRouter
  path is for R&D only and is explicitly unsafe for live rounds.)
- Keep the revealed commit == the code that produced the files.

---

## 4. Tuning the time budget

The 128-prompt run must finish inside the generation window (and the audit window).
`ensemble_size` is by far the biggest cost/time lever. If a full 128-batch is too
slow, lower these in `configuration.yaml` (in order of impact):

| Knob | Effect |
|------|--------|
| `actors.coder.ensemble_size` (40) | Fewer parallel coders per prompt — largest speedup |
| `actors.coder.max_model_len` | Shorter context = faster/cheaper |
| `actors.coder.max_num_seqs` | Lower concurrency, less VRAM pressure |
| `llm_clients.*.gpu_memory_utilization` | Fit within VRAM |
| `pipeline.refinement_enabled` | Currently `false`; turning it **on** improves quality but costs time — A/B it first (see below) |

> Whatever you change, **regenerate and re-audit-check** afterwards: the config in the
> revealed commit is the config validators will run.

---

## 5. Validate quality before you go live

Use the local duel harness to confirm your pipeline changes actually beat the baseline
before spending a live round on them:

```bash
# logic-only duel (both sides on the same VLM) — see local-eval/
./local-eval/run-duel-seq.sh pool --limit 20 --fresh
```

For a production-faithful signal, run the duel with the **Docker/vLLM** outputs (the
models baked into the image), not the OpenRouter dev profile.

---

## 6. Quick reference

| Thing | Value |
|-------|-------|
| Batch API port | `10006` |
| vLLM coder / judge ports | `8001` / `8002` |
| Entry point | `pipeline_service/run.sh` (Docker `CMD`) |
| Prompts per round | 128 image URLs |
| Audit batching | 4 × 32, same seed |
| Generation window | ~240 min (4 h) |
| Block time | ~12 s |
| Audit margin | **0%** (submitted must not beat Docker-regenerated) |
| Conviction lock | ≥ 100 ρ per hotkey |
| State to watch | `404-Repo/404-active-competition` → `state.json`, `rounds/{N}/` |

---

## 7. Pre-flight checklist before every live round

- [ ] `hardware.json` present and matches the box you generate on
- [ ] Public repo pushed; `commit` SHA noted
- [ ] Docker builds from that commit (CI green)
- [ ] Container `ready`; `/v1/models` up on 8001 and 8002
- [ ] On-chain reveal submitted in the reveal window (repo + commit + cdn_url)
- [ ] 128 `.js` generated with the **published seed** via the Docker batch API
- [ ] Every file passes `miner-reference/tools/validate.js`
- [ ] All files uploaded to `{cdn_url}/{stem}.js` before `generation_deadline_block`
- [ ] Revealed commit == code that produced the uploaded files
