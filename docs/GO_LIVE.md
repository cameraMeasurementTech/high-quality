# Miner Go-Live Runbook (404-GEN Subnet 17)

End-to-end guide for competing as a miner: prepare the public Git repo, reveal
on-chain, download the round’s image prompts, generate Three.js modules, upload
them to your R2 (CDN) bucket with the required naming, and survive audit.

Competition state (read-only):  
[https://github.com/404-Repo/404-active-competition](https://github.com/404-Repo/404-active-competition)

Raw base URL:

```text
https://raw.githubusercontent.com/404-Repo/404-active-competition/main/
```

This tree (`high-quality/` / `my-agent/`) is your production pipeline. Specs and
validator details live in [`../../docs/MINER_VALIDATOR_GUIDE.md`](../../docs/MINER_VALIDATOR_GUIDE.md).

---

## 0. Mental model (read this once)

There is **no** long-running Bittensor axon that receives prompts. You operate an
**ops loop** each round:

```text
OPEN
  ├─ prepare & push public Git repo (Dockerfile + hardware.json + CI)
  ├─ decide CDN prefix (R2 public URL for this round)
  └─ on-chain reveal: {repo, commit, cdn_url}   ← earlier block = higher priority
        │
        ▼  (after latest_reveal_block)
MINER_GENERATION
  ├─ download seed.json + prompts.txt (128 image URLs)
  ├─ generate {stem}.js with your pipeline (same seed; Docker-reproducible)
  ├─ validate every file locally
  └─ upload to {cdn_url}/{stem}.js before generation_deadline_block
        │
        ▼
DOWNLOADING → DUELS → (if winner-candidate) Docker audit at 0% margin
```

**Critical timing:** `seed.json` / `prompts.txt` are **not** published until the
reveal window closes. You must reveal a **code commit** and a **CDN URL prefix**
*during* `OPEN`, then generate and fill that CDN *during* `MINER_GENERATION`.

Validators fetch exactly:

```text
{cdn_url.rstrip('/')}/{stem}.js
```

(`submission_collector/download.py`). Your miner R2 is the CDN; validators later
re-host copies on *their* R2 for duels — that is not your upload target.

---

## 1. One-time setup (Phase A)

### 1.1 Accounts and locks

| Item | Requirement |
|------|-------------|
| Wallet | Coldkey + hotkey registered on **netuid 17** (finney) |
| Conviction | ≥ **100 ρ** locked on the coldkey per submitting hotkey (insufficient lock → reveal ignored) |
| HuggingFace | `HF_TOKEN` with access to coder / GLM / DINO gated repos |
| Public GitHub | Miner code with `docker/Dockerfile` (build context = repo root) |
| CDN | Cloudflare R2 (or any HTTPS static host) with a **public read** URL |

### 1.2 Public Git repository layout

Do **not** commit competition submissions inside `404-gen-subnet`. Use **your**
public miner repo (this `high-quality` / `my-agent` code). Minimum:

```text
your-miner-repo/
├── docker/
│   └── Dockerfile                 # REQUIRED
├── hardware.json                  # {"hardware": ["4xH200"]}
├── configuration.yaml             # pinned pipeline config
├── pipeline_service/              # FastAPI + vLLM spawn + run.sh
├── .github/workflows/
│   └── docker-build.yml           # MUST build docker/Dockerfile successfully
└── …
```

`hardware.json` example (must match the box you generate on — audit redeploys the same class):

```json
{ "hardware": ["4xH200"] }
```

Supported tokens: `4xH200` (default), `4xRTX6000Pro`.

### 1.3 Prepare Git before every round you intend to change code

```bash
cd /path/to/your-miner-repo

# 1. Make sure CI is green on the commit you will reveal
git status
git add -A
git commit -m "Round-ready: pin config and pipeline for round N"
git push origin HEAD

# 2. Record the FULL 40-char SHA (required on-chain)
COMMIT=$(git rev-parse HEAD)
echo "$COMMIT"   # must be 40 lowercase hex chars

# 3. Confirm remote is public and matches what you will reveal
#    repo field format: owner/name  (e.g. cameraMeasurementTech/high-quality)
```

Rules:

- `commit` on-chain must be the **exact** SHA that produced the CDN `.js` files.
- Do not hand-edit CDN outputs after generation (audit margin is **0%**).
- GitHub Actions must successfully build `docker/Dockerfile` from that commit
  (orchestrator tracks builds in `rounds/{N}/builds.json`).

### 1.4 Cloudflare R2 bucket (your miner CDN)

You need an R2 bucket with **public access** (custom domain or `r2.dev` public URL).

Example layout (recommended: one prefix per round):

```text
s3://my-404-miner-cdn/
  round-25/
    <stem1>.js
    <stem2>.js
    …
```

Public CDN base (what goes in `cdn_url` — **no trailing slash**, **no** filename):

```text
https://cdn.your-domain.com/round-25
```

or:

```text
https://pub-xxxx.r2.dev/round-25
```

Validators will GET:

```text
https://cdn.your-domain.com/round-25/<stem>.js
```

**Naming convenience (required):**

| Rule | Detail |
|------|--------|
| Object key | `{prefix}/{stem}.js` where `stem` = prompt URL path basename without extension |
| Content-Type | `application/javascript` (or `text/javascript`) |
| Visibility | Public HTTPS GET without auth |
| Count | One file per line in `rounds/{N}/prompts.txt` (128 typical) |
| Partial OK | Missing stems = automatic duel losses for those prompts |

Stem example:

```text
https://…/f9d89484ebe9a80aa4f41ab4fdc5ba3a91216c51ac3452281ba12fbd23d66ea2.png
→ stem = f9d89484ebe9a80aa4f41ab4fdc5ba3a91216c51ac3452281ba12fbd23d66ea2
→ upload key = round-25/f9d89484ebe9a80aa4f41ab4fdc5ba3a91216c51ac3452281ba12fbd23d66ea2.js
```

Configure AWS-compatible CLI once:

```bash
# ~/.aws/credentials  (profile name arbitrary)
[r2]
aws_access_key_id = <R2_ACCESS_KEY_ID>
aws_secret_access_key = <R2_SECRET_ACCESS_KEY>

# endpoint = https://<ACCOUNT_ID>.r2.cloudflarestorage.com
export AWS_PROFILE=r2
export R2_ENDPOINT="https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
export R2_BUCKET="my-404-miner-cdn"
export CDN_PUBLIC_BASE="https://cdn.your-domain.com"   # public URL for the bucket
```

---

## 2. Per-round schedule (Phase B)

### 2.1 Watch competition state

```bash
COMP=https://raw.githubusercontent.com/404-Repo/404-active-competition/main

curl -s "$COMP/state.json" | jq .
# Example:
# { "current_round": 25, "stage": "open", "next_stage_eta": "…" }
```

Stages: `OPEN` → `MINER_GENERATION` → `DOWNLOADING` → `DUELS` → `FINALIZING`.

```bash
ROUND=$(curl -s "$COMP/state.json" | jq -r .current_round)

curl -s "$COMP/rounds/${ROUND}/schedule.json" | jq .
# {
#   "earliest_reveal_block": …,
#   "latest_reveal_block": …,
#   "generation_deadline_block": …
# }
```

Block time ≈ **12 s**. Generation window ≈ **240 min** (`generation_stage_minutes`).

| Window | Action |
|--------|--------|
| `earliest_reveal_block` … `latest_reveal_block` | On-chain reveal (`repo` + `commit` + `cdn_url`) |
| After `latest_reveal_block` | Seed/prompts published; stage → `MINER_GENERATION` |
| Until `generation_deadline_block` | Generate + upload all `{stem}.js` |

---

## 3. On-chain reveal (during OPEN)

Reveal **as early as possible** in the window (earlier reveal block → higher tournament priority). Files do not need to exist on the CDN yet — only the **prefix** must be what you will upload to.

### 3.1 Choose `cdn_url` for this round

```bash
ROUND=25
CDN_URL="${CDN_PUBLIC_BASE}/round-${ROUND}"   # e.g. https://cdn.example.com/round-25
# NO trailing slash
```

### 3.2 Commit payload

```python
import json
import bittensor as bt

wallet = bt.Wallet(name="my_coldkey", hotkey="my_hotkey")
subtensor = bt.Subtensor(network="finney")

payload = json.dumps({
    "repo": "YOUR_ORG/your-miner-repo",          # owner/name
    "commit": "0123456789abcdef0123456789abcdef01234567",  # full 40-char SHA
    "cdn_url": "https://cdn.your-domain.com/round-25",
})

subtensor.commit(wallet=wallet, netuid=17, data=payload)
```

Field rules (`submission_collector/submission.py`):

| Field | Rule |
|-------|------|
| `repo` | `^[\w-]+/[\w-]+$` |
| `commit` | 40-char lowercase hex |
| `cdn_url` | HTTPS; trailing `/` stripped |

Latest value **per field** wins if you send multiple commits in the window. All three fields must be present (can be across commits).

Verify you appear after OPEN closes:

```bash
curl -s "$COMP/rounds/${ROUND}/submissions.json" | jq .
```

---

## 4. Download round prompts (when stage = MINER_GENERATION)

```bash
COMP=https://raw.githubusercontent.com/404-Repo/404-active-competition/main
ROUND=$(curl -s "$COMP/state.json" | jq -r .current_round)
STAGE=$(curl -s "$COMP/state.json" | jq -r .stage)
echo "round=$ROUND stage=$STAGE"

# Wait until stage is miner_generation / MINER_GENERATION (exact casing in JSON)
mkdir -p "rounds/${ROUND}" && cd "rounds/${ROUND}"

curl -sO "$COMP/rounds/${ROUND}/seed.json"
curl -sO "$COMP/rounds/${ROUND}/prompts.txt"

cat seed.json          # {"seed": <int>}
wc -l prompts.txt      # expect 128
head -3 prompts.txt
```

Build the harness input (`stem<TAB>url`), used by the batch client:

```bash
python3 - <<'PY'
from pathlib import Path
from urllib.parse import urlparse

urls = [u.strip() for u in open("prompts.txt") if u.strip()]
Path("round.txt").write_text(
    "".join(f"{Path(urlparse(u).path).stem}\t{u}\n" for u in urls)
)
print(f"wrote round.txt ({len(urls)} prompts)")
PY

ROUND_SEED=$(python3 -c 'import json; print(json.load(open("seed.json"))["seed"])')
export ROUND_SEED
echo "ROUND_SEED=$ROUND_SEED"
```

---

## 5. Generate Three.js files

Use the **same seed** and the **same code commit** you revealed. Prefer the
production Docker/vLLM path (audit-faithful). OpenRouter / CPU paths are R&D only
and are **unsafe** for live CDN uploads (0% audit margin).

### 5.1 Start the miner service (Docker — preferred)

```bash
cd /path/to/your-miner-repo   # revealed commit checked out

docker build -f docker/Dockerfile -t my-404-miner .

docker run --gpus all \
  -p 10006:10006 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e CONFIG_PATH=/workspace/configuration.yaml \
  my-404-miner
```

### 5.2 Start without Docker (bare metal)

If Docker-in-Docker is unavailable on your box:

```bash
cd pipeline_service
export CONFIG_PATH="$(pwd)/../configuration.yaml"
export CONFIG_FILE="$CONFIG_PATH"
export HF_TOKEN=hf_xxx
# Ensure vllm bins match configuration.yaml (coder + GLM envs)
./run.sh
```

### 5.3 Wait until ready

```bash
curl -sf http://localhost:10006/health
curl -s  http://localhost:10006/status          # wait until "ready"
curl -sf http://localhost:8001/v1/models        # coder
curl -sf http://localhost:8002/v1/models        # judge/critic
```

### 5.4 Drive the batch API (same protocol as audit)

Audit uses **4 × 32** prompts with the round seed. You may submit all 128 at once
or in matching batches.

**Option A — bundled harness** (saves `{stem}.js` under an out dir):

```bash
cd pipeline_service
python tests/test_pipeline.py ../rounds/${ROUND}/round.txt \
  --host localhost --port 10006 \
  --seed "$ROUND_SEED" \
  --name "round${ROUND}" \
  --out-dir ../runs \
  --timeout 14400
```

**Option B — raw HTTP:**

```bash
# Build JSON with stem + image_url for each line, then:
curl -X POST http://localhost:10006/generate \
  -H 'Content-Type: application/json' \
  -d "{\"prompts\":[...],\"seed\":${ROUND_SEED}}"

# Poll
curl -s http://localhost:10006/status

# Download ZIP of {stem}.js
curl -o results.zip http://localhost:10006/results
unzip -d "runs/round${ROUND}" results.zip
```

Expect ~tens of GB of HF weights on first boot; plan within the 4 h generation
window.

---

## 6. Validate every file before upload

```bash
OUT="runs/round${ROUND}"
VAL="miner-reference/tools/validate.js"   # path in 404-gen-subnet / your copy

fail=0
for f in "$OUT"/*.js; do
  node "$VAL" "$f" || { echo "INVALID: $f"; fail=1; }
done
test "$fail" -eq 0

# Count must match prompts
echo "js=$(ls -1 "$OUT"/*.js | wc -l) prompts=$(wc -l < rounds/${ROUND}/prompts.txt)"
```

Any invalid / missing stem auto-loses its duels. Do not upload broken files.

---

## 7. Upload to R2 with required naming

```bash
ROUND=25
OUT="runs/round${ROUND}"
PREFIX="round-${ROUND}"          # object key prefix inside the bucket
CDN_URL="${CDN_PUBLIC_BASE}/${PREFIX}"

# Sanity: every stem from prompts.txt has a matching .js
python3 - <<PY
from pathlib import Path
from urllib.parse import urlparse
out = Path("${OUT}")
urls = [u.strip() for u in open("rounds/${ROUND}/prompts.txt") if u.strip()]
stems = [Path(urlparse(u).path).stem for u in urls]
missing = [s for s in stems if not (out / f"{s}.js").is_file()]
print(f"prompts={len(stems)} files={len(list(out.glob('*.js')))} missing={len(missing)}")
if missing:
    raise SystemExit("missing: " + ", ".join(missing[:5]) + ("…" if len(missing)>5 else ""))
PY

# Upload (AWS CLI → R2). Keys must be {prefix}/{stem}.js
aws s3 sync "$OUT/" "s3://${R2_BUCKET}/${PREFIX}/" \
  --endpoint-url "$R2_ENDPOINT" \
  --exclude "*" --include "*.js" \
  --content-type "application/javascript" \
  --acl public-read

# Spot-check public GETs (must be 200)
stem=$(python3 -c "from pathlib import Path; from urllib.parse import urlparse; u=open('rounds/${ROUND}/prompts.txt').readline().strip(); print(Path(urlparse(u).path).stem)")
curl -sI "${CDN_URL}/${stem}.js" | head -5
```

`cdn_url` revealed on-chain must equal this public prefix (no trailing slash):

```text
https://cdn.your-domain.com/round-25
```

Deadline: finish **before** `generation_deadline_block`. After that, collectors
enter `DOWNLOADING` and fetch your URLs.

---

## 8. End-to-end checklist (copy per round)

### Before OPEN reveal

- [ ] Public repo pushed; CI builds `docker/Dockerfile` green
- [ ] `hardware.json` matches the GPUs you will use
- [ ] `COMMIT=$(git rev-parse HEAD)` recorded (40-char)
- [ ] R2 public prefix chosen: `CDN_URL=https://…/round-{N}`
- [ ] Hotkey registered; conviction (≥100 ρ) locked
- [ ] On-chain reveal submitted **inside** reveal window (earlier = better)

### During MINER_GENERATION

- [ ] `seed.json` + `prompts.txt` downloaded for current round
- [ ] Service `/status` = `ready`; `:8001` and `:8002` up
- [ ] All 128 `.js` generated with **published** `ROUND_SEED`
- [ ] Generated with the **revealed commit** (no extra local patches)
- [ ] Every file passes `validate.js`
- [ ] Uploaded to `{cdn_url}/{stem}.js`; sample `curl -I` returns 200
- [ ] Finished before `generation_deadline_block`

### After upload

- [ ] Do **not** hand-edit CDN files
- [ ] Watch `rounds/{N}/submissions.json` / `{hotkey}/submitted.json` for fetch failures
- [ ] If you become a winner candidate: audit redeploys **your** Docker at the
      revealed commit — must reproduce CDN quality at **0%** margin

---

## 9. Common failures

| Symptom | Cause | Fix |
|---------|--------|-----|
| Reveal ignored | Outside window or low conviction | Lock α; reveal in `[earliest, latest]` |
| `404` on download | Wrong stem / prefix / private bucket | Public GET `{cdn_url}/{stem}.js` |
| Wrong files scored | Trailing slash or nested path mismatch | `cdn_url` is directory prefix only |
| Audit fail (0% margin) | OpenRouter / hand-edits / wrong commit | Generate only with revealed Docker code |
| CI / no pod | Dockerfile build red | Fix Actions; confirm `builds.json` |
| Timeouts | Ensemble too large | Lower `ensemble_size` / context; re-benchmark full 128 |

---

## 10. Quick reference

| Thing | Value |
|-------|-------|
| Competition repo | `404-Repo/404-active-competition` |
| Netuid | `17` (finney) |
| Prompts / round | 128 (`prompts_per_round`) |
| Generation window | ~240 min |
| Batch API | `:10006` — `/health`, `/status`, `/generate`, `/results` |
| vLLM ports | coder `:8001`, judge/critic `:8002` |
| CDN object | `{cdn_url}/{stem}.js` |
| Audit batching | 4 × 32, same seed |
| Audit margin | **0%** submitted vs Docker-regenerated |
| Conviction | ≥ 100 ρ / hotkey |

---

## 11. Local smoke test (before a live round)

Prove generate → validate → R2 upload with a past round (e.g. **24**) before competing:

```bash
cd /home/404-gen-subnet/my-agent
bash scripts/fetch_round.sh 24
# API must be ready on :10006
bash scripts/smoke_round.sh 24 --smoke          # 2 prompts
bash scripts/smoke_round.sh 24 --smoke --upload # + R2
```

Full details: [`LOCAL_SMOKE.md`](LOCAL_SMOKE.md).

---

## 12. Related docs

| Doc | Use |
|-----|-----|
| [`LOCAL_SMOKE.md`](LOCAL_SMOKE.md) | Round-24 local generate / validate / R2 smoke |
| [`../../docs/MINER_VALIDATOR_GUIDE.md`](../../docs/MINER_VALIDATOR_GUIDE.md) | Full subnet / API / output specs |
| [`PIPELINE_WORKFLOW.md`](PIPELINE_WORKFLOW.md) | Internal pipeline stages |
| [`../miner-reference/AGENTS.md`](../miner-reference/AGENTS.md) | LLM-facing output constraints |
| [`../miner-reference/api_specification.md`](../miner-reference/api_specification.md) | Verification batch API contract |
| [`../../local-eval/run-generate-cdn.sh`](../../local-eval/run-generate-cdn.sh) | OpenRouter CDN prototype (**not** live-safe) |
