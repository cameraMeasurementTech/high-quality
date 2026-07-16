# Local smoke test (round prompts → generate → validate → R2)

Use this to prove the `high-quality` pipeline can produce valid `{stem}.js`
files and that your R2 CDN naming works — without competing in a live round.

Round **24** inputs are fetched from
[404-active-competition](https://github.com/404-Repo/404-active-competition).

## One-time setup

```bash
cd /home/high-quality

# Python helpers (upload + harness deps)
source ~/.local/bin/env   # uv on PATH
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python boto3 httpx

# Secrets
cp .env.example .env
# edit .env: HF_TOKEN, R2_*, CDN_PUBLIC_BASE

# Start the miner API — must reach /status == ready
#
# This workspace is a nested K8s pod: `docker run --gpus all` fails with
#   nvidia-container-cli … bpf_prog_query … operation not permitted
# Use bare metal on the host GPUs instead:
bash scripts/native_setup.sh          # once (builds .venvs + node_modules)
bash scripts/native_run.sh            # starts :10006 + vLLM on GPUs 0-3
#
# On a real VM/bare-metal host (not this pod), Docker works:
#   docker run --gpus all -p 10006:10006 \
#     -e HF_TOKEN -e CONFIG_PATH=/workspace/configuration.yaml my-404-miner
#   (CONFIG_PATH must be the *in-container* path /workspace/…, not the host path)
```

## Commands

```bash
# 1) Download round 24 seed + prompts (+ build round.txt / round.smoke.txt)
bash scripts/fetch_round.sh 24

# 2) End-to-end smoke (2 prompts only) — needs API on :10006
bash scripts/smoke_round.sh 24 --smoke

# 3) Same, then upload to R2 under prefix round-24-smoke
bash scripts/smoke_round.sh 24 --smoke --upload

# 4) Larger local check (first 5 of 128)
bash scripts/smoke_round.sh 24 --limit 5 --upload

# 5) Full round 24 (long; ~4h budget class)
bash scripts/smoke_round.sh 24 --upload
```

Individual steps:

```bash
# Generate only (API must be up)
.venv/bin/python tests/test_pipeline.py rounds/24/round.smoke.txt \
  --host localhost --port 10006 \
  --seed "$(.venv/bin/python -c 'import json;print(json.load(open("rounds/24/seed.json"))["seed"])')" \
  --name round24-smoke --out-dir runs --timeout 14400

# Validate only
bash scripts/validate_js.sh runs/round24-smoke/upload

# Upload only (dry-run / real)
.venv/bin/python scripts/upload_r2.py runs/round24-smoke/upload --prefix round-24-smoke --dry-run
.venv/bin/python scripts/upload_r2.py runs/round24-smoke/upload --prefix round-24-smoke
```

## R2 naming (must match live rounds)

| Piece | Example |
|-------|---------|
| Object key | `round-24-smoke/<stem>.js` |
| Revealed `cdn_url` | `https://cdn.example.com/round-24-smoke` |
| Validator fetch | `{cdn_url}/{stem}.js` |

`stem` = basename of the prompt image URL without extension.

## Layout after a run

```text
rounds/24/
  seed.json          # {"seed": 2996246273}
  prompts.txt        # 128 image URLs
  round.txt          # stem<TAB>url
  round.smoke.txt    # first 2 prompts
runs/round24-smoke/
  <stem>.js / .png / …
  upload/            # .js only — what goes to R2
```

## Notes

- OpenRouter / CPU paths in `image-three.js-localeval/local-eval` are for R&D;
  live CDN files must come from this Docker/vLLM pipeline (0% audit margin).
- Validator binary: `image-three.js-localeval/miner-reference/tools/validate.js`
- If Docker cannot run in this workspace, use bare-metal `pipeline_service/run.sh`.
