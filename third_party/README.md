# Bundled subnet validator for standalone training.

This directory contains a minimal copy of `miner-reference` so training does **not**
depend on the full 404-gen-subnet repo:

- `tools/validate.js` — CLI used by DPO scoring and GRPO rewards
- `validator/` — npm package (`@404-subnet/validator`)

Setup (also run by `./run/00_setup.sh`):

```bash
cd third_party/miner-reference/validator && npm install
node ../tools/validate.js --help   # run from tools/ dir (ESM)
```

Requires **Node.js ≥ 20**.
