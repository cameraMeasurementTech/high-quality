# Shiny-guide training — standalone

Everything lives in this folder: **bootstrap → install → data prep → DPO/GRPO train → merge → eval**.

**Start here:** [`STANDALONE.md`](STANDALONE.md)

## One command (fresh GPU machine)

```bash
cd training
cp .env.template .env          # set OPENROUTER_API_KEY + HF_TOKEN
chmod +x run/*.sh pipeline/*.sh

./run/run_all.sh               # full pipeline
# or smoke test:
SMOKE=1 ./run/run_all.sh
```

`run_all.sh` does:

1. **Bootstrap assets** — clone [shiny-guide](https://github.com/mokabetrade/shiny-guide), download `prompts.txt`, download `Tooony133/Qwen-3.6-27B-AstroWolf`
2. **Install all** — validator npm, training venv + CUDA torch, pipeline venv + vLLM
3. **Start pipeline** — native GPU shiny-guide on `:10006`
4. **Prepare datasets** — DPO pairs + GRPO prompts from validator pool
5. **Train** — DPO or GRPO (set `TRAIN=grpo` or `TRAIN=both`)

## Step-by-step

```bash
./run/00_bootstrap_assets.sh   # clone + prompts + model
./run/00_install_all.sh          # packages (INSTALL_SYSTEM=1 for apt deps)
source .env
./pipeline/start-native-bg.sh && ./pipeline/wait-ready.sh
ALIGN=dpo TRAIN_N=5000 ./run/01_prepare_shiny_align.sh
CONFIG=configs/dpo_shiny_27b.yaml ./run/03_dpo.sh
./run/04_merge_and_eval.sh
```

Base model: `Tooony133/Qwen-3.6-27B-AstroWolf` — skip SFT.
