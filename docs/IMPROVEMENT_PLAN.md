# my-agent Quality Improvement Plan

Concrete plan to make `my-agent` generate higher-quality Three.js that beats the current leader in duels, while staying within miner requirements (validator-safe output, reproducible Docker, open-source models in production).

Read alongside [PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md).

---

## 0. Miner requirements this plan must respect

| Requirement | Constraint |
|-------------|-----------|
| Output contract | `export default function generate(THREE)`, synchronous, no imports, fits in `[-0.5,0.5]^3`, Y-up/+Z |
| Hard limits | <= 250k verts, <= 200 draw calls, <= 5s exec, <= 1MB file, <= 50KB literals |
| Reproducibility | Public repo + commit + `docker/Dockerfile` on `:10006`; audit reruns 128 prompts, same seed |
| Models (production) | Commercial-use open-source only (Qwen / GLM), pinned versions |
| Scoring | Quality matters, speed does not (as long as the deadline is met) |

OpenRouter frontier models are used only for CPU R&D; the shipped Docker miner must use the GPU vLLM config with open-source models.

---

## 1. Changes already applied in this branch

### 1a. Turn on dormant quality levers (CPU profile)

`local-eval/configuration.my-agent.cpu-openrouter.yaml`:

- `refinement_enabled: true` — enables critic -> repair. This is the **single biggest lever** and is OFF in the shipped `configuration.yaml`.
- `event_bus.score_threshold: 0.6` — keeps refinement working on near-misses instead of stopping at 0.55.
- `ensemble_size: 3` on CPU (cost bound); keep 40 on GPU.
- Planner fully configured and one flag (`use_planner: true`) away from an A/B.

Recommended for the shipped GPU `configuration.yaml`: set `refinement_enabled: true` and evaluate `use_planner: true`. These are the two highest-ROI switches and cost nothing to flip.

### 1b. Prompt improvements

- `modules/scene_coder/prompts.py`: added a **Multi-view & duel awareness** section — build real 3D depth (side/back views are judged), enforce grounding (lowest point near `y=-0.5`, centered), silhouette-first priority.
- `modules/critic/prompts.py`: added **score calibration across repair iterations** — use full 0.01 resolution, move the score when a repair fixes/regresses an issue, anchor score to remaining-issue count. Fixes the observed failure where `score_history` was a flat `[0.45, 0.45, 0.45]`, which gave refinement no signal.
- `modules/scene_planner/prompts.py`: added a **Quality bar** — numeric proportions per part, a PBR material per part, mandatory `count_hint`, explicit front/+Z orientation.

---

## 2. Better VLM models (config-driven sweep)

Every `actors.*.model` in the CPU profile is a knob. The duel harness (Section 3) measures the effect of changing one.

### How to sweep without breaking the duel's fairness

- To measure **pipeline + prompt** improvements: keep `coder.model` identical to the shiny-guide opponent (`google/gemini-2.5-pro-preview`). Any win is attributable to pipeline/prompt changes.
- To measure **model** improvements: change only `coder.model` (or `critic.model`) and re-run the duel; compare win-rate deltas.

### Candidate models

| Role | Candidate | Notes |
|------|-----------|-------|
| Coder (R&D) | `google/gemini-2.5-pro-preview` | Baseline; matches opponent |
| Coder (R&D) | `anthropic/claude-sonnet-4` | Strong spatial decomposition |
| Coder (R&D) | `openai/gpt-5.1` | Strong multimodal code generation |
| Coder (production) | `qwen/qwen3-vl-235b-instruct` or the shipped `Tooony133/Qwen-3.6-27B-AstroWolf` | Open-weight, commercial-use OK |
| Critic/Judge | separate from coder model | A different model gives less self-favoring critiques |
| Critic/Judge (production) | `zai-org/GLM-4.6V-Flash` | Shipped choice; matches subnet judge family |

### Procedure

1. Establish a baseline duel win-rate with the default profile (Section 3).
2. Change one model id, re-run `./local-eval/run-duel.sh pool --limit N`.
3. Keep the change only if win-rate improves beyond noise (see N guidance in the duel README).
4. For any model kept for production, confirm it is available as open-source in the GPU `configuration.yaml` and pin its revision.

---

## 3. Duel-driven evaluation loop

Use the A/B duel harness ([local-eval/README.md](../../local-eval/README.md), `run-duel.sh` + `scripts/duel_pipelines.py`) to make every change measurable:

```mermaid
flowchart LR
  change["Change config / prompt / model"] --> gen["Generate both pipelines<br/>same N prompts"]
  gen --> duel["Multi-stage VLM duel<br/>AB + BA mirrored"]
  duel --> rate["Win-rate vs shiny-guide"]
  rate --> keep{"Improved?"}
  keep -->|yes| change
  keep -->|no| revert["Revert"]
  revert --> change
```

Target: my-agent win-rate > 50% vs shiny-guide on a representative prompt set, then port the winning config/prompts to the GPU `configuration.yaml`.

---

## 4. Backlog (higher effort, not yet applied)

| Idea | Rationale |
|------|-----------|
| Increase `ensemble_size` on GPU beyond 40 for hard categories | More bracket candidates -> better front match |
| Category-conditioned coder prompt (detect object class first, inject only the relevant handbook) | Shorter, more focused context per prompt |
| Add a second critic pass focused only on side/back views | Aligns with judge side-guard stage |
| Cache/reuse planner OSD across ensemble members | Consistent decomposition, lower cost |
| Per-category `score_threshold` | Spend more repair on historically weak categories |
