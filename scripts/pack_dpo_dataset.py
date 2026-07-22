#!/usr/bin/env python3
"""Pack validator-scored preference pairs into a HuggingFace DPO dataset.

Sources (--source):
  candidates  Score K JS per stem; chosen=best, rejected=worst (score margin filter)
  duel        Parse local-eval duel_detailed.json; winner vs loser JS per stem
  duel-scored Multiview S1–S4 scored pairs from duel_score_candidates.py JSON
  dirs        Explicit chosen-dir vs rejected-dir flat JS folders (same stems)

Output columns for TRL DPOTrainer (VLM):
  stem, image, prompt (json), chosen (json), rejected (json), reward_chosen, reward_rejected

Usage:
  # Score-mined pairs from K OpenRouter samples
  python pack_dpo_dataset.py --source candidates \\
    --candidates-dir ../data/candidates/openai_k4 \\
    --images ../data/images --list ../data/splits/train.txt \\
    --out ../data/hf/dpo_train --reward-mode cheap

  # Duel pairs (train my-agent to beat shiny-guide on lost prompts)
  python pack_dpo_dataset.py --source duel \\
    --duel-json ../../local-eval/runs/duel/duel_detailed.json \\
    --prefer-label shiny-guide --images ../data/images \\
    --out ../data/hf/dpo_duel
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, Image
from tqdm import tqdm

from coder_prompts import load_coder_prompts
from paths import default_data_root
from reward import RewardConfig, score_one


def load_stem_map(path: Path | None) -> dict[str, str]:
    """stem -> image_url from split list."""
    urls: dict[str, str] = {}
    if path is None:
        return urls
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        urls[parts[0]] = parts[1] if len(parts) > 1 else ""
    return urls


def build_prompt_messages(system: str, user_tpl: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_tpl},
            ],
        },
    ]


def assistant_message(js: str) -> list[dict]:
    return [{"role": "assistant", "content": [{"type": "text", "text": js}]}]


def score_candidates_for_stem(
    stem_dir: Path,
    cfg: RewardConfig,
    image_path: str | None,
) -> list[tuple[str, float, Path]]:
    scored: list[tuple[str, float, Path]] = []
    for js_path in sorted(stem_dir.glob("sample_*.js")):
        js = js_path.read_text(encoding="utf-8").strip()
        if "export default function generate" not in js:
            continue
        r = score_one(js, cfg, image_path)
        scored.append((js, r, js_path))
    return scored


def pairs_from_candidates(
    candidates_dir: Path,
    images_dir: Path,
    allow_stems: set[str] | None,
    cfg: RewardConfig,
    min_margin: float,
    require_chosen_valid: bool,
    min_candidates: int,
) -> tuple[list[dict], dict]:
    system, user_tpl = load_coder_prompts()
    rows: list[dict] = []
    skipped = {"few_candidates": 0, "low_margin": 0, "missing_img": 0, "invalid_chosen": 0}

    for stem_dir in tqdm(sorted(candidates_dir.iterdir()), desc="pair-candidates"):
        if not stem_dir.is_dir():
            continue
        stem = stem_dir.name
        if allow_stems is not None and stem not in allow_stems:
            continue
        img_path = images_dir / f"{stem}.png"
        if not img_path.is_file():
            skipped["missing_img"] += 1
            continue

        scored = score_candidates_for_stem(stem_dir, cfg, str(img_path))
        if len(scored) < min_candidates:
            skipped["few_candidates"] += 1
            continue

        scored.sort(key=lambda x: x[1], reverse=True)
        chosen_js, r_chosen, chosen_path = scored[0]
        rejected_js, r_rejected, rejected_path = scored[-1]

        if r_chosen - r_rejected < min_margin:
            skipped["low_margin"] += 1
            continue
        if require_chosen_valid and r_chosen <= -1.5:
            skipped["invalid_chosen"] += 1
            continue
        if chosen_js.strip() == rejected_js.strip():
            skipped["low_margin"] += 1
            continue

        prompt = build_prompt_messages(system, user_tpl)
        rows.append(
            {
                "stem": stem,
                "image": str(img_path.resolve()),
                "prompt": prompt,
                "chosen": assistant_message(chosen_js),
                "rejected": assistant_message(rejected_js),
                "reward_chosen": r_chosen,
                "reward_rejected": r_rejected,
                "pair_source": "candidates",
                "chosen_file": str(chosen_path),
                "rejected_file": str(rejected_path),
            }
        )

    return rows, skipped


def pairs_from_duel(
    duel_json: Path,
    images_dir: Path,
    allow_stems: set[str] | None,
    prefer_label: str,
    only_losses: bool,
) -> tuple[list[dict], dict]:
    """Build pairs where chosen=teacher/winner JS, rejected=student/loser JS.

    prefer_label: label whose JS becomes *chosen* when it wins the duel
      (e.g. ``shiny-guide`` → train on leader-beats-challenger pairs).
    only_losses: keep only stems where prefer_label won (teacher signal on hard prompts).
    """
    system, user_tpl = load_coder_prompts()
    payload = json.loads(duel_json.read_text(encoding="utf-8"))
    a_label = payload.get("a_label", "A")
    b_label = payload.get("b_label", "B")
    a_dir = Path(payload["a_dir"])
    b_dir = Path(payload["b_dir"])

    label_to_paths = {
        a_label: a_dir,
        b_label: b_dir,
    }

    rows: list[dict] = []
    skipped = {"missing_js": 0, "missing_img": 0, "draw": 0, "filtered": 0}

    for rec in tqdm(payload.get("records", []), desc="pair-duel"):
        stem = rec["stem"]
        if allow_stems is not None and stem not in allow_stems:
            continue
        winner = rec.get("winner")
        if winner in {"draw", "DRAW", None}:
            skipped["draw"] += 1
            continue

        a_js = a_dir / f"{stem}.js"
        b_js = b_dir / f"{stem}.js"
        img_path = images_dir / f"{stem}.png"
        if not a_js.is_file() or not b_js.is_file():
            skipped["missing_js"] += 1
            continue
        if not img_path.is_file():
            skipped["missing_img"] += 1
            continue

        winner_label = a_label if winner == "A" else b_label
        loser_label = b_label if winner == "A" else a_label

        if prefer_label:
            if only_losses and winner_label != prefer_label:
                skipped["filtered"] += 1
                continue
            if winner_label == prefer_label:
                chosen_path = label_to_paths[prefer_label] / f"{stem}.js"
                other = b_label if prefer_label == a_label else a_label
                rejected_path = label_to_paths[other] / f"{stem}.js"
                pair_kind = f"{prefer_label}_wins"
            else:
                # prefer_label lost — still usable if only_losses is False
                chosen_path = label_to_paths[winner_label] / f"{stem}.js"
                rejected_path = label_to_paths[loser_label] / f"{stem}.js"
                pair_kind = f"{winner_label}_wins"
        else:
            chosen_path = a_js if winner == "A" else b_js
            rejected_path = b_js if winner == "A" else a_js
            pair_kind = "duel_winner"

        chosen_js = chosen_path.read_text(encoding="utf-8").strip()
        rejected_js = rejected_path.read_text(encoding="utf-8").strip()
        if chosen_js.strip() == rejected_js.strip():
            skipped["filtered"] += 1
            continue

        prompt = build_prompt_messages(system, user_tpl)
        rows.append(
            {
                "stem": stem,
                "image": str(img_path.resolve()),
                "prompt": prompt,
                "chosen": assistant_message(chosen_js),
                "rejected": assistant_message(rejected_js),
                "reward_chosen": None,
                "reward_rejected": None,
                "pair_source": pair_kind,
                "chosen_file": str(chosen_path),
                "rejected_file": str(rejected_path),
                "duel_winner": winner_label,
            }
        )

    return rows, skipped


def pairs_from_duel_scored(
    duel_json: Path,
    images_dir: Path,
    allow_stems: set[str] | None,
    skip_draws: bool = True,
) -> tuple[list[dict], dict]:
    """Build DPO pairs from multiview duel scores (chosen_file / rejected_file per stem)."""
    system, user_tpl = load_coder_prompts()
    payload = json.loads(duel_json.read_text(encoding="utf-8"))
    rows: list[dict] = []
    skipped = {"draw": 0, "missing": 0, "filtered": 0}

    for rec in tqdm(payload.get("records", []), desc="pair-duel-scored"):
        stem = rec["stem"]
        if allow_stems is not None and stem not in allow_stems:
            continue
        if skip_draws and rec.get("winner") in {"draw", None}:
            skipped["draw"] += 1
            continue
        chosen_path = rec.get("chosen_file")
        rejected_path = rec.get("rejected_file")
        if not chosen_path or not rejected_path:
            skipped["draw"] += 1
            continue
        chosen_path = Path(chosen_path)
        rejected_path = Path(rejected_path)
        img_path = images_dir / f"{stem}.png"
        if not chosen_path.is_file() or not rejected_path.is_file() or not img_path.is_file():
            skipped["missing"] += 1
            continue

        chosen_js = chosen_path.read_text(encoding="utf-8").strip()
        rejected_js = rejected_path.read_text(encoding="utf-8").strip()
        if chosen_js == rejected_js:
            skipped["filtered"] += 1
            continue

        prompt = build_prompt_messages(system, user_tpl)
        rows.append(
            {
                "stem": stem,
                "image": str(img_path.resolve()),
                "prompt": prompt,
                "chosen": assistant_message(chosen_js),
                "rejected": assistant_message(rejected_js),
                "reward_chosen": None,
                "reward_rejected": None,
                "pair_source": "duel_scored",
                "chosen_file": str(chosen_path),
                "rejected_file": str(rejected_path),
                "duel_winner": rec.get("winner"),
                "ab_decided_by": rec.get("ab_decided_by"),
                "ba_decided_by": rec.get("ba_decided_by"),
            }
        )

    return rows, skipped


def pairs_from_dirs(
    chosen_dir: Path,
    rejected_dir: Path,
    images_dir: Path,
    allow_stems: set[str] | None,
    cfg: RewardConfig | None,
    min_margin: float,
) -> tuple[list[dict], dict]:
    system, user_tpl = load_coder_prompts()
    rows: list[dict] = []
    skipped = {"missing": 0, "low_margin": 0}

    stems = {p.stem for p in chosen_dir.glob("*.js")}
    stems &= {p.stem for p in rejected_dir.glob("*.js")}
    if allow_stems is not None:
        stems &= allow_stems

    for stem in tqdm(sorted(stems), desc="pair-dirs"):
        c_path = chosen_dir / f"{stem}.js"
        r_path = rejected_dir / f"{stem}.js"
        img_path = images_dir / f"{stem}.png"
        if not c_path.is_file() or not r_path.is_file() or not img_path.is_file():
            skipped["missing"] += 1
            continue
        chosen_js = c_path.read_text(encoding="utf-8").strip()
        rejected_js = r_path.read_text(encoding="utf-8").strip()
        if chosen_js.strip() == rejected_js.strip():
            skipped["missing"] += 1
            continue

        r_chosen = r_rejected = None
        if cfg is not None:
            r_chosen = score_one(chosen_js, cfg, str(img_path))
            r_rejected = score_one(rejected_js, cfg, str(img_path))
            if r_chosen - r_rejected < min_margin:
                skipped["low_margin"] += 1
                continue

        prompt = build_prompt_messages(system, user_tpl)
        rows.append(
            {
                "stem": stem,
                "image": str(img_path.resolve()),
                "prompt": prompt,
                "chosen": assistant_message(chosen_js),
                "rejected": assistant_message(rejected_js),
                "reward_chosen": r_chosen,
                "reward_rejected": r_rejected,
                "pair_source": "dirs",
                "chosen_file": str(c_path),
                "rejected_file": str(r_path),
            }
        )

    return rows, skipped


def save_dataset(rows: list[dict], out: Path, meta_extra: dict) -> None:
    if not rows:
        raise SystemExit("no DPO pairs packed — relax filters or collect more candidates")

    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "dpo.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "stem": row["stem"],
                        "image": row["image"],
                        "prompt": row["prompt"],
                        "chosen": row["chosen"],
                        "rejected": row["rejected"],
                        "reward_chosen": row.get("reward_chosen"),
                        "reward_rejected": row.get("reward_rejected"),
                        "pair_source": row.get("pair_source"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    ds_rows = [
        {
            "stem": r["stem"],
            "image": r["image"],
            "prompt": json.dumps(r["prompt"]),
            "chosen": json.dumps(r["chosen"]),
            "rejected": json.dumps(r["rejected"]),
            "reward_chosen": r.get("reward_chosen"),
            "reward_rejected": r.get("reward_rejected"),
            "pair_source": r.get("pair_source", ""),
        }
        for r in rows
    ]
    ds = Dataset.from_list(ds_rows)
    ds = ds.cast_column("image", Image())
    ds.save_to_disk(str(out / "dataset"))

    meta = {
        "n_pairs": len(rows),
        "jsonl": str(jsonl_path),
        "dataset": str(out / "dataset"),
        **meta_extra,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack DPO preference dataset")
    ap.add_argument("--source", required=True, choices=["candidates", "duel", "duel-scored", "dirs"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--list", type=Path, default=None, help="optional stem allow-list")
    ap.add_argument("--images", type=Path, default=default_data_root() / "images")

    ap.add_argument("--candidates-dir", type=Path, default=None)
    ap.add_argument("--duel-json", type=Path, default=None)
    ap.add_argument("--chosen-dir", type=Path, default=None)
    ap.add_argument("--rejected-dir", type=Path, default=None)
    ap.add_argument("--prefer-label", type=str, default="", help="e.g. shiny-guide for teacher pairs")
    ap.add_argument("--only-losses", action="store_true", help="duel: only stems prefer-label lost")
    ap.add_argument("--include-draws", action="store_true", help="duel-scored: keep draw pairs")

    ap.add_argument("--reward-mode", type=str, default="cheap", choices=["cheap", "render", "s1"])
    ap.add_argument("--min-margin", type=float, default=0.15)
    ap.add_argument("--min-candidates", type=int, default=2)
    ap.add_argument("--require-chosen-valid", action="store_true", default=True)
    ap.add_argument("--allow-invalid-chosen", action="store_false", dest="require_chosen_valid")
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args()

    allow = set(load_stem_map(args.list).keys()) if args.list else None

    reward_cfg = RewardConfig(
        mode=args.reward_mode,
        cache_dir=args.cache_dir,
        render_url=os.environ.get("RENDER_URL"),
        judge_base_url=os.environ.get("JUDGE_BASE_URL"),
        judge_model=os.environ.get("JUDGE_MODEL"),
    )

    if args.source == "candidates":
        if not args.candidates_dir:
            raise SystemExit("--candidates-dir required for source=candidates")
        rows, skipped = pairs_from_candidates(
            args.candidates_dir,
            args.images,
            allow,
            reward_cfg,
            args.min_margin,
            args.require_chosen_valid,
            args.min_candidates,
        )
        meta_extra = {"source": "candidates", "skipped": skipped, "reward_mode": args.reward_mode}
    elif args.source == "duel":
        if not args.duel_json:
            raise SystemExit("--duel-json required for source=duel")
        rows, skipped = pairs_from_duel(
            args.duel_json,
            args.images,
            allow,
            args.prefer_label,
            args.only_losses,
        )
        meta_extra = {"source": "duel", "skipped": skipped, "duel_json": str(args.duel_json)}
    elif args.source == "duel-scored":
        if not args.duel_json:
            raise SystemExit("--duel-json required for source=duel-scored")
        rows, skipped = pairs_from_duel_scored(
            args.duel_json,
            args.images,
            allow,
            skip_draws=not args.include_draws,
        )
        meta_extra = {
            "source": "duel-scored",
            "skipped": skipped,
            "duel_json": str(args.duel_json),
            "judge": "multiview S1-S4 via OpenRouter",
        }
    else:
        if not args.chosen_dir or not args.rejected_dir:
            raise SystemExit("--chosen-dir and --rejected-dir required for source=dirs")
        rows, skipped = pairs_from_dirs(
            args.chosen_dir,
            args.rejected_dir,
            args.images,
            allow,
            reward_cfg,
            args.min_margin,
        )
        meta_extra = {"source": "dirs", "skipped": skipped}

    save_dataset(rows, args.out, meta_extra)


if __name__ == "__main__":
    main()
