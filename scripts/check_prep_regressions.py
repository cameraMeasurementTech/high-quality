#!/usr/bin/env python3
"""Regression checks for duel dataset-prep throughput changes (no GPU needed)."""
from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
SCRIPTS = TRAINING / "scripts"
sys.path.insert(0, str(SCRIPTS))


def check_yaml_keys() -> None:
    import yaml

    duel = yaml.safe_load(
        (TRAINING / "pipeline/configuration.h200x4-dpo-duel.yaml").read_text()
    )
    cheap = yaml.safe_load(
        (TRAINING / "pipeline/configuration.h200x4-dpo.yaml").read_text()
    )
    native = yaml.safe_load(
        (TRAINING / "pipeline/configuration.gpu-native.yaml").read_text()
    )
    judge = yaml.safe_load(
        (TRAINING / "pipeline/configuration.duel-judge.yaml").read_text()
    )

    assert duel["pipeline"].get("skip_render") is True, "duel Phase A must skip_render"
    assert duel["actors"]["coder"]["ensemble_size"] == 1
    assert duel["actors"]["coder"]["temperature"] == 0.6
    assert duel["pipeline"]["refinement_enabled"] is False
    assert cheap["pipeline"].get("skip_render") is True
    # Production-like native profile must still render by default
    assert native["pipeline"].get("skip_render", False) is False
    assert judge["renderer"]["judge_multiview"] is True
    assert judge["embedder"]["enabled"] is True
    print("  PASS  yaml workflow flags (duel skip_render, native still renders)")


def check_seed_formula() -> None:
    # Avoid importing collect_candidates (pulls tqdm); formula is one line.
    def sample_seed(base_seed: int, sample_i: int, seed_stride: int, batch_i: int) -> int:
        return base_seed + sample_i * seed_stride + batch_i

    assert sample_seed(42, 0, 1000, 0) == 42
    assert sample_seed(42, 1, 1000, 0) == 1042
    assert sample_seed(42, 0, 1000, 96) == 138
    assert sample_seed(42, 1, 1000, 96) == 1138
    # Must match collect_candidates.sample_seed
    src = (SCRIPTS / "collect_candidates.py").read_text(encoding="utf-8")
    assert "return base_seed + sample_i * seed_stride + batch_i" in src
    print("  PASS  seed diversity formula unchanged (sample_0 vs sample_1)")


def check_collect_interleave_writes() -> None:
    def sample_seed(base_seed: int, sample_i: int, seed_stride: int, batch_i: int) -> int:
        return base_seed + sample_i * seed_stride + batch_i

    batch_size = 2
    pairs = [("a", "u1"), ("b", "u2"), ("c", "u3")]
    seeds_by_sample: dict[int, set[int]] = {0: set(), 1: set()}
    for batch_i in range(0, len(pairs), batch_size):
        for sample_i in range(2):
            seeds_by_sample[sample_i].add(sample_seed(42, sample_i, 1000, batch_i))
    assert seeds_by_sample[0].isdisjoint(seeds_by_sample[1])
    assert 42 in seeds_by_sample[0] and 1042 in seeds_by_sample[1]
    print("  PASS  interleaved collect keeps sample seeds disjoint")


def check_duel_combine() -> None:
    # Inline to avoid heavy imports; must match duel_score_candidates.combine_verdicts
    def combine_verdicts(ab: str, ba: str) -> str:
        ba_norm = "B" if ba == "A" else "A"
        if ab == ba_norm:
            return ab
        return "draw"

    assert combine_verdicts("A", "B") == "A"
    assert combine_verdicts("B", "A") == "B"
    assert combine_verdicts("A", "A") == "draw"
    assert combine_verdicts("B", "B") == "draw"
    src = (SCRIPTS / "duel_score_candidates.py").read_text(encoding="utf-8")
    assert "ba_norm = \"B\" if ba == \"A\" else \"A\"" in src
    print("  PASS  duel AB/BA combine logic")


def check_settings_conf_loads() -> None:
    """SettingsConf must load for Phase A duel yaml and Phase B judge yaml."""
    try:
        import pydantic  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:
        print("  SKIP  SettingsConf load (pydantic not installed in this python)")
        return

    import os
    import subprocess

    py = sys.executable
    root = TRAINING / "pipeline"
    for name in (
        "configuration.h200x4-dpo-duel.yaml",
        "configuration.h200x4-dpo.yaml",
        "configuration.gpu-native.yaml",
        "configuration.duel-judge.yaml",
    ):
        env = os.environ.copy()
        env["CONFIG_FILE"] = str(root / name)
        env["PYTHONPATH"] = "/home/404-gen-subnet/shiny-guide/pipeline_service"
        # Prefer sibling shiny-guide; fall back via paths if needed
        code = (
            "from config.settings import settings\n"
            "print(settings.pipeline.skip_render, settings.pipeline.refinement_enabled)\n"
        )
        r = subprocess.run(
            [py, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            raise AssertionError(f"{name} failed to load SettingsConf:\n{r.stderr}")
        print(f"  PASS  SettingsConf loads {name} ({r.stdout.strip()})")



def check_pack_duel_scored_quality_filters() -> None:
    try:
        from pack_dpo_dataset import pairs_from_duel_scored
    except ImportError as exc:
        print(f"  SKIP  pack_dpo filters ({exc})")
        return

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        images = root / "images"
        images.mkdir()
        js_a = root / "a.js"
        js_b = root / "b.js"
        js_a.write_text("chosen_code_aaa\n")
        js_b.write_text("rejected_code_bbb\n")
        (images / "stem1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        duel = root / "duels.json"
        duel.write_text(
            """{
              "records": [
                {
                  "stem": "stem1",
                  "winner": "A",
                  "chosen_file": "%s",
                  "rejected_file": "%s",
                  "ab_decided_by": "s4",
                  "ba_decided_by": "s4"
                },
                {
                  "stem": "stem2",
                  "winner": "draw",
                  "chosen_file": null,
                  "rejected_file": null
                },
                {
                  "stem": "stem3",
                  "winner": "B",
                  "chosen_file": "%s",
                  "rejected_file": "%s"
                }
              ]
            }"""
            % (js_a, js_b, js_b, js_a)
        )
        # stem3 missing image → skipped missing
        rows, skipped = pairs_from_duel_scored(duel, images, None, skip_draws=True)
        assert len(rows) == 1 and rows[0]["stem"] == "stem1"
        assert rows[0]["pair_source"] == "duel_scored"
        assert skipped["draw"] >= 1
        assert skipped["missing"] >= 1
        # identical JS filtered
        same = root / "same.js"
        same.write_text("same\n")
        duel2 = root / "duels2.json"
        duel2.write_text(
            """{"records":[{"stem":"stem1","winner":"A","chosen_file":"%s","rejected_file":"%s"}]}"""
            % (same, same)
        )
        rows2, skipped2 = pairs_from_duel_scored(duel2, images, None)
        assert rows2 == [] and skipped2["filtered"] == 1
    print("  PASS  pack duel-scored keeps draws/identical out; decisive pairs in")


def check_pipeline_syntax() -> None:
    root = Path("/home/404-gen-subnet/shiny-guide/pipeline_service")
    files = [
        root / "config/settings.py",
        root / "pipeline/factory.py",
        root / "pipeline/orchestrator.py",
        root / "pipeline/generation_pipeline.py",
        root / "modules/js_checker/module.py",
        TRAINING / "scripts/collect_candidates.py",
        TRAINING / "scripts/duel_score_candidates.py",
        TRAINING / "pipeline/apply_throughput_overlays.py",
    ]
    for f in files:
        ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
    print("  PASS  python syntax for patched modules")


def check_overlays_idempotent() -> None:
    import subprocess

    r = subprocess.run(
        [
            sys.executable,
            str(TRAINING / "pipeline/apply_throughput_overlays.py"),
            "/home/404-gen-subnet/shiny-guide",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "warn" not in r.stdout.lower() or "already" in r.stdout
    print("  PASS  throughput overlays idempotent")


def main() -> int:
    print("==> Dataset-prep regression checks")
    check_yaml_keys()
    check_seed_formula()
    check_collect_interleave_writes()
    check_duel_combine()
    check_pack_duel_scored_quality_filters()
    check_pipeline_syntax()
    check_overlays_idempotent()
    check_settings_conf_loads()
    print("==> All checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
