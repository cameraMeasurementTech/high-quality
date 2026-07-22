#!/usr/bin/env python3
"""GRPO training for the image→Three.js coder with validator-shaped rewards.

Usage:
  accelerate launch train_grpo.py --config ../configs/grpo_8b.yaml
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

from reward import RewardConfig, make_reward_fn


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = path.resolve().parents[1]
    for key in ("dataset_path", "output_dir", "sft_adapter_path"):
        if key in cfg and cfg[key] and not Path(cfg[key]).is_absolute():
            cfg[key] = str((root / cfg[key]).resolve())
    return cfg


def load_policy(cfg: dict):
    model_id = cfg["model_name_or_path"]
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    quant = None
    if cfg.get("load_in_4bit"):
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quant,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
        )
    except Exception:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            quantization_config=quant,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    if cfg.get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model)

    # Optional prior LoRA checkpoint; omit when base model is already production-ready (e.g. AstroWolf).
    if cfg.get("sft_adapter_path"):
        model = PeftModel.from_pretrained(model, cfg["sft_adapter_path"], is_trainable=True)
    elif cfg.get("use_lora", True):
        lora = LoraConfig(
            r=int(cfg.get("lora_r", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cfg.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    return model, processor


def prepare_dataset(ds, processor):
    """Convert packed rows into the prompt format GRPOTrainer expects for VLMs."""

    def _map(row):
        prompt = row["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        # Attach image object into message content for processor
        for msg in prompt:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    part["image"] = row["image"]
        return {
            "prompt": prompt,
            "images": [row["image"]],
            "stem": row.get("stem", ""),
        }

    return ds.map(_map, remove_columns=[c for c in ds.column_names if c not in {"prompt", "images", "stem", "image"}])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    ds = load_from_disk(cfg["dataset_path"])
    if cfg.get("max_samples"):
        ds = ds.select(range(min(int(cfg["max_samples"]), len(ds))))

    model, processor = load_policy(cfg)
    ds = prepare_dataset(ds, processor)

    reward_cfg = RewardConfig(
        mode=cfg.get("reward_mode", "cheap"),
        cache_dir=Path(cfg["output_dir"]) / "reward_cache" if cfg.get("output_dir") else None,
        render_url=os.environ.get("RENDER_URL") or cfg.get("render_url"),
        judge_base_url=os.environ.get("JUDGE_BASE_URL") or cfg.get("judge_base_url"),
        judge_model=os.environ.get("JUDGE_MODEL") or cfg.get("judge_model"),
        w_s1=float(cfg.get("w_s1", 0.45)),
        w_fmt=float(cfg.get("w_fmt", 0.05)),
    )
    reward_fn = make_reward_fn(reward_cfg)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    grpo_args = GRPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(cfg.get("learning_rate", 1e-6)),
        logging_steps=int(cfg.get("logging_steps", 5)),
        save_steps=int(cfg.get("save_steps", 100)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        bf16=True,
        report_to=cfg.get("report_to", "none"),
        remove_unused_columns=False,
        max_completion_length=int(cfg.get("max_completion_length", 12288)),
        num_generations=int(cfg.get("num_generations", 4)),
        temperature=float(cfg.get("temperature", 0.8)),
        beta=float(cfg.get("beta", 0.04)),
        use_vllm=bool(cfg.get("use_vllm", False)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=ds,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    processor.save_pretrained(str(out_dir / "final"))
    (out_dir / "train_config_used.yaml").write_text(yaml.dump(cfg), encoding="utf-8")
    print(f"GRPO complete -> {out_dir / 'final'}")


if __name__ == "__main__":
    main()
