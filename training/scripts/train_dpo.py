#!/usr/bin/env python3
"""DPO training for the image→Three.js coder VLM.

Uses TRL DPOTrainer with the same production coder prompts as SFT/GRPO (train=serve).

Usage:
  accelerate launch train_dpo.py --config ../configs/dpo_8b.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from trl import DPOConfig, DPOTrainer


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = path.resolve().parents[1]
    for key in ("dataset_path", "output_dir", "sft_adapter_path", "dpo_adapter_path"):
        if key in cfg and cfg[key] and not Path(cfg[key]).is_absolute():
            cfg[key] = str((root / cfg[key]).resolve())
    return cfg


def load_model_and_processor(cfg: dict):
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

    adapter_path = cfg.get("dpo_adapter_path") or cfg.get("sft_adapter_path")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    elif cfg.get("use_lora", True):
        lora = LoraConfig(
            r=int(cfg.get("lora_r", 32)),
            lora_alpha=int(cfg.get("lora_alpha", 64)),
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


def prepare_dataset(ds):
    """Inject PIL images into prompt messages for TRL vision DPO."""

    def _map(row):
        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        if isinstance(chosen, str):
            chosen = json.loads(chosen)
        if isinstance(rejected, str):
            rejected = json.loads(rejected)

        for msg in prompt:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    part["image"] = row["image"]

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "images": [row["image"]],
            "stem": row.get("stem", ""),
        }

    drop_cols = [c for c in ds.column_names if c not in {"prompt", "chosen", "rejected", "image", "stem"}]
    return ds.map(_map, remove_columns=drop_cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    ds = load_from_disk(cfg["dataset_path"])
    if cfg.get("max_samples"):
        ds = ds.select(range(min(int(cfg["max_samples"]), len(ds))))

    model, processor = load_model_and_processor(cfg)
    ds = prepare_dataset(ds)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # max_length=None is required for VLMs — truncating drops image tokens.
    dpo_args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 5e-7)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=int(cfg.get("save_steps", 100)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        bf16=True,
        beta=float(cfg.get("beta", 0.1)),
        max_length=None,
        max_prompt_length=None,
        max_completion_length=int(cfg.get("max_completion_length", 12288)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        report_to=cfg.get("report_to", "none"),
        remove_unused_columns=False,
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 2)),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=ds,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    processor.save_pretrained(str(out_dir / "final"))
    (out_dir / "train_config_used.yaml").write_text(yaml.dump(cfg), encoding="utf-8")
    print(f"DPO complete -> {out_dir / 'final'}")


if __name__ == "__main__":
    main()
