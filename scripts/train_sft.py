#!/usr/bin/env python3
"""LoRA / QLoRA SFT for the image→Three.js coder VLM.

Usage:
  accelerate launch train_sft.py --config ../configs/sft_8b.yaml
  # or
  python train_sft.py --config ../configs/sft_8b.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from paths import resolve_model_path


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve relative dataset/output paths against training root (parent of configs/).
    root = path.resolve().parents[1]
    for key in ("dataset_path", "output_dir", "sft_adapter_path"):
        if key in cfg and cfg[key] and not Path(cfg[key]).is_absolute():
            cfg[key] = str((root / cfg[key]).resolve())
    cfg["model_name_or_path"] = resolve_model_path(cfg["model_name_or_path"])
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

    # Prefer Qwen2.5-VL class; fall back to AutoModel for other VLMs.
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

    if cfg.get("use_lora", True):
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
    elif not cfg.get("use_lora", True):
        for p in model.parameters():
            p.requires_grad = True
        model.print_trainable_parameters()

    if cfg.get("use_lora", True) and not cfg.get("load_in_4bit"):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model, processor


def build_collator(processor, max_length: int, mask_prompt: bool = True):
    """Collate HF rows {image, messages(json)} into model inputs with labels.

    When mask_prompt=True (default), only assistant tokens contribute to loss —
    system/user (including image tokens) are set to -100.
    """

    def collate(batch):
        texts: list[str] = []
        prompt_texts: list[str] = []
        images = []
        for row in batch:
            messages = row["messages"]
            if isinstance(messages, str):
                messages = json.loads(messages)
            # Inject actual image into the user content placeholder.
            for msg in messages:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        part.clear()
                        part["type"] = "image"
                        part["image"] = row["image"]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(row["image"])
            if mask_prompt:
                prompt_only = [m for m in messages if m.get("role") != "assistant"]
                prompt_texts.append(
                    processor.apply_chat_template(
                        prompt_only, tokenize=False, add_generation_prompt=True
                    )
                )

        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        pad_id = processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        if mask_prompt and prompt_texts:
            prompt_tok = processor.tokenizer(
                prompt_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_lens = (prompt_tok["attention_mask"].sum(dim=1)).tolist()
            for i, plen in enumerate(prompt_lens):
                # Mask leading prompt tokens; leave assistant completion for loss.
                n = min(int(plen), labels.shape[1])
                labels[i, :n] = -100

        inputs["labels"] = labels
        return inputs

    return collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    ds = load_from_disk(cfg["dataset_path"])
    if cfg.get("max_samples"):
        ds = ds.select(range(min(int(cfg["max_samples"]), len(ds))))

    model, processor = load_model_and_processor(cfg)
    collator = build_collator(
        processor,
        int(cfg.get("max_seq_length", 16384)),
        mask_prompt=bool(cfg.get("mask_prompt", True)),
    )

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=int(cfg.get("save_steps", 200)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        bf16=True,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        report_to=cfg.get("report_to", "none"),
        remove_unused_columns=False,
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 2)),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    processor.save_pretrained(str(out_dir / "final"))
    (out_dir / "train_config_used.yaml").write_text(yaml.dump(cfg), encoding="utf-8")
    print(f"SFT complete -> {out_dir / 'final'}")


if __name__ == "__main__":
    main()
