#!/usr/bin/env python3
"""LoRA SFT training script for the digital-human companion task.

Runs on Migu Tongxue platform (2x RTX 4090) inside the competition vLLM image
(which includes pytorch; install peft/datasets if missing).

Model: Qwen/Qwen2.5-7B-Instruct (or local path)
Data: train_sft.jsonl (messages format)
Output: adapter merged into ./model (final model dir for inference)

Usage:
  python train_lora.py --model_path Qwen/Qwen2.5-7B-Instruct \
      --train_file train_sft.jsonl --val_file val_sft.jsonl --out ./model
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train_file", default="train_sft.jsonl")
    ap.add_argument("--val_file", default="val_sft.jsonl")
    ap.add_argument("--out", default="./model")
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--val_samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_steps", type=int, default=400)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    return ap.parse_args()


class SFTDataset(Dataset):
    """Renders messages with the model chat template, masks prompt tokens."""

    def __init__(self, rows, tokenizer, max_len):
        self.data = []
        skipped = 0
        for r in rows:
            msgs = r["messages"]
            try:
                prompt_ids = tokenizer.apply_chat_template(
                    msgs[:-1], tokenize=True, add_generation_prompt=True
                )
                full_ids = tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=False
                )
            except Exception:
                skipped += 1
                continue
            if len(full_ids) > max_len:
                skipped += 1
                continue
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            self.data.append({"input_ids": full_ids, "labels": labels})
        print(f"dataset: {len(self.data)} usable samples ({skipped} skipped)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    # MUST return dict: Trainer feeds tensors to model.forward(input_ids, attention_mask, labels)
    # a tuple here would silently reorder fields and crash compute_loss
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    train_rows = load_jsonl(args.train_file)
    val_rows = load_jsonl(args.val_file)[: args.val_samples]
    if args.max_samples:
        train_rows = train_rows[: args.max_samples]

    train_ds = SFTDataset(train_rows, tokenizer, args.max_len)
    val_ds = SFTDataset(val_rows, tokenizer, args.max_len)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if _has_fa2() else "sdpa",
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    # gradient checkpointing: mandatory for single 24G card with long sequences
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    eff_bs = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(train_ds) / eff_bs)
    print(f"train samples={len(train_ds)}  eff_batch={eff_bs}  steps/epoch={steps_per_epoch}  total={int(steps_per_epoch*args.epochs)}")
    total_steps = int(steps_per_epoch * args.epochs)

    targs = TrainingArguments(
        output_dir="./ckpt",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    class TqdmCallback(TrainerCallback):
        pass

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda b: collate(b, pad_id),
    )
    trainer.train()

    # merge & save final model
    print("merging LoRA adapter...")
    model = model.merge_and_unload()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    print(f"final model saved to {args.out}")


def _has_fa2():
    try:
        import flash_attn  # noqa
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
