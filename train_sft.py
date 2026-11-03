#!/usr/bin/env python3
"""QLoRA fine-tune Qwen3-4B on LifeMate SFT data.

Kaggle T4 (16GB VRAM), ~2-3h for 3 epochs on 1198 samples.

    !pip install unsloth
    !python train_sft.py --data sft_train.jsonl --epochs 3 --out lora_out
"""
from __future__ import annotations

import argparse, inspect, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("sft_train.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", type=Path, default=Path("lora_out"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--ga", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    # Only truncates; the collator pads to the longest sample in each batch, so
    # lowering this saves nothing and silently cuts the CALL targets off the end.
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--r", type=int, default=32)
    a = ap.parse_args()

    from unsloth import FastLanguageModel, is_bfloat16_supported

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=a.model,
        max_seq_length=a.seq_len,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=a.r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=a.r * 2,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
    )

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    from datasets import Dataset
    dataset = Dataset.from_list(rows)
    print(f"Dataset: {len(dataset)} conversations")

    def fmt(examples):
        texts = []
        for msgs in examples["messages"]:
            kw = {"tokenize": False, "add_generation_prompt": False}
            try:
                t = tokenizer.apply_chat_template(msgs, enable_thinking=False, **kw)
            except TypeError:
                t = tokenizer.apply_chat_template(msgs, **kw)
            texts.append(t)
        return {"text": texts}

    dataset = dataset.map(fmt, batched=True, remove_columns=["messages"])

    from trl import SFTTrainer
    from transformers import TrainingArguments
    try:
        from trl import SFTConfig
    except ImportError:
        SFTConfig = None

    # transformers 5.x removes legacy fields one release at a time (warmup_ratio,
    # group_by_length, ...) and TRL moves seq-len options into SFTConfig. Pass
    # only what the installed versions actually accept instead of crashing.
    args_cls = SFTConfig or TrainingArguments
    ta = dict(
        per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=a.ga,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        group_by_length=True,  # mean seq 1612 vs max 2828 — cuts padding waste
        save_strategy="epoch",
        output_dir=str(a.out),
        seed=42,
        optim="adamw_8bit",
    )
    sig = inspect.signature(args_cls.__init__).parameters
    ta_params = set(sig)
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.values())
    for k in ("max_length", "max_seq_length"):
        if k in ta_params:
            ta[k] = a.seq_len
            break
    if "dataset_text_field" in ta_params:
        ta["dataset_text_field"] = "text"
    dropped = [] if accepts_any else sorted(set(ta) - ta_params)
    if not accepts_any:
        ta = {k: v for k, v in ta.items() if k in ta_params}
    if dropped:
        print(f"bỏ qua tham số {args_cls.__name__} không hỗ trợ: {dropped}")
    training_args = args_cls(**ta)

    tr_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    trainer_kw = {"model": model, "args": training_args, "train_dataset": dataset}
    if "processing_class" in tr_params:
        trainer_kw["processing_class"] = tokenizer
    elif "tokenizer" in tr_params:
        trainer_kw["tokenizer"] = tokenizer
    if "packing" in tr_params:
        trainer_kw["packing"] = False
    if SFTConfig is None:  # old TRL keeps these on the trainer, not the config
        if "dataset_text_field" in tr_params:
            trainer_kw["dataset_text_field"] = "text"
        if "max_seq_length" in tr_params:
            trainer_kw["max_seq_length"] = a.seq_len
    trainer = SFTTrainer(**trainer_kw)

    # Loss only on assistant turns
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("Training...")
    stats = trainer.train()
    print(f"Done — {stats.metrics['train_runtime']:.0f}s, "
          f"loss={stats.metrics['train_loss']:.4f}")

    model.save_pretrained(str(a.out))
    tokenizer.save_pretrained(str(a.out))
    print(f"LoRA adapter → {a.out}")


if __name__ == "__main__":
    main()
