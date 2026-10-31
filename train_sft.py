#!/usr/bin/env python3
"""QLoRA fine-tune Qwen3-4B on LifeMate SFT data.

Kaggle T4 (16GB VRAM), ~2-3h for 3 epochs on 1198 samples.

    !pip install unsloth
    !python train_sft.py --data sft_train.jsonl --epochs 3 --out lora_out
"""
from __future__ import annotations

import argparse, json
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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=a.seq_len,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=a.bs,
            gradient_accumulation_steps=a.ga,
            num_train_epochs=a.epochs,
            learning_rate=a.lr,
            lr_scheduler_type="cosine",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            save_strategy="epoch",
            output_dir=str(a.out),
            seed=42,
            optim="adamw_8bit",
        ),
    )

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
