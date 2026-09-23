"""LoRA fine-tuning script for Qwen3-VL 8B floor plan grounding.

Optimized for 15GB VRAM (Google Colab Free T4 GPU) using 4-bit QLoRA and Unsloth / TRL SFTTrainer.
"""

import argparse
import os
import json
from pathlib import Path
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

from floorplan_reader.models.qwen_vl_wrapper import (
    load_qwen_vl_model,
    apply_lora_to_model,
    DEFAULT_MODEL_ID,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen-VL on floor plan datasets.")
    parser.add_argument(
        "--train_file",
        type=str,
        default="./data/processed/train.jsonl",
        help="Path to train.jsonl",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="./data/processed/val.jsonl",
        help="Path to val.jsonl",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="HuggingFace model ID (e.g. Qwen/Qwen3-VL-8B-Instruct or Qwen/Qwen2.5-VL-7B-Instruct).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/floorplan_qwen3_vl_lora",
        help="Directory to save fine-tuned LoRA adapter checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device train batch size.")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (effective batch size = 8).")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank dimension.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every N steps.")
    parser.add_argument("--use_unsloth", action="store_true", default=True, help="Use Unsloth memory-optimized kernels.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Floor Plan Reader - Qwen-VL LoRA Training Pipeline")
    print("=" * 60)
    print(f"Model ID:              {args.model_id}")
    print(f"Train Dataset:         {args.train_file}")
    print(f"Output Directory:      {args.output_dir}")
    print(f"Learning Rate:         {args.lr}")
    print(f"Batch Size (Effective): {args.batch_size * args.grad_accum}")
    print(f"LoRA Rank / Alpha:     {args.lora_r} / {args.lora_alpha}")
    print("=" * 60)

    # 1. Load Dataset
    print(f"Loading datasets from {args.train_file}...")
    data_files = {"train": args.train_file}
    if os.path.exists(args.val_file):
        data_files["val"] = args.val_file

    raw_datasets = load_dataset("json", data_files=data_files)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets.get("val")

    print(f"Loaded {len(train_dataset)} training samples.")
    if eval_dataset:
        print(f"Loaded {len(eval_dataset)} evaluation samples.")

    # 2. Load Model and Processor with 4-bit Quantization
    print("\nLoading model with 4-bit quantization for T4 GPU compatibility...")
    model, processor = load_qwen_vl_model(
        model_id=args.model_id,
        load_in_4bit=True,
        use_unsloth=args.use_unsloth,
    )

    # 3. Apply LoRA Adapters
    print(f"\nInjecting LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})...")
    model = apply_lora_to_model(
        model=model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_unsloth=args.use_unsloth,
    )

    # Print trainable parameters
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    # 4. Configure Training Arguments
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="paged_adamw_8bit",
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=True,
        dataset_text_field="",  # Multimodal conversational dataset handled by collator
        report_to="none",
    )

    # 5. Initialize SFTTrainer
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer if hasattr(processor, "tokenizer") else processor,
    )

    # 6. Run Training
    print("\nStarting LoRA Fine-Tuning...")
    trainer.train()

    # 7. Save Model Adapter and Processor
    print(f"\nSaving fine-tuned LoRA weights and processor to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(args.output_dir)

    print("\nTraining successfully completed!")


if __name__ == "__main__":
    main()
