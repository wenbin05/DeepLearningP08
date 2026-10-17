"""
Training pipeline for BART-base PEFT summarization experiments.
Supports multiple seeds, budget-matched LoRA placement, and evaluation.
Project P08: Abstractive Summarization via Parameter-Efficient Fine-Tuning
"""

import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from src.config import ExperimentConfig, get_target_modules
from src.evaluate import compute_rouge_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Train BART-base with targeted LoRA placement.")
    parser.add_argument("--placement", type=str, default="cross_attn",
                        choices=["all_attn", "encoder_self_attn", "decoder_self_attn", "cross_attn", "full_ft", "zero_shot"],
                        help="LoRA architectural placement or baseline mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional subset size for rapid testing.")
    parser.add_argument("--max_eval_samples", type=int, default=500, help="Evaluation sample size.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save checkpoints.")
    return parser.parse_args()

def preprocess_function(examples, tokenizer, max_src=1024, max_tgt=128):
    inputs = [doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=max_src, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples["highlights"], max_length=max_tgt, truncation=True, padding="max_length")
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    args = parse_args()
    set_seed(args.seed)
    config = ExperimentConfig(placement=args.placement)
    
    print(f"=== Starting Experiment: Placement={args.placement}, Seed={args.seed} ===")
    
    tokenizer = BartTokenizer.from_pretrained(config.model_name_or_path)
    model = BartForConditionalGeneration.from_pretrained(config.model_name_or_path)
    
    if args.placement == "full_ft":
        print("Training Mode: Full Fine-Tuning (all parameters trainable)")
    elif args.placement == "zero_shot":
        print("Evaluation Mode: Zero-Shot BART-base")
    else:
        target_mods = get_target_modules(args.placement)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_mods,
            bias="none"
        )
        model = get_peft_model(model, peft_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA injected into: {args.placement}")
        print(f"Trainable Parameters: {trainable:,} / {total:,} ({trainable/total*100:.4f}%)")
    
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    train_data = dataset["train"]
    val_data = dataset["validation"]
    
    if args.max_train_samples:
        train_data = train_data.select(range(args.max_train_samples))
    if args.max_eval_samples:
        val_data = val_data.select(range(args.max_eval_samples))
        
    train_encoded = train_data.map(lambda ex: preprocess_function(ex, tokenizer), batched=True)
    val_encoded = val_data.map(lambda ex: preprocess_function(ex, tokenizer), batched=True)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_dir, f"{args.placement}_seed{args.seed}"),
        evaluation_strategy="epoch",
        learning_rate=config.lr_full_ft if args.placement == "full_ft" else config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
        generation_num_beams=config.num_beams,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        seed=args.seed
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=val_encoded,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    if args.placement != "zero_shot":
        trainer.train()
        
    print("Evaluating final checkpoint...")
    metrics = trainer.evaluate()
    print("Evaluation Results:", metrics)

if __name__ == "__main__":
    main()
