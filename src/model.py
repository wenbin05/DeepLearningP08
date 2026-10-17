"""
Model loader and targeted LoRA configuration for BART-base.
Project P08: Abstractive Summarization via Parameter-Efficient Fine-Tuning
"""

import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from src.config import ExperimentConfig, get_target_modules

def load_tokenizer_and_model(config: ExperimentConfig, is_baseline: bool = False):
    """
    Loads facebook/bart-base and optionally injects LoRA into specified sub-modules.
    """
    tokenizer = BartTokenizer.from_pretrained(config.model_name_or_path)
    base_model = BartForConditionalGeneration.from_pretrained(config.model_name_or_path)
    
    if is_baseline:
        # Return raw model for zero-shot or full fine-tuning
        return tokenizer, base_model
        
    target_modules = get_target_modules(config.placement)
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none"
    )
    
    model = get_peft_model(base_model, peft_config)
    return tokenizer, model
