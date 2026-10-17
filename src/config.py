"""
Configuration for BART-base Parameter-Efficient Fine-Tuning (PEFT) experiments.
Project P08: Abstractive Summarization via Parameter-Efficient Fine-Tuning
"""

import dataclasses
from typing import List, Optional

@dataclasses.dataclass
class ExperimentConfig:
    # Model backbone
    model_name_or_path: str = "facebook/bart-base"
    
    # Dataset settings
    dataset_name: str = "cnn_dailymail"
    dataset_config_name: str = "3.0.0"
    max_source_length: int = 1024
    max_target_length: int = 128
    
    # LoRA hyperparameters
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Placement options: 'all_attn', 'encoder_self_attn', 'decoder_self_attn', 'cross_attn'
    placement: str = "cross_attn"
    
    # Training hyperparameters (kept strictly consistent across all configurations)
    learning_rate: float = 3e-4  # Standard for LoRA (full FT uses 3e-5)
    lr_full_ft: float = 3e-5
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 32
    per_device_eval_batch_size: int = 8
    num_train_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    fp16: bool = True
    
    # Generation parameters (consistent during evaluation)
    num_beams: int = 4
    length_penalty: float = 2.0
    min_length: int = 56
    no_repeat_ngram_size: int = 3
    
    # Reproducibility
    seeds: List[int] = dataclasses.field(default_factory=lambda: [42, 123, 999])


def get_target_modules(placement: str) -> List[str]:
    """
    Returns regex or module name list for targeted LoRA injection in BART-base.
    
    BART-base attention structure:
      - encoder.layers.{0..5}.self_attn.{q_proj, v_proj, k_proj, out_proj}
      - decoder.layers.{0..5}.self_attn.{q_proj, v_proj, k_proj, out_proj}
      - decoder.layers.{0..5}.encoder_attn.{q_proj, v_proj, k_proj, out_proj}
    """
    if placement == "all_attn":
        # Standard LoRA: all q_proj and v_proj across encoder and decoder
        return ["q_proj", "v_proj"]
    elif placement == "encoder_self_attn":
        # Only encoder self-attention
        return [f"model.encoder.layers.{i}.self_attn.{p}" for i in range(6) for p in ["q_proj", "v_proj"]]
    elif placement == "decoder_self_attn":
        # Only decoder self-attention
        return [f"model.decoder.layers.{i}.self_attn.{p}" for i in range(6) for p in ["q_proj", "v_proj"]]
    elif placement == "cross_attn":
        # Only encoder-decoder cross-attention
        return [f"model.decoder.layers.{i}.encoder_attn.{p}" for i in range(6) for p in ["q_proj", "v_proj"]]
    else:
        raise ValueError(f"Unknown placement: {placement}. Choose from 'all_attn', 'encoder_self_attn', 'decoder_self_attn', 'cross_attn'.")
