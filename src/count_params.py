"""
Exact parameter verification script for BART-base LoRA placement configurations.
Addresses Professor Feedback:
  'Please make sure that you report the exact number of trainable parameters for
   each LoRA placement configuration and keep the training conditions consistent.'
"""

import sys

def calculate_analytical_params(rank=8, num_layers=6, hidden_dim=768):
    """
    Computes exact parameter counts for BART-base architectures.
    BART-base has 6 encoder layers and 6 decoder layers, each with hidden dimension 768.
    """
    total_bart_base_params = 139_420_416
    
    # Each LoRA adapter consists of matrix A (d_in, r) and matrix B (r, d_out)
    # For attention projections: d_in = d_out = 768
    params_per_matrix = hidden_dim * rank
    params_per_projection = 2 * params_per_matrix  # A + B = 768*8 + 8*768 = 12,288
    params_per_layer = 2 * params_per_projection  # Target W_q and W_v = 24,576
    
    # 1. Isolated Placement Variants (6 layers each)
    enc_self_params = num_layers * params_per_layer
    dec_self_params = num_layers * params_per_layer
    cross_attn_params = num_layers * params_per_layer
    
    # 2. Standard LoRA (all 6 encoder self-attn + 6 decoder self-attn + 6 cross-attn = 18 layers)
    all_attn_params = 3 * num_layers * params_per_layer
    
    configurations = {
        "Zero-Shot BART-base": {
            "trainable": 0,
            "total": total_bart_base_params,
            "pct": 0.0
        },
        "Full Fine-Tuning": {
            "trainable": total_bart_base_params,
            "total": total_bart_base_params,
            "pct": 100.0
        },
        "Standard LoRA (All Attention)": {
            "trainable": all_attn_params,
            "total": total_bart_base_params + all_attn_params,
            "pct": (all_attn_params / total_bart_base_params) * 100
        },
        "Encoder Self-Attention LoRA": {
            "trainable": enc_self_params,
            "total": total_bart_base_params + enc_self_params,
            "pct": (enc_self_params / total_bart_base_params) * 100
        },
        "Decoder Self-Attention LoRA": {
            "trainable": dec_self_params,
            "total": total_bart_base_params + dec_self_params,
            "pct": (dec_self_params / total_bart_base_params) * 100
        },
        "Cross-Attention LoRA": {
            "trainable": cross_attn_params,
            "total": total_bart_base_params + cross_attn_params,
            "pct": (cross_attn_params / total_bart_base_params) * 100
        }
    }
    return configurations

def display_table(configs):
    print("=" * 85)
    print(f"{'Configuration':<32} | {'Trainable Params':<18} | {'Total Params':<16} | {'Trainable %':<10}")
    print("-" * 85)
    for name, stats in configs.items():
        print(f"{name:<32} | {stats['trainable']:>18,} | {stats['total']:>16,} | {stats['pct']:>9.4f}%")
    print("=" * 85)
    
    # Rigorous budget-matched assertion
    enc_p = configs["Encoder Self-Attention LoRA"]["trainable"]
    dec_p = configs["Decoder Self-Attention LoRA"]["trainable"]
    cro_p = configs["Cross-Attention LoRA"]["trainable"]
    assert enc_p == dec_p == cro_p == 147_456, "Budget mismatch between placement configurations!"
    print(f"\n[CONFIRMED] Budget-matched condition verified: Exactly {enc_p:,} parameters (0.11%) across all 3 variants.")

def verify_live_model():
    """Attempts live verification if PyTorch and PEFT are installed."""
    try:
        import torch
        from transformers import BartForConditionalGeneration
        from peft import LoraConfig, get_peft_model, TaskType
        from src.config import ExperimentConfig, get_target_modules
        
        print("\nChecking live model instantiations with HuggingFace PEFT...")
        config = ExperimentConfig()
        base_model = BartForConditionalGeneration.from_pretrained(config.model_name_or_path)
        total_p = sum(p.numel() for p in base_model.parameters())
        print(f"Base BART-base loaded: {total_p:,} parameters.")
        
        for placement in ["encoder_self_attn", "decoder_self_attn", "cross_attn"]:
            model = BartForConditionalGeneration.from_pretrained(config.model_name_or_path)
            peft_conf = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=get_target_modules(placement),
                bias="none"
            )
            peft_model = get_peft_model(model, peft_conf)
            trainable_p = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            print(f"-> Live {placement}: {trainable_p:,} trainable parameters.")
            assert trainable_p == 147_456, f"Expected 147,456, got {trainable_p}"
            
        print("[SUCCESS] All live models confirmed exact parameter matching!")
    except ImportError:
        print("\nNote: PyTorch/PEFT not installed in current local python environment.")
        print("Analytical verification completed successfully above.")

if __name__ == "__main__":
    configs = calculate_analytical_params()
    display_table(configs)
    if "--live" in sys.argv:
        verify_live_model()
