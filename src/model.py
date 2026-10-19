"""Shared model construction, adapter injection, and live budget checks."""
from transformers import BartForConditionalGeneration, BartTokenizerFast
from peft import LoraConfig, PrefixTuningConfig, TaskType, get_peft_model
from src.config import ExperimentConfig, get_target_modules

PLACEMENTS = ("encoder_self_attn", "decoder_self_attn", "cross_attn")
MODES = (*PLACEMENTS, "all_attn", "prefix", "full_ft", "zero_shot")


def parameter_counts(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "trainable_pct": 100 * trainable / total}


def configure_model(base_model, config):
    if config.placement not in MODES:
        raise ValueError(f"Unknown mode {config.placement}")
    if config.placement == "zero_shot":
        base_model.requires_grad_(False)
        return base_model
    if config.placement == "full_ft":
        return base_model
    if config.placement == "prefix":
        return get_peft_model(base_model, PrefixTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=config.prefix_tokens,
            prefix_projection=False))
    # Explicit paths for isolated placements assume BART-base's symmetric architecture.
    if (base_model.config.encoder_layers, base_model.config.decoder_layers,
            base_model.config.d_model) != (6, 6, 768):
        raise ValueError("Placement experiments require BART-base: 6+6 layers, d_model=768")
    targets = get_target_modules(config.placement, config.projections)
    names = dict(base_model.named_modules())
    matched = [n for n in names if any(n == t or n.endswith('.' + t) for t in targets)]
    # Each region has six blocks; all_attn contains three regions.
    expected_modules = 6 * (3 if config.placement == "all_attn" else 1) * len(config.projections.split(','))
    if len(matched) != expected_modules:
        raise ValueError(f"Expected {expected_modules} targets, found {len(matched)}")
    expected = sum(config.lora_rank * (names[n].in_features + names[n].out_features) for n in matched)
    model = get_peft_model(base_model, LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=config.lora_rank,
        lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=targets, bias="none"))
    actual = parameter_counts(model)["trainable"]
    if actual != expected:
        raise ValueError(f"Adapter budget mismatch: expected {expected}, got {actual}")
    model.p08_target_modules = matched
    return model


def load_tokenizer_and_model(config: ExperimentConfig):
    tokenizer = BartTokenizerFast.from_pretrained(config.model_name_or_path, revision=config.model_revision)
    base = BartForConditionalGeneration.from_pretrained(
        config.model_name_or_path, revision=config.model_revision, attn_implementation="eager")
    return tokenizer, configure_model(base, config)
