"""Verify real BART-base parameter counts using meta tensors without downloading pretrained weights."""
import argparse
import gc
import json
import torch
from transformers import BartConfig, BartForConditionalGeneration
from src.config import ExperimentConfig
from src.model import MODES, PLACEMENTS, configure_model, parameter_counts


def bart_base_config():
    return BartConfig(encoder_layers=6, decoder_layers=6, d_model=768,
                      encoder_attention_heads=12, decoder_attention_heads=12,
                      encoder_ffn_dim=3072, decoder_ffn_dim=3072)


def verify_live_model():
    counts = {}
    for mode in MODES:
        with torch.device("meta"):
            model = configure_model(BartForConditionalGeneration(bart_base_config()), ExperimentConfig(placement=mode))
        counts[mode] = parameter_counts(model)
        del model
        gc.collect()
    assert {counts[p]['trainable'] for p in PLACEMENTS} == {147456}
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Retained alias; all counts are live')
    parser.parse_args()
    print(json.dumps(verify_live_model(), indent=2))
